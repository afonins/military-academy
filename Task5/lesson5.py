import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class MilitaryVehicleGenerator:
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.classes = {
            0: 'T-72',
            1: 'BTR-82A', 
            2: 'Ural-4320'
        }
        
    def add_noise(self, image, sigma=0.1):
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def generate_background(self):
        bg_type = random.choice(['grass', 'sand', 'urban', 'forest'])
        
        if bg_type == 'grass':
            base = np.array([0.2, 0.4, 0.15])
            noise = np.random.normal(0, 0.05, (self.img_size, self.img_size, 3))
            bg = base + noise
        elif bg_type == 'sand':
            base = np.array([0.7, 0.6, 0.4])
            noise = np.random.normal(0, 0.03, (self.img_size, self.img_size, 3))
            bg = base + noise
        elif bg_type == 'urban':
            base = np.array([0.4, 0.4, 0.4])
            noise = np.random.normal(0, 0.08, (self.img_size, self.img_size, 3))
            bg = base + noise
        else:
            base = np.array([0.1, 0.25, 0.1])
            noise = np.random.normal(0, 0.06, (self.img_size, self.img_size, 3))
            bg = base + noise
            
        return np.clip(bg, 0, 1)
    
    def draw_rect(self, draw, center, size, angle, color, outline=None):
        cx, cy = center
        w, h = size
        angle_rad = np.radians(angle)
        
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        rotated = []
        for x, y in corners:
            rx = x * np.cos(angle_rad) - y * np.sin(angle_rad) + cx
            ry = x * np.sin(angle_rad) + y * np.cos(angle_rad) + cy
            rotated.append((rx, ry))
        
        draw.polygon(rotated, fill=color, outline=outline)
        return rotated
    
    def draw_tank(self, img_array, center, angle, scale=1.0):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        body_color = (38, 51, 38)
        turret_color = (51, 64, 51)
        gun_color = (25, 38, 25)
        
        w, h = int(30 * scale), int(15 * scale)
        self.draw_rect(draw, (cx, cy), (w, h), angle, body_color)
        
        r = int(9 * scale)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=turret_color)
        
        rad = np.radians(angle)
        gun_len = int(20 * scale)
        x2 = int(cx + gun_len * np.cos(rad))
        y2 = int(cy + gun_len * np.sin(rad))
        draw.line([(cx, cy), (x2, y2)], fill=gun_color, width=max(2, int(3*scale)))
        
        r2 = int(4 * scale)
        draw.ellipse([cx-r2-2, cy-r2-2, cx+r2-2, cy+r2-2], fill=(25, 30, 25))
        
        return np.array(img).astype(np.float32) / 255.0
    
    def draw_btr(self, img_array, center, angle, scale=1.0):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        body_color = (46, 56, 46)
        wheel_color = (13, 13, 13)
        turret_color = (56, 66, 56)
        
        w, h = int(28 * scale), int(12 * scale)
        self.draw_rect(draw, (cx, cy), (w, h), angle, body_color)
        
        rad = np.radians(angle)
        perp = rad + np.pi/2
        
        for i in range(4):
            offset = (i - 1.5) * 6 * scale
            wx = cx + offset * np.cos(rad)
            wy = cy + offset * np.sin(rad)
            for sign in [-1, 1]:
                wheel_x = int(wx + sign * 5 * scale * np.cos(perp))
                wheel_y = int(wy + sign * 5 * scale * np.sin(perp))
                r = int(2 * scale)
                draw.ellipse([wheel_x-r, wheel_y-r, wheel_x+r, wheel_y+r], fill=wheel_color)
        
        r = int(6 * scale)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=turret_color)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def draw_truck(self, img_array, center, angle, scale=1.0):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        cab_color = (64, 51, 38)
        body_color = (38, 46, 30)
        wheel_color = (13, 13, 13)
        
        rad = np.radians(angle)
        
        cab_w, cab_h = int(10 * scale), int(8 * scale)
        cab_cx = cx - int(8 * scale * np.cos(rad))
        cab_cy = cy - int(8 * scale * np.sin(rad))
        self.draw_rect(draw, (cab_cx, cab_cy), (cab_w, cab_h), angle, cab_color)
        
        body_w, body_h = int(16 * scale), int(9 * scale)
        body_cx = cx + int(6 * scale * np.cos(rad))
        body_cy = cy + int(6 * scale * np.sin(rad))
        self.draw_rect(draw, (body_cx, body_cy), (body_w, body_h), angle, body_color)
        
        perp = rad + np.pi/2
        positions = [
            (cab_cx, cab_cy, 3),
            (cx, cy, 3),
            (int(body_cx + 4*scale*np.cos(rad)), int(body_cy + 4*scale*np.sin(rad)), 3)
        ]
        
        for wx, wy, dist in positions:
            for sign in [-1, 1]:
                wheel_x = int(wx + sign * dist * scale * np.cos(perp))
                wheel_y = int(wy + sign * dist * scale * np.sin(perp))
                r = int(2.5 * scale)
                draw.ellipse([wheel_x-r, wheel_y-r, wheel_x+r, wheel_y+r], fill=wheel_color)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def generate(self, class_id, add_noise=True):
        bg = self.generate_background()
        
        margin = 15
        cx = random.randint(margin, self.img_size - margin)
        cy = random.randint(margin, self.img_size - margin)
        angle = random.uniform(0, 360)
        scale = random.uniform(0.8, 1.2)
        brightness = random.uniform(0.8, 1.2)
        
        if class_id == 0:
            img = self.draw_tank(bg, (cx, cy), angle, scale)
        elif class_id == 1:
            img = self.draw_btr(bg, (cx, cy), angle, scale)
        else:
            img = self.draw_truck(bg, (cx, cy), angle, scale)
            
        img = np.clip(img * brightness, 0, 1)
        
        if add_noise:
            img = self.add_noise(img)
            
        return img.astype(np.float32), class_id


class MilitaryDataset(Dataset):
    def __init__(self, num_samples=100, img_size=64, augment=True):
        self.generator = MilitaryVehicleGenerator(img_size)
        self.num_samples = num_samples
        self.augment = augment
        self.labels = [i % 3 for i in range(num_samples)]
        random.shuffle(self.labels)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img, _ = self.generator.generate(label)
        
        if self.augment:
            return self.transform(img), label
        else:
            return transforms.ToTensor()(img), label


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.conv(x)
        return x * attn, attn


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.attn2 = SpatialAttention(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.attn3 = SpatialAttention(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x, attn2 = self.attn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x, attn3 = self.attn3(x)
        x = self.pool3(x)
        
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        
        if return_attention:
            return x, {'attn2': attn2, 'attn3': attn3}
        else:
            return x, {}


class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def random_mask(self, images, max_ratio=0.2):
        b, c, h, w = images.shape
        masked = images.clone()
        
        for i in range(b):
            mh = int(h * random.uniform(0.1, max_ratio))
            mw = int(w * random.uniform(0.1, max_ratio))
            y = random.randint(0, h - mh)
            x = random.randint(0, w - mw)
            masked[i, :, y:y+mh, x:x+mw] = torch.rand(c, mh, mw).to(self.device)
            
        return masked
    
    def train_epoch(self, loader, optimizer, use_robust=True):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if use_robust and random.random() > 0.5:
                images = self.random_mask(images)
            
            optimizer.zero_grad()
            outputs, _ = self.model(images, return_attention=False)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(loader), 100. * correct / total
    
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.model(images, return_attention=False)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return 100. * correct / total


class Explainer:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or ['T-72', 'BTR-82A', 'Ural-4320']
        
    def visualize(self, image_tensor, save_path=None):
        self.model.eval()
        
        with torch.no_grad():
            output, attns = self.model(image_tensor.unsqueeze(0), return_attention=True)
            pred = output.argmax(dim=1).item()
            
        attn_map = attns['attn3'][0, 0].cpu().numpy()
        
        from scipy.ndimage import zoom
        zoom_factor = 64 / attn_map.shape[0]
        attn_resized = zoom(attn_map, zoom_factor, order=1)
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        im = axes[1].imshow(attn_resized, cmap='hot')
        axes[1].set_title('Attention')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        axes[2].imshow(np.clip(img_np, 0, 1))
        axes[2].imshow(attn_resized, cmap='hot', alpha=0.6)
        axes[2].set_title(f'Pred: {self.class_names[pred]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, attn_resized, pred


def main():
    print("=" * 60)
    print("СИСТЕМА ДЕТЕКЦИИ ТЕХНИКИ")
    print("=" * 60)
    
    print("\n[1] Генерация данных...")
    train_data = MilitaryDataset(num_samples=500, augment=True)
    test_data = MilitaryDataset(num_samples=100, augment=False)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    print("[2] Инициализация модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightCNN(num_classes=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Параметров: {total_params:,}")
    print(f"    Размер: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n[3] Обучение...")
    trainer = Trainer(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_acc = 0
    for epoch in range(20):
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, use_robust=True)
        test_acc = trainer.evaluate(test_loader)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: Loss={train_loss:.3f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    print(f"\n    Лучшая точность: {best_acc:.1f}%")
    
    print("\n[4] Тестирование...")
    model.load_state_dict(torch.load('best_model.pth'))
    noisy_acc = trainer.evaluate(DataLoader(MilitaryDataset(num_samples=100, augment=False), batch_size=16))
    print(f"    Точность на шуме: {noisy_acc:.1f}%")
    
    print("\n[5] Визуализация...")
    explainer = Explainer(model)
    
    for idx in [0, 5, 10]:
        img, label = test_data[idx]
        fig, attn, pred = explainer.visualize(img, save_path=f'attention_{idx}.png')
        print(f"    Образец {idx}: {explainer.class_names[pred]}")
        
    print("\n[6] Замер скорости...")
    model.eval()
    dummy = torch.randn(1, 3, 64, 64).to(device)
    
    for _ in range(10):
        _ = model(dummy)
        
    import time
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
            
    avg_time = np.mean(times)
    print(f"    Задержка: {avg_time:.2f} ms")
    print(f"    FPS: {1000/avg_time:.1f}")
    
    print("\n[7] Экспорт...")
    
    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save('model_scripted.pt')
        print("    Сохранено: model_scripted.pt")
    except Exception as e:
        print(f"    TorchScript: {str(e)[:50]}")
    
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, dummy_input, 'model.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                   'output': {0: 'batch_size'}})
    print("    Сохранено: model.onnx")
    
    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    
    return model, explainer


if __name__ == "__main__":
    model, explainer = main()
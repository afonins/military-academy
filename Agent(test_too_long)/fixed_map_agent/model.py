import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingPatrolNet(nn.Module):
    """
    Dueling DQN архитектура для патрулирования.
    Разделяет оценку состояния (value) и преимуществ действий (advantage).
    """
    def __init__(self, map_size=10):
        super(DuelingPatrolNet, self).__init__()
        self.map_size = map_size
        
        # Convolutional layers для извлечения признаков карты
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Размер после сверток
        conv_out_size = 64 * map_size * map_size
        
        # Value stream - оценка ценности состояния
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream - оценка преимуществ действий
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PatrolNet(nn.Module):
    """
    Простая CNN архитектура для патрулирования (fallback вариант).
    """
    def __init__(self, map_size=10):
        super(PatrolNet, self).__init__()
        self.map_size = map_size
        
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * map_size * map_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

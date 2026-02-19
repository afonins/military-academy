import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
from typing import List, Tuple
from model import DuelingPatrolNet, PatrolNet


# Опыт для обычного replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Приоритетные выборки для более эффективного обучения.
    """
    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Степень приоритета (0 = uniform, 1 = full priority)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Добавление опыта с максимальным приоритетом."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Выборка с приоритетами."""
        if len(self.buffer) == 0:
            return None
        
        # Вычисляем вероятности выборки
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Выбираем индексы
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Вычисляем веса для коррекции смещения
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Извлекаем опыт
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Обновление приоритетов после обучения."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Малое число для стабильности
    
    def __len__(self):
        return len(self.buffer)


class PatrolAgent:
    """
    DQN агент для патрулирования с улучшенными техниками обучения.
    """
    
    def __init__(
        self,
        map_size: int = 10,
        use_dueling: bool = True,
        load_model: bool = True,
        model_dir: str = "models"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.map_size = map_size
        self.model_dir = model_dir
        
        # Создаем директорию для моделей
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "patrol_model.pth")
        self.config_path = os.path.join(model_dir, "agent_config.pth")
        
        # Инициализация сетей
        NetClass = DuelingPatrolNet if use_dueling else PatrolNet
        self.policy_net = NetClass(map_size).to(self.device)
        self.target_net = NetClass(map_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Оптимизатор
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        
        # Параметры обучения
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005  # Для soft update target сети
        
        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Счетчики
        self.steps_done = 0
        self.episodes_done = 0
        
        # Статистика
        self.training_history = {
            'losses': [],
            'rewards': [],
            'coverages': [],
            'epsilons': []
        }
        
        # Загрузка существующей модели
        if load_model:
            self.load()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Выбор действия с epsilon-greedy стратегией."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Сохранение перехода в буфер."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> float:
        """Обучение на батче из буфера."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Выборка с приоритетами
        sample = self.memory.sample(self.batch_size, beta=0.4)
        if sample is None:
            return 0.0
        
        states, actions, rewards, next_states, dones, indices, weights = sample
        
        # Конвертация в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Текущие Q-значения
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: используем policy_net для выбора действия, target_net для оценки
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Вычисление ошибки с весами
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Обновление приоритетов
        self.memory.update_priorities(indices, td_errors)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update target сети
        self._soft_update_target()
        
        self.steps_done += 1
        
        return loss.item()
    
    def _soft_update_target(self):
        """Мягкое обновление target сети."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        """Уменьшение epsilon для exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def save(self):
        """Сохранение модели и конфигурации."""
        # Сохраняем веса модели
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, self.model_path)
        
        # Сохраняем конфигурацию агента
        torch.save({
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'training_history': self.training_history
        }, self.config_path)
        
        print(f"💾 Модель сохранена: {self.model_path}")
        print(f"   Epsilon: {self.epsilon:.4f}, Steps: {self.steps_done}, Episodes: {self.episodes_done}")
    
    def load(self):
        """Загрузка модели и конфигурации."""
        if os.path.exists(self.model_path) and os.path.exists(self.config_path):
            try:
                # Загружаем веса
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
                # Загружаем конфигурацию
                config = torch.load(self.config_path, map_location=self.device)
                self.epsilon = config['epsilon']
                self.steps_done = config['steps_done']
                self.episodes_done = config['episodes_done']
                self.training_history = config['training_history']
                
                print(f"🧠 Модель загружена: {self.model_path}")
                print(f"   Epsilon: {self.epsilon:.4f}, Steps: {self.steps_done}, Episodes: {self.episodes_done}")
                return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("   Начинаем обучение с нуля.")
                return False
        else:
            print("🆕 Модель не найдена, начинаем обучение с нуля.")
            return False
    
    def get_stats(self) -> dict:
        """Получение статистики агента."""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'memory_size': len(self.memory),
            'device': str(self.device)
        }

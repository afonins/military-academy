import numpy as np
import random
from typing import List, Tuple, Set, Dict


class FixedPatrolEnv:
    """
    Среда для патрулирования с фиксированной картой.
    Карта 10x10 со стенами и зонами риска.
    """
    
    # Действия: 0=вверх, 1=вниз, 2=влево, 3=вправо
    ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, size: int = 10, max_steps: int = 200):
        self.size = size
        self.max_steps = max_steps
        
        # Фиксированные стены (препятствия)
        self.walls: Set[Tuple[int, int]] = {
            (2, 2), (2, 3), (3, 2), (3, 3),  # Левый верхний кластер
            (6, 6), (6, 7), (7, 6), (7, 7),  # Правый нижний кластер
            (2, 7), (3, 7),                  # Правый верхний
            (7, 2), (7, 3),                  # Левый нижний
            (4, 5), (5, 4)                   # Центральные препятствия
        }
        
        # Зоны риска - где появляются враги
        self.risk_zones: List[Tuple[int, int]] = [
            (1, 1), (1, 8), (8, 1), (8, 8),  # Углы
            (5, 5),                          # Центр
            (3, 5), (5, 3), (6, 5), (5, 6)   # Дополнительные зоны
        ]
        
        # Враги: [y, x, lifetime]
        self.targets: List[List[int]] = []
        
        # Текущая позиция агента
        self.pos: List[int] = [5, 5]
        
        # Счетчики
        self.steps = 0
        self.caught = 0
        self.missed = 0
        
        # Матрица посещений для поощрения исследования
        self.visit_count: np.ndarray = np.zeros((size, size), dtype=np.float32)
        
        # Матрица времени с последнего посещения (для патрулирования)
        self.time_since_visit: np.ndarray = np.zeros((size, size), dtype=np.float32)
        
        # История пути для визуализации
        self.path_history: List[Tuple[int, int]] = []
        
    def reset(self) -> np.ndarray:
        """Сброс среды к начальному состоянию."""
        # Начальная позиция - центр или случайная свободная клетка
        self.pos = [5, 5]
        if tuple(self.pos) in self.walls:
            self.pos = self._find_random_free_cell()
        
        self.targets = []
        self.steps = 0
        self.caught = 0
        self.missed = 0
        self.visit_count = np.zeros((self.size, self.size), dtype=np.float32)
        self.time_since_visit = np.zeros((self.size, self.size), dtype=np.float32)
        self.path_history = [tuple(self.pos)]
        
        # Начальное заспавнивание врагов
        self._spawn_targets(force=True)
        
        return self.get_state()
    
    def _find_random_free_cell(self) -> List[int]:
        """Найти случайную свободную клетку."""
        while True:
            y, x = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (y, x) not in self.walls:
                return [y, x]
    
    def get_state(self) -> np.ndarray:
        """
        Формирование состояния для нейросети.
        4 канала:
        - Канал 0: позиция агента (one-hot)
        - Канал 1: стены (1) и враги (-1)
        - Канал 2: нормализованное количество посещений
        - Канал 3: время с последнего посещения (нормализованное)
        """
        state = np.zeros((4, self.size, self.size), dtype=np.float32)
        
        # Канал 0: позиция агента
        state[0, self.pos[0], self.pos[1]] = 1.0
        
        # Канал 1: стены и враги
        for (y, x) in self.walls:
            state[1, y, x] = 1.0
        for t in self.targets:
            state[1, t[0], t[1]] = -1.0  # Враги отрицательные
        
        # Канал 2: нормализованные посещения
        max_visits = np.max(self.visit_count) + 1
        state[2] = self.visit_count / max_visits
        
        # Канал 3: время с последнего посещения (чем больше, тем важнее посетить)
        max_time = np.max(self.time_since_visit) + 1
        state[3] = self.time_since_visit / max_time
        
        return state
    
    def _spawn_targets(self, force: bool = False):
        """Спавн врагов в зонах риска."""
        max_targets = 2
        
        if len(self.targets) >= max_targets:
            return
        
        # Шанс спавна (увеличен если force=True)
        spawn_chance = 0.3 if force else 0.15
        if not force and random.random() > spawn_chance:
            return
        
        # Выбираем случайную зону риска
        zone = random.choice(self.risk_zones)
        
        # Добавляем случайное смещение ±1
        y = max(0, min(self.size - 1, zone[0] + random.randint(-1, 1)))
        x = max(0, min(self.size - 1, zone[1] + random.randint(-1, 1)))
        
        # Проверяем, что клетка свободна
        if (y, x) not in self.walls and [y, x] != self.pos:
            # Время жизни врага (шагов)
            lifetime = random.randint(15, 30)
            self.targets.append([y, x, lifetime])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Выполнение действия.
        
        Returns:
            state: новое состояние
            reward: награда
            done: флаг окончания эпизода
            info: дополнительная информация
        """
        reward = 0.0
        info = {'caught': False, 'hit_wall': False, 'missed_target': False}
        
        old_pos = self.pos.copy()
        dy, dx = self.ACTION_DELTAS[action]
        new_y, new_x = self.pos[0] + dy, self.pos[1] + dx
        
        # Проверка столкновения со стеной или границей
        hit_wall = False
        if (new_y, new_x) in self.walls or not (0 <= new_y < self.size and 0 <= new_x < self.size):
            hit_wall = True
            info['hit_wall'] = True
            new_y, new_x = old_pos  # Остаемся на месте
            reward -= 1.0  # Штраф за удар о стену
        
        self.pos = [new_y, new_x]
        self.steps += 1
        
        # Обновление матриц посещений
        self.visit_count[new_y, new_x] += 1
        self.time_since_visit += 1
        self.time_since_visit[new_y, new_x] = 0
        
        # Награды за перемещение
        
        # 1. Награда за посещение новой клетки
        visits = self.visit_count[new_y, new_x]
        if visits == 1:
            reward += 3.0  # Первая посещение - большая награда
        elif visits <= 2:
            reward += 1.0  # Вторая посещение - средняя награда
        else:
            # Штраф за частое посещение (чтобы избежать зацикливания)
            reward -= 0.3 * (visits - 2)
        
        # 2. Награда за исследование соседних непосещенных клеток
        unexplored_neighbors = 0
        for ndy, ndx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = new_y + ndy, new_x + ndx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                if self.visit_count[ny, nx] == 0 and (ny, nx) not in self.walls:
                    unexplored_neighbors += 1
        reward += 0.5 * unexplored_neighbors
        
        # 3. Награда за движение (штраф за стояние на месте)
        if old_pos == self.pos and not hit_wall:
            reward -= 0.5
        
        # 4. Обработка врагов
        caught_target = False
        new_targets = []
        
        for t in self.targets:
            if t[0] == new_y and t[1] == new_x:
                # Поймали врага!
                reward += 10.0
                self.caught += 1
                caught_target = True
                info['caught'] = True
            else:
                # Уменьшаем время жизни врага
                t[2] -= 1
                if t[2] > 0:
                    new_targets.append(t)
                else:
                    # Враг исчез - штраф
                    reward -= 5.0
                    self.missed += 1
                    info['missed_target'] = True
        
        self.targets = new_targets
        
        # 5. Дополнительная награда за покрытие зон риска
        if (new_y, new_x) in self.risk_zones:
            reward += 1.0  # Бонус за патрулирование зоны риска
        
        # Обновление истории
        self.path_history.append(tuple(self.pos))
        
        # Спавн новых врагов
        self._spawn_targets()
        
        # Проверка окончания эпизода
        done = False
        if self.missed >= 3:  # Слишком много пропущенных врагов
            reward -= 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            # Бонус за покрытие карты при завершении
            coverage = np.sum(self.visit_count > 0) / (self.size * self.size - len(self.walls))
            reward += coverage * 20.0
        
        return self.get_state(), reward, done, info
    
    def get_stats(self) -> Dict:
        """Получение статистики текущего эпизода."""
        coverage = np.sum(self.visit_count > 0) / (self.size * self.size - len(self.walls))
        return {
            'steps': self.steps,
            'caught': self.caught,
            'missed': self.missed,
            'coverage': coverage,
            'total_visits': np.sum(self.visit_count),
            'unique_cells': np.sum(self.visit_count > 0)
        }
    
    def render_text(self):
        """Текстовая визуализация карты."""
        print("\n" + "=" * (self.size * 2 + 1))
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if [y, x] == self.pos:
                    row += "A "  # Агент
                elif (y, x) in self.walls:
                    row += "# "  # Стена
                elif any(t[0] == y and t[1] == x for t in self.targets):
                    row += "E "  # Враг
                elif (y, x) in self.risk_zones:
                    row += "! "  # Зона риска
                elif self.visit_count[y, x] > 0:
                    row += ". "  # Посещенная клетка
                else:
                    row += "  "  # Пустая клетка
            print(f"|{row}|")
        print("=" * (self.size * 2 + 1))
        stats = self.get_stats()
        print(f"Steps: {stats['steps']} | Caught: {stats['caught']} | "
              f"Missed: {stats['missed']} | Coverage: {stats['coverage']*100:.1f}%")

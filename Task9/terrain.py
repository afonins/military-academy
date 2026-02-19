import numpy as np
import random
from typing import Tuple, List
from config import TerrainType, CombatParams


class TerrainMap:
    """Карта местности 20x20 км"""
    
    def __init__(self, size_km: int = 20, grid_size: int = 100):
        self.size_km = size_km
        self.grid_size = grid_size
        self.cell_size = size_km / grid_size
        
        # Сетка местности (числовые коды)
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.elevation = np.zeros((grid_size, grid_size))
        
        self._generate_terrain()
        
    def _generate_terrain(self):
        """Генерация реалистичного ландшафта"""
        
        # 1. Город (юго-запад)
        city_center = (int(self.grid_size * 0.25), int(self.grid_size * 0.75))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist_to_city = np.sqrt((i - city_center[0])**2 + (j - city_center[1])**2)
                if dist_to_city < 15:
                    self.grid[i, j] = 1  # CITY = 1
        
        # 2. Лес (север и восток)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] != 0:
                    continue
                if i < self.grid_size * 0.6 and j > self.grid_size * 0.4:
                    if random.random() < 0.4:
                        self.grid[i, j] = 2  # FOREST = 2
        
        # 3. Река (с запада на восток, изгиб)
        for x in range(self.grid_size):
            river_y = int(self.grid_size * 0.5 + 10 * np.sin(x * 0.1))
            for dy in range(-2, 3):
                if 0 <= river_y + dy < self.grid_size:
                    self.grid[x, river_y + dy] = 4  # RIVER = 4
        
        # 4. Высоты (центр)
        hill_center = (int(self.grid_size * 0.6), int(self.grid_size * 0.4))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - hill_center[0])**2 + (j - hill_center[1])**2)
                self.elevation[i, j] = max(0, 200 - dist * 5)
                if 150 < self.elevation[i, j] < 180 and self.grid[i, j] == 0:
                    self.grid[i, j] = 5  # HILL = 5
        
        # 5. Остальное - равнина
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    self.grid[i, j] = 3  # PLAIN = 3
    
    def get_terrain(self, x_km: float, y_km: float) -> TerrainType:
        """Получить тип местности по координатам"""
        i = int(x_km / self.cell_size)
        j = int(y_km / self.cell_size)
        i = max(0, min(i, self.grid_size - 1))
        j = max(0, min(j, self.grid_size - 1))
        
        code = self.grid[i, j]
        mapping = {
            1: TerrainType.CITY,
            2: TerrainType.FOREST,
            3: TerrainType.PLAIN,
            4: TerrainType.RIVER,
            5: TerrainType.HILL
        }
        return mapping.get(code, TerrainType.PLAIN)
    
    def get_elevation(self, x_km: float, y_km: float) -> float:
        """Высота в точке"""
        i = int(x_km / self.cell_size)
        j = int(y_km / self.cell_size)
        i = max(0, min(i, self.grid_size - 1))
        j = max(0, min(j, self.grid_size - 1))
        return self.elevation[i, j]
    
    def get_visibility(self, x_km: float, y_km: float) -> float:
        """Дальность видимости в км"""
        terrain = self.get_terrain(x_km, y_km)
        visibility_map = {
            TerrainType.CITY: CombatParams.VISIBILITY_CITY,
            TerrainType.FOREST: CombatParams.VISIBILITY_FOREST,
            TerrainType.PLAIN: CombatParams.VISIBILITY_PLAIN,
            TerrainType.HILL: CombatParams.VISIBILITY_HILL,
            TerrainType.RIVER: CombatParams.VISIBILITY_PLAIN
        }
        base = visibility_map.get(terrain, 3)
        
        # С высоты видно дальше
        elev = self.get_elevation(x_km, y_km)
        if elev > 150:
            base += (elev - 150) / 50
        
        return base
    
    def can_move(self, x_km: float, y_km: float, unit_type) -> bool:
        """Проверка проходимости"""
        terrain = self.get_terrain(x_km, y_km)
        if terrain == TerrainType.RIVER:
            return False
        return True
    
    def get_movement_speed(self, x_km: float, y_km: float, is_tank: bool) -> float:
        """Скорость движения с учетом местности"""
        terrain = self.get_terrain(x_km, y_km)
        
        if is_tank:
            speeds = {
                TerrainType.PLAIN: CombatParams.SPEED_TANK_PLAIN,
                TerrainType.CITY: CombatParams.SPEED_TANK_CITY,
                TerrainType.FOREST: CombatParams.SPEED_TANK_PLAIN * 0.6,
                TerrainType.HILL: CombatParams.SPEED_TANK_PLAIN * 0.5,
                TerrainType.RIVER: 0
            }
        else:
            speeds = {
                TerrainType.PLAIN: CombatParams.SPEED_INFANTRY * 2,
                TerrainType.CITY: CombatParams.SPEED_INFANTRY,
                TerrainType.FOREST: CombatParams.SPEED_INFANTRY * 1.5,
                TerrainType.HILL: CombatParams.SPEED_INFANTRY,
                TerrainType.RIVER: 0
            }
        
        return speeds.get(terrain, 10)
    
    def print_map(self):
        """Текстовая визуализация карты"""
        symbols = {
            1: "⌂",  # CITY
            2: "𖣂",  # FOREST
            3: "_",  # PLAIN
            4: "~",  # RIVER
            5: "◠"   # HILL
        }
        
        print("\n" + "="*60)
        print("КАРТА МЕСТНОСТИ (20x20 км)")
        print("="*60)
        print("⌂ =город, 𖣂 =лес, _ =равнина, ~ =река, ◠ =высота")
        print("-"*60)
        
        # Уменьшенная сетка для вывода
        step = self.grid_size // 20
        for j in range(self.grid_size - 1, -1, -step):
            line = ""
            for i in range(0, self.grid_size, step):
                code = self.grid[i, j]
                line += symbols.get(code, "?")
            print(line)
        
        print("-"*60)
        print("Север ↑")
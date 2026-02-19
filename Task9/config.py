"""
Конфигурация симуляции бригадной операции
"""

from enum import Enum
from dataclasses import dataclass


class TerrainType(Enum):
    CITY = "город"
    FOREST = "лес"
    PLAIN = "равнина"
    RIVER = "река"
    HILL = "высота"


class UnitType(Enum):
    TANK = "танковый"
    MOTORIZED = "мотострелковый"
    MECHANIZED = "механизированный"
    ARTILLERY = "артиллерийский"
    RECON = "разведывательный"
    AIR = "авиация"


@dataclass
class CombatParams:
    """Боевые параметры"""
    # Эффективность по местности (множитель)
    TANK_CITY = 0.3      # Танки в городе слабы
    TANK_PLAIN = 1.0     # На равнине сильны
    TANK_FOREST = 0.5
    
    INFANTRY_CITY = 1.0   # Пехота в городе сильна
    INFANTRY_PLAIN = 0.6
    INFANTRY_FOREST = 0.9
    
    # Дальности обнаружения
    VISIBILITY_CITY = 2   # км
    VISIBILITY_FOREST = 1
    VISIBILITY_PLAIN = 5
    VISIBILITY_HILL = 8
    
    # Скорости движения (км/ч)
    SPEED_TANK_PLAIN = 40
    SPEED_TANK_CITY = 15
    SPEED_INFANTRY = 5


# Организационная структура бригады
BRIGADE_STRUCTURE = {
    "name": "150-я отдельная мотострелковая бригада",
    "battalions": [
        {
            "id": "1-й батальон",
            "type": UnitType.MECHANIZED,
            "companies": 3,
            "strength": 450,  # человек
            "tanks": 10,
            "ifvs": 30,
            "apcs": 10
        },
        {
            "id": "2-й батальон", 
            "type": UnitType.MOTORIZED,
            "companies": 3,
            "strength": 400,
            "tanks": 0,
            "ifvs": 0,
            "apcs": 40
        },
        {
            "id": "3-й батальон",
            "type": UnitType.TANK,
            "companies": 3,
            "strength": 300,
            "tanks": 31,
            "ifvs": 0,
            "apcs": 5
        }
    ],
    "support": {
        "artillery_battalion": {
            "guns": 18,      # 152-мм гаубицы
            "rockets": 6,    # РСЗО
            "ammo": 500
        },
        "recon_company": {
            "drones": 8,
            "vehicles": 12,
            "personnel": 80
        },
        "air_support": {
            "helicopters": 4,
            "sorties_per_day": 12
        }
    }
}

# Вражеские силы (оппонент)
ENEMY_FORCES = {
    "name": "Условный противник",
    "battalions": 2,
    "strongpoints": 5,
    "minefields": 3,
    "artillery_batteries": 2
}

# Параметры симуляции
SIMULATION_CONFIG = {
    "duration_hours": 24,
    "time_step_minutes": 30,  # Шаг симуляции
    "map_size_km": 20,        # 20x20 км
    "grid_size": 100          # Разрешение сетки
}
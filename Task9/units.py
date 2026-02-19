"""
Воинские подразделения
"""

import math
import random
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from config import UnitType, TerrainType, CombatParams


@dataclass
class CombatUnit:
    """Боевая единица (рота/батальон)"""
    id: str
    unit_type: UnitType
    strength: int          # Личный состав
    x: float = 0.0         # км
    y: float = 0.0         # км
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    
    # Боевая мощь
    combat_effectiveness: float = 1.0  # 0-1
    ammo: int = 100
    morale: float = 100.0   # 0-100
    fatigue: float = 0.0    # Усталость 0-100
    
    # Техника
    tanks: int = 0
    ifvs: int = 0          # БМП
    apcs: int = 0          # БТР
    
    # Состояние
    is_alive: bool = True
    in_combat: bool = False
    entrenchment: float = 0.0  # 0-1 (окопание)
    
    # Лог
    history: List[dict] = field(default_factory=list)
    
    def get_speed(self, terrain: TerrainType) -> float:
        """Скорость с учетом типа подразделения и местности"""
        has_tanks = self.tanks > 0
        
        if terrain == TerrainType.PLAIN:
            base = CombatParams.SPEED_TANK_PLAIN if has_tanks else CombatParams.SPEED_INFANTRY * 2
        elif terrain == TerrainType.CITY:
            base = CombatParams.SPEED_TANK_CITY if has_tanks else CombatParams.SPEED_INFANTRY
        elif terrain == TerrainType.FOREST:
            base = CombatParams.SPEED_TANK_PLAIN * 0.6 if has_tanks else CombatParams.SPEED_INFANTRY * 1.5
        else:
            base = 10
        
        # Модификаторы
        if self.fatigue > 50:
            base *= 0.7
        if self.morale < 30:
            base *= 0.5
        
        return base
    
    def get_combat_power(self, terrain: TerrainType) -> float:
        """Боевая мощь с учетом местности"""
        base_power = (
            self.tanks * 5 +
            self.ifvs * 2 +
            self.apcs * 1 +
            self.strength * 0.01
        ) * self.combat_effectiveness * (self.morale / 100)
        
        # Модификатор местности
        if self.tanks > 0:
            if terrain == TerrainType.CITY:
                modifier = CombatParams.TANK_CITY
            elif terrain == TerrainType.FOREST:
                modifier = CombatParams.TANK_FOREST
            else:
                modifier = CombatParams.TANK_PLAIN
        else:
            if terrain == TerrainType.CITY:
                modifier = CombatParams.INFANTRY_CITY
            elif terrain == TerrainType.FOREST:
                modifier = CombatParams.INFANTRY_FOREST
            else:
                modifier = CombatParams.INFANTRY_PLAIN
        
        # Окопание дает бонус обороне
        if self.entrenchment > 0:
            modifier *= (1 + self.entrenchment * 0.5)
        
        return base_power * modifier
    
    def move_to(self, target_x: float, target_y: float):
        """Установить цель движения"""
        self.target_x = target_x
        self.target_y = target_y
    
    def update_position(self, terrain_map, time_hours: float):
        """Обновить позицию"""
        if not self.is_alive:
            return
        
        if self.target_x is None or self.target_y is None:
            return
        
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.5:  # Достигли цели
            self.target_x = None
            self.target_y = None
            self.entrenchment = min(1.0, self.entrenchment + 0.1 * time_hours)
            return
        
        # Движение
        terrain = terrain_map.get_terrain(self.x, self.y)
        speed = self.get_speed(terrain)  # км/ч
        move_dist = speed * time_hours
        
        if move_dist >= dist:
            self.x = self.target_x
            self.y = self.target_y
        else:
            self.x += (dx / dist) * move_dist
            self.y += (dy / dist) * move_dist
        
        # Усталость растет при движении
        self.fatigue = min(100, self.fatigue + 5 * time_hours)
        self.entrenchment = max(0, self.entrenchment - 0.05 * time_hours)
        
        self.history.append({
            "time": time_hours,
            "action": "move",
            "x": self.x,
            "y": self.y,
            "fatigue": self.fatigue
        })
    
    def take_damage(self, damage: float):
        """Получить урон"""
        self.strength -= int(damage * 10)
        self.morale -= damage * 5
        self.combat_effectiveness -= damage * 0.1
        
        if self.strength <= 0 or self.combat_effectiveness <= 0:
            self.is_alive = False
            self.strength = 0
        
        self.morale = max(0, self.morale)
        self.combat_effectiveness = max(0.1, self.combat_effectiveness)
    
    def rest(self, time_hours: float):
        """Отдых и восстановление"""
        self.fatigue = max(0, self.fatigue - 20 * time_hours)
        self.morale = min(100, self.morale + 5 * time_hours)
        
        if self.ammo < 100:
            self.ammo = min(100, self.ammo + 30 * time_hours)


class Battalion:
    """Батальон (3 роты)"""
    
    def __init__(self, config: dict):
        self.id = config["id"]
        self.unit_type = config["type"]
        self.companies: List[CombatUnit] = []
        
        # Создаем 3 роты
        for i in range(3):
            company = CombatUnit(
                id=f"{self.id}-рота-{i+1}",
                unit_type=self.unit_type,
                strength=config["strength"] // 3,
                tanks=config["tanks"] // 3,
                ifvs=config["ifvs"] // 3,
                apcs=config["apcs"] // 3
            )
            self.companies.append(company)
        
        self.total_strength = config["strength"]
    
    def deploy(self, x: float, y: float, formation: str = "line"):
        """Развернуть батальон в боевом порядке"""
        if formation == "line":
            offsets = [(-2, 0), (0, 0), (2, 0)]
        elif formation == "wedge":
            offsets = [(-1, -1), (0, 1), (1, -1)]
        else:
            offsets = [(0, 0), (1, 0), (2, 0)]
        
        for company, (dx, dy) in zip(self.companies, offsets):
            company.x = x + dx
            company.y = y + dy
    
    def set_objective(self, x: float, y: float):
        """Установить цель для всего батальона"""
        # Роты двигаются с небольшим смещением
        for i, company in enumerate(self.companies):
            offset_x = (i - 1) * 1.5
            offset_y = (i - 1) * 1.0
            company.move_to(x + offset_x, y + offset_y)
    
    def update(self, terrain_map, time_hours: float):
        """Обновить состояние батальона"""
        for company in self.companies:
            company.update_position(terrain_map, time_hours)
    
    def get_status(self) -> dict:
        """Статус батальона"""
        alive_companies = [c for c in self.companies if c.is_alive]
        total_strength = sum(c.strength for c in alive_companies)
        
        return {
            "id": self.id,
            "type": self.unit_type.value,
            "companies_alive": len(alive_companies),
            "total_strength": total_strength,
            "avg_morale": sum(c.morale for c in alive_companies) / max(1, len(alive_companies)),
            "position": (sum(c.x for c in alive_companies) / max(1, len(alive_companies)),
                        sum(c.y for c in alive_companies) / max(1, len(alive_companies)))
        }
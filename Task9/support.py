"""
Средства поддержки: артиллерия, авиация, разведка
"""

import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ArtilleryStrike:
    """Артудар"""
    target_x: float
    target_y: float
    intensity: float  # 0-1 (залп/полный)
    ammo_used: int
    damage_dealt: float = 0.0
    accuracy: float = 0.0


class ArtillerySupport:
    """Артиллерийский дивизион"""
    
    def __init__(self, config: dict):
        self.guns = config["guns"]
        self.rockets = config["rockets"]
        self.ammo = config["ammo"]
        self.available = True
        
        # Позиция (в тылу)
        self.x = 5.0
        self.y = 5.0
        
        self.strikes_made = 0
        self.total_damage = 0.0
    
    def calculate_strike(self, target_x: float, target_y: float, 
                        urgency: str = "normal") -> Optional[ArtilleryStrike]:
        """Расчет артудора"""
        if self.ammo <= 0:
            return None
        
        # Расход снарядов
        if urgency == "high":
            ammo = min(50, self.ammo)
            intensity = 1.0
        elif urgency == "suppression":
            ammo = min(20, self.ammo)
            intensity = 0.5
        else:
            ammo = min(30, self.ammo)
            intensity = 0.7
        
        # Точность зависит от разведки (упрощение)
        distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
        accuracy = max(0.3, 1.0 - distance / 30)  # Дальше = хуже точность
        
        strike = ArtilleryStrike(
            target_x=target_x,
            target_y=target_y,
            intensity=intensity,
            ammo_used=ammo,
            accuracy=accuracy
        )
        
        return strike
    
    def execute_strike(self, strike: ArtilleryStrike, enemy_units: List) -> float:
        """Выполнить удар"""
        self.ammo -= strike.ammo_used
        self.strikes_made += 1
        
        damage = 0.0
        
        for enemy in enemy_units:
            dist = math.sqrt((enemy.x - strike.target_x)**2 + 
                           (enemy.y - strike.target_y)**2)
            
            if dist < 2.0:  # Радиус поражения 2 км
                hit_prob = strike.accuracy * (1 - dist / 2)
                if random.random() < hit_prob:
                    dmg = strike.intensity * random.uniform(10, 30)
                    enemy.take_damage(dmg)
                    damage += dmg
        
        strike.damage_dealt = damage
        self.total_damage += damage
        
        return damage
    
    def get_status(self) -> dict:
        return {
            "ammo_remaining": self.ammo,
            "strikes_made": self.strikes_made,
            "total_damage": round(self.total_damage, 1),
            "guns_operational": self.guns
        }


class ReconSupport:
    """Разведывательная рота с БПЛА"""
    
    def __init__(self, config: dict):
        self.drones = config["drones"]
        self.vehicles = config["vehicles"]
        self.personnel = config["personnel"]
        
        self.detected_targets: List[Tuple[float, float, str]] = []
        self.missions_flown = 0
    
    def reconnaissance_mission(self, sector_x: float, sector_y: float, 
                               radius: float = 5.0) -> List[Tuple[float, float]]:
        """Разведывательный вылет"""
        if self.drones <= 0:
            return []
        
        self.missions_flown += 1
        self.drones = max(0, self.drones - random.randint(0, 1))  # Потери
        
        # Обнаружение целей (симуляция)
        detected = []
        # В реальности - проверка against enemy positions
        for _ in range(random.randint(1, 3)):
            x = sector_x + random.uniform(-radius, radius)
            y = sector_y + random.uniform(-radius, radius)
            detected.append((x, y))
            self.detected_targets.append((x, y, "enemy_unit"))
        
        return detected
    
    def get_status(self) -> dict:
        return {
            "drones_available": self.drones,
            "missions_flown": self.missions_flown,
            "targets_detected": len(self.detected_targets)
        }


class AirSupport:
    """Авиационная поддержка (вертолеты)"""
    
    def __init__(self, config: dict):
        self.helicopters = config["helicopters"]
        self.sorties_available = config["sorties_per_day"]
        self.sorties_used = 0
    
    def request_strike(self, priority: str = "normal") -> bool:
        """Запрос авиаудара"""
        if self.sorties_used >= self.sorties_available:
            return False
        
        # Проверка погоды (упрощение)
        if random.random() < 0.2:  # 20% отказ по погоде
            return False
        
        self.sorties_used += 1
        return True
    
    def execute_strike(self, target_x: float, target_y: float, 
                      enemy_units: List) -> float:
        """Удар с воздуха"""
        damage = 0.0
        
        for enemy in enemy_units:
            dist = math.sqrt((enemy.x - target_x)**2 + (enemy.y - target_y)**2)
            if dist < 1.5:  # Точность выше артиллерии
                if random.random() < 0.7:  # 70% попадание
                    dmg = random.uniform(20, 50)
                    enemy.take_damage(dmg)
                    damage += dmg
        
        # Потери вертолетов (маловероятны)
        if random.random() < 0.05:
            self.helicopters -= 1
        
        return damage
    
    def get_status(self) -> dict:
        return {
            "helicopters_operational": self.helicopters,
            "sorties_used": self.sorties_used,
            "sorties_remaining": self.sorties_available - self.sorties_used
        }
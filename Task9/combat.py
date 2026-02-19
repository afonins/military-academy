"""
Боевые расчеты и взаимодействие - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import random
import math
from typing import List, Tuple
from config import TerrainType, CombatParams


class CombatUnitInterface:
    """Интерфейс для боевых единиц (наших и врага)"""
    def get_combat_power(self, terrain: TerrainType) -> float:
        raise NotImplementedError
    
    def take_damage(self, damage: float):
        raise NotImplementedError


class CombatEngine:
    """Движок боевых расчетов"""
    
    @staticmethod
    def calculate_combat(attacker, defender, terrain: TerrainType) -> Tuple[float, float]:
        """
        Расчет боя между двумя подразделениями
        Возвращает: (урон атакующему, урон защищающемуся)
        """
        # Получаем боевую мощь (с проверкой типа)
        att_power = CombatEngine._get_power(attacker, terrain)
        def_power = CombatEngine._get_power(defender, terrain)
        
        # Формула Ланчестера (упрощенная)
        if att_power > 0 and def_power > 0:
            ratio = att_power / def_power
            
            # Атакующий получает меньше урона при превосходстве
            att_losses = def_power * 0.1 / max(0.5, ratio)
            def_losses = att_power * 0.15 * min(2.0, ratio)
        else:
            att_losses = 0
            def_losses = 0
        
        # Случайность
        att_losses *= random.uniform(0.8, 1.2)
        def_losses *= random.uniform(0.8, 1.2)
        
        return att_losses, def_losses
    
    @staticmethod
    def _get_power(unit, terrain: TerrainType) -> float:
        """Получить боевую мощь любого объекта"""
        # Если это наше подразделение (есть метод)
        if hasattr(unit, 'get_combat_power'):
            return unit.get_combat_power(terrain)
        
        # Если это враг (словарь или объект с полями)
        if hasattr(unit, 'strength'):
            base = unit.strength * 0.5
            fort = getattr(unit, 'fortification', 0.5)
            return base * (1 + fort)
        
        # Если это словарь
        if isinstance(unit, dict):
            base = unit.get('strength', 50) * 0.5
            fort = unit.get('fortification', 0.5)
            return base * (1 + fort)
        
        return 50  # Значение по умолчанию
    
    @staticmethod
    def check_detection(unit, enemy, terrain_map) -> bool:
        """Проверка обнаружения"""
        # Координаты (работаем и с объектами, и со словарями)
        def get_x(obj):
            if hasattr(obj, 'x'):
                return obj.x
            if isinstance(obj, dict):
                return obj.get('x', 0)
            return getattr(obj, 'x', 0)
        
        def get_y(obj):
            if hasattr(obj, 'y'):
                return obj.y
            if isinstance(obj, dict):
                return obj.get('y', 0)
            return getattr(obj, 'y', 0)
        
        ux, uy = get_x(unit), get_y(unit)
        ex, ey = get_x(enemy), get_y(enemy)
        
        dist = math.sqrt((ux - ex)**2 + (uy - ey)**2)
        
        # Дальность видимости
        visibility = terrain_map.get_visibility(ux, uy)
        
        # Лес и город снижают заметность
        terrain = terrain_map.get_terrain(ex, ey)
        concealment = 1.0
        if terrain == TerrainType.FOREST:
            concealment = 0.5
        elif terrain == TerrainType.CITY:
            concealment = 0.3
        
        detection_range = visibility * concealment
        
        return dist <= detection_range
    
    @staticmethod
    def apply_damage(unit, damage: float):
        """Нанести урон подразделению"""
        if hasattr(unit, 'take_damage'):
            unit.take_damage(damage)
        elif isinstance(unit, dict):
            unit['strength'] = unit.get('strength', 100) - damage * 10
    
    @staticmethod
    def calculate_suppression(attacker, target) -> float:
        """Расчет подавления (моральный эффект)"""
        base_suppression = 0.1
        
        # Пулеметы и артиллерия лучше подавляют
        if hasattr(attacker, 'unit_type'):
            if "пулемет" in str(attacker.unit_type).lower():
                base_suppression = 0.2
        
        # Массированный огонь
        if hasattr(attacker, 'strength'):
            if attacker.strength > 50:
                base_suppression *= 1.5
        
        return min(0.5, base_suppression)  # Макс подавление 50%


class EnemyForce:
    """Условный противник (упрощенный)"""
    
    def __init__(self, config: dict):
        self.strongpoints = []
        self.minefields = []
        self.artillery = []
        
        # Создать опорные пункты
        for i in range(config["strongpoints"]):
            self.strongpoints.append({
                "id": f"OP-{i+1}",
                "x": 15 + random.uniform(-3, 3),
                "y": 10 + random.uniform(-3, 3),
                "strength": random.randint(30, 80),
                "fortification": random.uniform(0.3, 0.8),
                "is_alive": True
            })
        
        # Минные поля
        for i in range(config["minefields"]):
            self.minefields.append({
                "id": f"Mine-{i+1}",
                "x": 12 + random.uniform(-2, 2),
                "y": 8 + random.uniform(-2, 2),
                "radius": 1.5,
                "density": random.choice(["low", "medium", "high"])
            })
    
    def check_minefield(self, x: float, y: float) -> bool:
        """Проверка попадания в минное поле"""
        for mine in self.minefields:
            dist = math.sqrt((x - mine["x"])**2 + (y - mine["y"])**2)
            if dist < mine["radius"]:
                # Вероятность подрыва
                probs = {"low": 0.1, "medium": 0.3, "high": 0.5}
                if random.random() < probs.get(mine["density"], 0.2):
                    return True
        return False
    
    def update(self, time_hours: float):
        """Обновление состояния противника"""
        for sp in self.strongpoints:
            if sp["strength"] <= 0:
                sp["is_alive"] = False
    
    def get_active_defenders(self) -> List[dict]:
        """Получить активные опорные пункты"""
        return [sp for sp in self.strongpoints if sp["is_alive"]]
    
    def get_status(self) -> dict:
        alive = len([sp for sp in self.strongpoints if sp["is_alive"]])
        total_strength = sum(sp["strength"] for sp in self.strongpoints)
        return {
            "strongpoints_alive": alive,
            "total_strength": total_strength,
            "minefields_active": len(self.minefields)
        }
"""
Главный движок симуляции - ИСПРАВЛЕННАЯ ВЕРСИЯ (опечатка battalions)
"""

import random
from typing import List, Dict
from datetime import datetime, timedelta

from config import BRIGADE_STRUCTURE, ENEMY_FORCES, SIMULATION_CONFIG
from terrain import TerrainMap
from units import Battalion, CombatUnit
from support import ArtillerySupport, ReconSupport, AirSupport
from combat import CombatEngine, EnemyForce


class BrigadeSimulation:
    """Симуляция бригадной операции"""
    
    def __init__(self):
        print("Инициализация симуляции бригадной операции...")
        
        # Карта
        self.terrain = TerrainMap(
            SIMULATION_CONFIG["map_size_km"],
            SIMULATION_CONFIG["grid_size"]
        )
        
        # Наши силы (3 батальона)
        self.battalions: List[Battalion] = []
        for batt_config in BRIGADE_STRUCTURE["battalions"]:
            battalion = Battalion(batt_config)
            self.battalions.append(battalion)
        
        # Средства поддержки
        support = BRIGADE_STRUCTURE["support"]
        self.artillery = ArtillerySupport(support["artillery_battalion"])
        self.recon = ReconSupport(support["recon_company"])
        self.air = AirSupport(support["air_support"])
        
        # Противник
        self.enemy = EnemyForce(ENEMY_FORCES)
        
        # Боевой движок
        self.combat = CombatEngine()
        
        # Время
        self.current_time = datetime(2024, 6, 15, 6, 0)  # 06:00 начало
        self.end_time = self.current_time + timedelta(hours=24)
        self.time_step = timedelta(minutes=SIMULATION_CONFIG["time_step_minutes"])
        
        # Статистика
        self.events_log: List[Dict] = []
        self.phase = "подготовка"  # подготовка, наступление, бои, закрепление
        
        self._initial_deployment()
    
    def _initial_deployment(self):
        """Начальное развертывание"""
        # Батальоны в исходном районе (юго-запад)
        positions = [(3, 3), (5, 2), (4, 5)]
        
        for battalion, (x, y) in zip(self.battalions, positions):
            battalion.deploy(x, y, formation="line")
            
            # Установка целей
            battalion.set_objective(12, 10)  # Общее направление на восток
        
        self._log_event("Начальное развертывание завершено")
    
    def _log_event(self, message: str, level: str = "info"):
        """Запись события"""
        self.events_log.append({
            "time": self.current_time.strftime("%H:%M"),
            "phase": self.phase,
            "message": message,
            "level": level
        })
        print(f"[{self.current_time.strftime('%H:%M')}] {message}")
    
    def run(self):
        """Запуск симуляции"""
        print("\n" + "="*60)
        print("НАЧАЛО ОПЕРАЦИИ")
        print(f"Время старта: {self.current_time.strftime('%d.%m.%Y %H:%M')}")
        print("="*60)
        
        turn = 0
        
        while self.current_time < self.end_time and turn < 100:
            turn += 1
            time_hours = SIMULATION_CONFIG["time_step_minutes"] / 60
            
            # Определение фазы операции
            self._update_phase()
            
            # Обновление всех систем
            self._update_battalions(time_hours)  # ИСПРАВЛЕНО: battalions с 'i'
            self._update_support(time_hours)
            self._update_combat(time_hours)
            self._update_enemy(time_hours)
            
            # Проверка условий окончания
            if self._check_victory():
                self._log_event("ОПЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО", "success")
                break
            
            if self._check_defeat():
                self._log_event("ОПЕРАЦИЯ ПРОВАЛЕНА", "error")
                break
            
            # Время
            self.current_time += self.time_step
        
        self._print_final_report()
    
    def _update_phase(self):
        """Обновление фазы операции"""
        hour = self.current_time.hour
        
        if 6 <= hour < 8:
            new_phase = "артподготовка"
        elif 8 <= hour < 12:
            new_phase = "прорыв_обороны"
        elif 12 <= hour < 18:
            new_phase = "развитие_успеха"
        elif 18 <= hour < 20:
            new_phase = "закрепление"
        else:
            new_phase = "оборона_достигнутого"
        
        if new_phase != self.phase:
            self.phase = new_phase
            self._log_event(f"Переход в фазу: {self.phase}", "important")
    
    def _update_battalions(self, time_hours: float):  # ИСПРАВЛЕНО: правильное название
        """Обновление батальонов"""
        for battalion in self.battalions:
            battalion.update(self.terrain, time_hours)
            
            # Проверка минных полей
            for company in battalion.companies:
                if company.is_alive and company.target_x is not None:
                    if self.enemy.check_minefield(company.x, company.y):
                        damage = random.uniform(0.1, 0.3)
                        company.take_damage(damage)
                        self._log_event(
                            f"{company.id} подорвался на мине! Потери {damage*100:.0f}%",
                            "warning"
                        )
    
    def _update_support(self, time_hours: float):
        """Обновление средств поддержки"""
        # Разведка каждые 2 часа
        if self.current_time.hour % 2 == 0 and self.current_time.minute == 0:
            detected = self.recon.reconnaissance_mission(12, 10, radius=5)
            if detected:
                self._log_event(f"Разведка обнаружила {len(detected)} целей")
        
        # Артиллерия по расписанию фазы
        if self.phase == "артподготовка":
            # Массированный залп
            enemies = self.enemy.get_active_defenders()
            if enemies:
                target = random.choice(enemies)
                strike = self.artillery.calculate_strike(
                    target["x"], target["y"], urgency="high"
                )
                if strike:
                    # Создаем фиктивные объекты для урона
                    class DummyTarget:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y
                            self.strength = 100
                        def take_damage(self, dmg):
                            self.strength -= dmg * 2
                    
                    dummy = DummyTarget(target["x"], target["y"])
                    damage = self.artillery.execute_strike(strike, [dummy])
                    target["strength"] -= damage * 2
                    self._log_event(f"Артудар по {target['id']}: урон {damage:.1f}")
        
        # Авиация по запросу
        if self.phase in ["прорыв_обороны", "развитие_успеха"]:
            if random.random() < 0.3:  # 30% шанс запроса
                if self.air.request_strike(priority="high"):
                    enemies = self.enemy.get_active_defenders()
                    if enemies:
                        target = random.choice(enemies)
                        class DummyTarget:
                            def __init__(self, x, y):
                                self.x = x
                                self.y = y
                            def take_damage(self, dmg):
                                pass
                        
                        dummy = DummyTarget(target["x"], target["y"])
                        damage = self.air.execute_strike(
                            target["x"], target["y"], [dummy]
                        )
                        self._log_event(f"Авиаудар: урон {damage:.1f}")
    
    def _update_combat(self, time_hours: float):
        """Боевые контакты"""
        for battalion in self.battalions:
            for company in battalion.companies:
                if not company.is_alive:
                    continue
                
                # Поиск целей
                for enemy_sp in self.enemy.get_active_defenders():
                    # Проверка обнаружения
                    class DummyEnemy:
                        def __init__(self, data):
                            self.x = data["x"]
                            self.y = data["y"]
                            self.__dict__.update(data)
                        def take_damage(self, dmg):
                            self.strength -= dmg * 10
                    
                    dummy = DummyEnemy(enemy_sp)
                    
                    if self.combat.check_detection(company, dummy, self.terrain):
                        # Бой!
                        terrain = self.terrain.get_terrain(company.x, company.y)
                        
                        # Упрощенный расчет
                        if random.random() < 0.3:  # 30% шанс боя за такт
                            att_dmg, def_dmg = self.combat.calculate_combat(
                                company, dummy, terrain
                            )
                            
                            company.take_damage(def_dmg / 100)
                            enemy_sp["strength"] -= att_dmg * 5
                            
                            if random.random() < 0.1:  # 10% на сообщение
                                self._log_event(
                                    f"{company.id} в бою с {enemy_sp['id']}",
                                    "combat"
                                )
    
    def _update_enemy(self, time_hours: float):
        """Обновление противника"""
        self.enemy.update(time_hours)
        
        # Контратаки
        if self.phase == "развитие_успеха" and random.random() < 0.2:
            target_batt = random.choice(self.battalions)
            target_co = random.choice(target_batt.companies)
            damage = random.uniform(0.05, 0.15)
            target_co.take_damage(damage)
            self._log_event(
                f"Контратака противника по {target_co.id}!",
                "warning"
            )
    
    def _check_victory(self) -> bool:
        """Проверка победы"""
        # Все опорные пункты уничтожены
        alive_ops = len(self.enemy.get_active_defenders())
        return alive_ops == 0
    
    def _check_defeat(self) -> bool:
        """Проверка поражения"""
        # Потеряно более 50% личного состава
        total_strength = sum(
            sum(c.strength for c in b.companies if c.is_alive)
            for b in self.battalions
        )
        initial_strength = sum(b.total_strength for b in self.battalions)
        return total_strength < initial_strength * 0.5
    
    def _print_final_report(self):
        """Итоговый отчет"""
        print("\n" + "="*60)
        print("ИТОГОВЫЙ ОТЧЕТ ОПЕРАЦИИ")
        print("="*60)
        
        print(f"\nВремя окончания: {self.current_time.strftime('%H:%M')}")
        print(f"Финальная фаза: {self.phase}")
        
        print("\n--- НАШИ ПОТЕРИ ---")
        for battalion in self.battalions:
            status = battalion.get_status()
            print(f"{battalion.id}:")
            print(f"  Живых рот: {status['companies_alive']}/3")
            print(f"  Личный состав: {status['total_strength']}/{battalion.total_strength}")
            print(f"  Средняя мораль: {status['avg_morale']:.1f}%")
        
        print("\n--- СРЕДСТВА ПОДДЕРЖКИ ---")
        print("Артиллерия:", self.artillery.get_status())
        print("Разведка:", self.recon.get_status())
        print("Авиация:", self.air.get_status())
        
        print("\n--- ПРОТИВНИК ---")
        print(self.enemy.get_status())
        
        # Сохранение лога
        import json
        with open("brigade_simulation_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "events": self.events_log,
                "final_state": {
                    "time": self.current_time.isoformat(),
                    "phase": self.phase,
                    "battalions": [b.get_status() for b in self.battalions],
                    "enemy": self.enemy.get_status()
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nЛог сохранен: brigade_simulation_log.json")	
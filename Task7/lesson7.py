import random
import math
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SoldierState(Enum):
    IDLE = "ожидание"
    MOVING = "движение"
    FIRING = "стрельба"
    TAKING_COVER = "в укрытии"
    WOUNDED = "ранен"
    DEAD = "погиб"


class TargetPriority(Enum):
    MACHINE_GUN = 5
    OFFICER = 4
    SNIPER = 4
    RPG = 3
    RIFLEMAN = 2


@dataclass
class Soldier:
    id: int
    x: float
    y: float
    role: str
    state: SoldierState = SoldierState.IDLE
    health: float = 100.0
    morale: float = 100.0
    ammo: int = 150
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    decision_log: List[Dict] = field(default_factory=list)
    
    def distance_to(self, x, y):
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def decide(self, enemies, allies):
        alive = [e for e in enemies if e.health > 0]
        if not alive:
            return {"action": "hold", "reason": "Нет живых противников"}
        
        nearest = min(alive, key=lambda e: self.distance_to(e.x, e.y))
        dist = self.distance_to(nearest.x, nearest.y)
        
        if self.health < 30:
            return {"action": "retreat", "reason": "Критический уровень здоровья"}
        
        if dist > 100:
            return {
                "action": "advance", 
                "target": (nearest.x, nearest.y),
                "reason": f"Сокращение дистанции до {nearest.priority.name} ({dist:.0f}м)"
            }
        
        if dist < 30:
            return {"action": "retreat", "reason": "Слишком близко к противнику"}
        
        if self.ammo > 0:
            return {
                "action": "fire",
                "target": nearest,
                "reason": f"Атака {nearest.priority.name} на дистанции {dist:.0f}м"
            }
        
        return {"action": "hold", "reason": "Нет боеприпасов"}
    
    def execute(self, decision, enemies, allies):
        action = decision["action"]
        
        if action == "advance" and "target" in decision:
            tx, ty = decision["target"]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                speed = 15
                self.x += (dx / dist) * min(speed, dist * 0.5)
                self.y += (dy / dist) * min(speed, dist * 0.5)
                self.state = SoldierState.MOVING
                return f"Продвижение к ({self.x:.0f}, {self.y:.0f})"
                
        elif action == "fire" and "target" in decision:
            target = decision["target"]
            self.ammo -= random.randint(1, 3)
            self.state = SoldierState.FIRING
            
            hit_chance = 0.6 + (self.morale / 500)
            if random.random() < hit_chance:
                dmg = random.uniform(20, 40)
                if self.role == "пулеметчик":
                    dmg *= 1.5
                elif self.role == "снайпер":
                    dmg *= 2.0
                target.health -= dmg
                return f"ПОПАДАНИЕ в {target.priority.name}! Урон {dmg:.1f}"
            return "Промах"
            
        elif action == "retreat":
            self.x -= 10
            self.state = SoldierState.MOVING
            return "Отход на безопасную дистанцию"
        
        else:
            self.state = SoldierState.IDLE
            return "Ожидание"
    
    def update_morale(self, allies):
        alive = len([a for a in allies if a.health > 0])
        commander_alive = any(a.role == "командир" and a.health > 0 for a in allies)
        
        base = 50 + alive * 5
        if commander_alive:
            base += 20
        
        self.morale += (base - self.morale) * 0.1
        self.morale = max(0, min(100, self.morale))


@dataclass
class Enemy:
    x: float
    y: float
    priority: TargetPriority
    health: float = 100.0
    is_alive: bool = True
    
    def turn(self, soldiers):
        alive = [s for s in soldiers if s.health > 0]
        if not alive:
            return
        
        target = min(alive, key=lambda s: math.sqrt((s.x - self.x)**2 + (s.y - self.y)**2))
        dist = math.sqrt((target.x - self.x)**2 + (target.y - self.y)**2)
        
        if dist < 150 and random.random() < 0.3:
            dmg = random.uniform(10, 25)
            target.health -= dmg
            target.morale -= 5
            return f"ВРАГ попал в {target.role}! Урон {dmg:.1f}"
        return None


def print_field(soldiers, enemies, turn):
    print(f"\n{'='*70}")
    print(f"ХОД {turn}")
    print(f"{'='*70}")
    
    width, height = 80, 20
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    def scale_x(x): return int((x / 500) * (width - 1))
    def scale_y(y): return int((y / 600) * (height - 1))
    
    for ox, oy in [(200, 300), (350, 400), (300, 200)]:
        sx, sy = scale_x(ox), scale_y(oy)
        if 0 <= sx < width and 0 <= sy < height:
            grid[sy][sx] = '#'
    
    for e in enemies:
        if e.health > 0:
            sx, sy = scale_x(e.x), scale_y(e.y)
            if 0 <= sx < width and 0 <= sy < height:
                symbols = {5: 'M', 4: 'O', 3: 'R', 2: 'V'}
                grid[sy][sx] = symbols.get(e.priority.value, 'V')
    
    for s in soldiers:
        if s.health > 0:
            sx, sy = scale_x(s.x), scale_y(s.y)
            if 0 <= sx < width and 0 <= sy < height:
                symbols = {
                    "командир": "K", "пулеметчик": "P", "снайпер": "S",
                    "стрелок": "B", "гранатометчик": "G", "связист": "Sv"
                }
                grid[sy][sx] = symbols.get(s.role, "B")
    
    print("\n  " + "".join(f"{i%10}" for i in range(width)))
    for i, row in enumerate(grid):
        print(f"{i:2}" + "".join(row))
    print("  K=командир, P=пулеметчик, S=снайпер, B=стрелок, G=гранатометчик")
    print("  M=пулемет, O=офицер, R=РПГ, V=стрелок, #=укрытие")
    
    print(f"\nВАШИ ({len([s for s in soldiers if s.health > 0])}/6 живы):")
    for s in soldiers:
        status = "X" if s.health <= 0 else "+" if s.health > 70 else "~" if s.health > 30 else "-"
        print(f"{status} {s.role:12} HP:{s.health:5.1f} MP:{s.morale:5.1f} {s.state.value}")
    
    print(f"\nВРАГИ ({len([e for e in enemies if e.health > 0])}/4 живы):")
    for e in enemies:
        status = "X" if e.health <= 0 else "-"
        print(f"{status} {e.priority.name:12} HP:{e.health:5.1f}")


def main():
    print("="*70)
    print("АГЕНТ УПРАВЛЕНИЯ ОТДЕЛЕНИЕМ")
    print("="*70)
    
    roles = ["командир", "пулеметчик", "снайпер", "стрелок", "гранатометчик", "связист"]
    soldiers = []
    for i, role in enumerate(roles):
        x = 50 + random.randint(-20, 20)
        y = 200 + i * 80 + random.randint(-10, 10)
        soldiers.append(Soldier(i, x, y, role))
    
    enemies = [
        Enemy(250, 250, TargetPriority.MACHINE_GUN),
        Enemy(300, 350, TargetPriority.OFFICER),
        Enemy(280, 200, TargetPriority.RIFLEMAN),
        Enemy(320, 300, TargetPriority.SNIPER),
    ]
    
    turn = 0
    log = []
    
    while turn < 30:
        turn += 1
        
        for s in soldiers:
            if s.health <= 0:
                continue
            
            s.update_morale(soldiers)
            decision = s.decide(enemies, soldiers)
            result = s.execute(decision, enemies, soldiers)
            
            log.append({
                "turn": turn,
                "soldier": s.role,
                "action": decision["action"],
                "reason": decision["reason"],
                "result": result
            })
        
        enemy_actions = []
        for e in enemies:
            if e.health > 0:
                action = e.turn(soldiers)
                if action:
                    enemy_actions.append(action)
        
        print_field(soldiers, enemies, turn)
        
        print(f"\nСОБЫТИЯ ХОДА {turn}:")
        recent = [e for e in log if e["turn"] == turn]
        for e in recent[-4:]:
            print(f"  {e['soldier']:12} -> {e['action']:10} | {e['result'][:50]}")
        
        if enemy_actions:
            print("  ДЕЙСТВИЯ ПРОТИВНИКА:")
            for a in enemy_actions:
                print(f"    {a}")
        
        alive_soldiers = [s for s in soldiers if s.health > 0]
        alive_enemies = [e for e in enemies if e.health > 0]
        
        if not alive_enemies:
            print("\n" + "="*70)
            print("ПОБЕДА! Все противники уничтожены!")
            print(f"Выжило бойцов: {len(alive_soldiers)}/6")
            break
        elif not alive_soldiers:
            print("\n" + "="*70)
            print("ПОРАЖЕНИЕ. Отделение уничтожено.")
            break
        
        cmd = input("\n[Enter - следующий ход | I - инфо | Q - выход]: ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 'i':
            print(f"\nПодробный лог (последние 10 записей):")
            for e in log[-10:]:
                print(f"  Ход {e['turn']}: {e['soldier']} - {e['reason']}")
    
    with open("squad_battle.json", "w", encoding="utf-8") as f:
        json.dump({
            "turns": turn,
            "survivors": len([s for s in soldiers if s.health > 0]),
            "enemies_destroyed": len([e for e in enemies if e.health <= 0]),
            "log": log
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nЛог сохранен: squad_battle.json")


if __name__ == "__main__":
    main()
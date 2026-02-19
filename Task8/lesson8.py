import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class ThreatLevel(Enum):
    LOW = "низкая"
    MEDIUM = "средняя" 
    HIGH = "высокая"
    CRITICAL = "критическая"


@dataclass
class XAIReport:
    decision_id: str
    timestamp: str
    decision: str
    threat_level: ThreatLevel
    tactical_summary: str
    probabilities: Dict[str, float]
    confidence: float
    key_factors: List[tuple]
    lime_rules: List[str]
    shap_values: Dict[str, float]
    attention_map: List[List[float]]
    human_override: bool = False


class MilitaryXAI:
    def __init__(self):
        self.audit_log = []
        self.counter = 0
        
        self.feature_names = {
            'tank_count': 'Танки',
            'apc_count': 'БТР/БМП', 
            'artillery_count': 'Артиллерия',
            'distance_km': 'Дистанция, км',
            'movement_speed': 'Скорость, км/ч',
            'electronic_activity': 'РЭБ активность'
        }
    
    def analyze(self, data: Dict, operator: str = "auto") -> XAIReport:
        self.counter += 1
        dec_id = f"XAI-{self.counter:04d}"
        
        threat = self._calc_threat(data)
        
        if threat > 0.8:
            level, decision = ThreatLevel.CRITICAL, "Боевая готовность, вызов резервов"
        elif threat > 0.6:
            level, decision = ThreatLevel.HIGH, "Усилить разведку, готовность к обороне"
        elif threat > 0.3:
            level, decision = ThreatLevel.MEDIUM, "Продолжить наблюдение"
        else:
            level, decision = ThreatLevel.LOW, "Обычное наблюдение"
        
        lime_rules = []
        if data.get('tank_count', 0) >= 3:
            lime_rules.append("Причина: обнаружено 3+ танка (вес +30%)")
        if data.get('distance_km', 100) < 15:
            lime_rules.append("Причина: дистанция <15 км (вес +20%)")
        if data.get('movement_speed', 0) > 20:
            lime_rules.append("Причина: высокая скорость = наступление")
        
        shap = self._calc_shap(data, threat)
        key_factors = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        report = XAIReport(
            decision_id=dec_id,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            decision=decision,
            threat_level=level,
            tactical_summary=self._make_summary(data, level),
            probabilities={
                "Атака в ближайший час": round(threat * 0.8, 2),
                "Артобстрел": round(threat * 0.5, 2),
                "Маневр/учения": round(1 - threat, 2)
            },
            confidence=round(min(0.95, 0.5 + threat * 0.5), 2),
            key_factors=key_factors,
            lime_rules=lime_rules,
            shap_values=shap,
            attention_map=self._make_heatmap(data)
        )
        
        self.audit_log.append(report)
        return report
    
    def _calc_threat(self, d: Dict) -> float:
        score = 0.0
        score += min(d.get('tank_count', 0) * 0.15, 0.4)
        score += min(d.get('apc_count', 0) * 0.08, 0.3)
        score += min(d.get('artillery_count', 0) * 0.1, 0.3)
        if d.get('distance_km', 50) < 15:
            score += 0.2
        if d.get('movement_speed', 0) > 20:
            score += 0.15
        if d.get('electronic_activity', 0) > 0.7:
            score += 0.15
        return min(score, 1.0)
    
    def _calc_shap(self, data: Dict, total: float) -> Dict[str, float]:
        shap = {}
        for key, name in self.feature_names.items():
            without = data.copy()
            without[key] = 0
            score_without = self._calc_threat(without)
            shap[name] = round(total - score_without, 3)
        return shap
    
    def _make_summary(self, data: Dict, level: ThreatLevel) -> str:
        parts = [f"Угроза: {level.value}."]
        if data.get('tank_count'):
            parts.append(f"Танков: {data['tank_count']}.")
        if data.get('movement_speed', 0) > 20:
            parts.append("Высокая скорость = наступление.")
        return " ".join(parts)
    
    def _make_heatmap(self, data: Dict) -> List[List[float]]:
        grid = [[0.1 for _ in range(5)] for _ in range(5)]
        if data.get('tank_count', 0) > 0:
            for i in range(1, 4):
                for j in range(1, 4):
                    grid[i][j] = 0.8
        return grid
    
    def detect_attack(self, data: Dict) -> Dict:
        issues = []
        
        if data.get('tank_count', 0) > 10 and data.get('distance_km', 0) < 5:
            issues.append("Подозрительно: >10 танков на дистанции <5 км")
        
        if data.get('tank_count', 0) > 0 and data.get('electronic_activity', 1) == 0:
            issues.append("Подозрительно: танки без радиоактивности")
        
        if data.get('movement_speed', 0) > 60:
            issues.append("Нереальная скорость для танков")
        
        return {
            "attack_detected": len(issues) > 0,
            "issues": issues,
            "action": "Проверить другими источниками!" if issues else "Данные нормальные"
        }
    
    def human_override(self, report_id: str, operator: str, new_decision: str, reason: str):
        for r in self.audit_log:
            if r.decision_id == report_id:
                r.human_override = True
                print(f"\nОператор {operator} изменил решение:")
                print(f"   Было:  {r.decision}")
                print(f"   Стало: {new_decision}")
                print(f"   Почему: {reason}")
                return
    
    def print_report(self, r: XAIReport):
        print(f"\n{'='*60}")
        print(f"РЕШЕНИЕ {r.decision_id} | {r.timestamp}")
        print(f"{'='*60}")
        
        print(f"\nРЕЗЮМЕ ДЛЯ КОМАНДИРА:")
        print(f"   {r.tactical_summary}")
        print(f"   Решение: {r.decision}")
        print(f"   Уверенность: {r.confidence*100:.0f}%")
        
        print(f"\nВЕРОЯТНОСТИ:")
        for scenario, prob in r.probabilities.items():
            bar = "█" * int(prob * 20)
            print(f"   {scenario:20} {bar} {prob*100:.0f}%")
        
        print(f"\nКЛЮЧЕВЫЕ ФАКТОРЫ:")
        for factor, weight in r.key_factors:
            direction = "повышает" if weight > 0 else "снижает"
            print(f"   {factor} {direction} риск (вклад: {abs(weight)*100:.1f}%)")
        
        print(f"\nТЕХНИЧЕСКИЕ ДЕТАЛИ:")
        print("   LIME-правила:")
        for rule in r.lime_rules:
            print(f"      {rule}")
        
        print("   Карта внимания:")
        for row in r.attention_map:
            line = "      "
            for val in row:
                if val > 0.7: line += "██"
                elif val > 0.3: line += "▓▓"
                else: line += "░░"
            print(line)
        
        status = "ИЗМЕНЕНО ОПЕРАТОРОМ" if r.human_override else "ПОДТВЕРЖДЕНО"
        print(f"\nСТАТУС: {status}")
        print(f"{'='*60}")


def main():
    print("="*60)
    print("XAI ДЛЯ ВОЕННЫХ СИСТЕМ")
    print("="*60)
    
    xai = MilitaryXAI()
    
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 1: Наступающая колонна")
    print("="*60)
    
    data1 = {
        'tank_count': 5, 'apc_count': 8, 'artillery_count': 3,
        'distance_km': 12, 'movement_speed': 25,
        'electronic_activity': 0.8
    }
    
    check1 = xai.detect_attack(data1)
    print(f"\nПроверка данных: {check1['action']}")
    
    if not check1['attack_detected']:
        report1 = xai.analyze(data1)
        xai.print_report(report1)
    
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 2: Поддельные данные")
    print("="*60)
    
    data2 = {
        'tank_count': 15,
        'apc_count': 0, 'artillery_count': 0,
        'distance_km': 3,
        'movement_speed': 0,
        'electronic_activity': 0
    }
    
    check2 = xai.detect_attack(data2)
    print(f"\nОБНАРУЖЕНЫ АНОМАЛИИ:")
    for issue in check2['issues']:
        print(f"   {issue}")
    print(f"\nДействие: {check2['action']}")
    print("(Анализ ИИ заблокирован - требуется проверка)")
    
    print("\n" + "="*60)
    print("СЦЕНАРИЙ 3: Оператор отменяет решение ИИ")
    print("="*60)
    
    data3 = {
        'tank_count': 2, 'apc_count': 3,
        'distance_km': 25, 'movement_speed': 10,
        'electronic_activity': 0.3
    }
    
    report3 = xai.analyze(data3)
    xai.print_report(report3)
    
    xai.human_override(
        report3.decision_id,
        "майор Иванов",
        "Не привлекать резервы",
        "Это учения соседней части"
    )
    
    print("\n" + "="*60)
    print("ИТОГОВЫЙ АУДИТ")
    print("="*60)
    
    total = len(xai.audit_log)
    overridden = len([r for r in xai.audit_log if r.human_override])
    
    print(f"\nВсего решений: {total}")
    print(f"Подтверждено ИИ: {total - overridden}")
    print(f"Изменено оператором: {overridden}")
    
    with open("xai_audit.json", "w", encoding="utf-8") as f:
        json.dump([{
            "id": r.decision_id,
            "time": r.timestamp,
            "threat": r.threat_level.value,
            "decision": r.decision,
            "confidence": r.confidence,
            "factors": r.key_factors,
            "human_override": r.human_override
        } for r in xai.audit_log], f, ensure_ascii=False, indent=2)
    
    print(f"\nАудит сохранен: xai_audit.json")


if __name__ == "__main__":
    main()
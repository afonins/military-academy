import json
import re
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class IntelReport:
    id: str
    timestamp: str
    coordinates: Tuple[float, float]
    location: str
    enemy_forces: Dict[str, int]
    equipment: Dict[str, int]
    activity: str
    confidence: str
    source: str
    raw_text: str
    
    def to_dict(self):
        return asdict(self)


class IntelDataGenerator:
    def __init__(self):
        self.reports = []
        
    def generate(self, count: int = 10) -> List[IntelReport]:
        templates = [
            {
                "coords": (48.8566, 37.6173),
                "location": "северная окраина н.п. Красное",
                "forces": {"пехота": 15, "саперы": 3},
                "equip": {"танк": 3, "БТР": 2, "грузовик": 4},
                "activity": "инженерное оборудование позиций",
                "conf": "высокая"
            },
            {
                "coords": (48.7234, 37.8123),
                "location": "высота 215.3, лесной массив",
                "forces": {"пехота": 8},
                "equip": {"БМП": 4, "БТР": 1},
                "activity": "засада на дороге подхода",
                "conf": "средняя"
            },
            {
                "coords": (48.9123, 37.4567),
                "location": "южная промзона",
                "forces": {"пехота": 25, "снайперы": 2},
                "equip": {"танк": 5, "БТР": 6, "САУ": 2},
                "activity": "сосредоточение перед наступлением",
                "conf": "высокая"
            },
            {
                "coords": (48.6543, 37.7890),
                "location": "переправа через р. Северский Донец",
                "forces": {"инженеры": 10},
                "equip": {"понтон": 2, "грузовик": 8, "БТР": 3},
                "activity": "оборудование переправы",
                "conf": "высокая"
            },
            {
                "coords": (48.8123, 37.2345),
                "location": "ж/д станция 'Мирная'",
                "forces": {"охрана": 12},
                "equip": {"вагон": 15, "тепловоз": 2},
                "activity": "погрузка техники на платформы",
                "conf": "средняя"
            },
            {
                "coords": (48.7345, 37.6789),
                "location": "база отдыха 'Лесная'",
                "forces": {"штаб": 8, "связисты": 4},
                "equip": {"КШМ": 3, "радиостанция": 5},
                "activity": "развертывание пунктов управления",
                "conf": "высокая"
            },
            {
                "coords": (48.8901, 37.8901),
                "location": "траса М-03, км 45",
                "forces": {"пехота": 6},
                "equip": {"танк": 2, "БТР": 2},
                "activity": "передвижение колонны на запад",
                "conf": "низкая"
            },
            {
                "coords": (48.5678, 37.4567),
                "location": "н.п. Степное, школа №3",
                "forces": {"пехота": 30, "минометчики": 6},
                "equip": {"танк": 4, "БТР": 5, "миномет": 6},
                "activity": "занятие опорного пункта",
                "conf": "высокая"
            },
            {
                "coords": (48.6789, 37.7890),
                "location": "склад ГСМ, северная промзона",
                "forces": {"охрана": 10, "пожарные": 4},
                "equip": {"цистерна": 12, "грузовик": 6},
                "activity": "перекачка топлива",
                "conf": "средняя"
            },
            {
                "coords": (48.7890, 37.1234),
                "location": "аэродром временного базирования",
                "forces": {"техники": 15, "охрана": 20},
                "equip": {"вертолет": 4, "ЗРК": 2, "радар": 1},
                "activity": "обслуживание авиации",
                "conf": "высокая"
            }
        ]
        
        for i, t in enumerate(templates[:count], 1):
            raw = self._format(t, i)
            report = IntelReport(
                id=f"RPT-{datetime.datetime.now().year}-{i:03d}",
                timestamp=datetime.datetime.now().isoformat(),
                coordinates=t["coords"],
                location=t["location"],
                enemy_forces=t["forces"],
                equipment=t["equip"],
                activity=t["activity"],
                confidence=t["conf"],
                source="БПЛА 'Орлан-10'" if i % 2 == 0 else "Агентурная разведка",
                raw_text=raw
            )
            self.reports.append(report)
            
        return self.reports
    
    def _format(self, data: dict, num: int) -> str:
        forces = ", ".join([f"{k}: {v} чел" for k, v in data["forces"].items()])
        equip = ", ".join([f"{k}: {v} ед" for k, v in data["equip"].items()])
        
        return f"""ДОНЕСЕНИЕ №{num:03d}
Время: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}
Координаты: {data['coords'][0]:.4f} N, {data['coords'][1]:.4f} E
Район: {data['location']}

СИЛЫ:
Живая сила: {forces}
Техника: {equip}

ДЕЯТЕЛЬНОСТЬ: {data['activity']}

ДОВЕРИЕ: {data['conf'].upper()}
ИСТОЧНИК: {'БПЛА' if num % 2 == 0 else 'Агентурная разведка'}"""


class KnowledgeBase:
    def __init__(self):
        self.docs = []
        self._load()
        
    def _load(self):
        docs = [
            {
                "title": "Устав РВиА. Тактика танковых подразделений",
                "content": "При обнаружении танковой роты (3-4 танка) противник, вероятно, готовит наступление на узком участке фронта или контратаку. Рекомендуется усиление разведки и готовность резервов.",
                "source": "Устав РВиА, 2023"
            },
            {
                "title": "Пособие по противодиверсионной борьбе",
                "content": "Обнаружение саперных групп указывает на подготовку проходов в минных полях или установку заграждений. Требуется усиление наблюдения и проверка маршрутов.",
                "source": "ПУ-34"
            },
            {
                "title": "Тактика применения БПЛА",
                "content": "Разведка с БПЛА обеспечивает выявление позиций ПВО, корректировку огня артиллерии и оценку результатов ударов. Ограничения: погода и РЭБ противника.",
                "source": "Руководство по эксплуатации БПЛА"
            },
            {
                "title": "Анализ сосредоточения войск",
                "content": "Признаки подготовки наступления: накопление техники в 15-20 км от фронта, развертывание ПУ, инженерное оборудование позиций артиллерии, переправы через водные преграды. Время готовности: 24-48 часов.",
                "source": "Аналитическая записка ГРУ №45/2023"
            },
            {
                "title": "Противодействие засадам",
                "content": "Признаки засады в лесу: отсутствие видимого перемещения при наличии техники, свежие следы гусеничной техники в лес, нарушение растительности. Меры: обход, зачистка с тепловизорами, огневое прикрытие.",
                "source": "Наставление по тактике, ч. 3"
            },
            {
                "title": "Оценка инженерной подготовки",
                "content": "Оборудование переправ указывает на намерение форсировать водную преграду или создание резервных маршрутов. Переправы — уязвимое звено, требуют прикрытия ПВО.",
                "source": "Инженерный устав ВС РФ"
            },
            {
                "title": "Анализ железнодорожных перевозок",
                "content": "Погрузка техники на ж/д платформы: массовое переброшение резервов или эвакуация поврежденной техники. Скорость до 500 км/сутки, сложно перехватить.",
                "source": "Справочник военного железнодорожника"
            },
            {
                "title": "Разведка пунктов управления",
                "content": "Признаки штаба: концентрация КШМ, антенны радиостанций, усиленная охрана в 2-3 раза выше нормы. Приоритет цели: высокий.",
                "source": "Методичка ГРУ по целеуказанию"
            },
            {
                "title": "Оценка авиационной активности",
                "content": "Вертолеты на временном аэродроме: ударные готовятся к поддержке наступления, транспортные — к переброске десанта, разведывательные — к уточнению целей. Дальность до 300 км.",
                "source": "Справочник ВВС и ПВО"
            },
            {
                "title": "Анализ складов ГСМ",
                "content": "Склады горюче-смазочных материалов обеспечивают операции на 7-10 суток, являются критичной уязвимостью и требуют постоянной охраны. Уничтожение парализует механизированные части.",
                "source": "Тыловой устав ВС РФ"
            },
            {
                "title": "Противодействие снайперам",
                "content": "Снайперские пары работают в пригородной застройке или высотных зданиях на дистанции 300-800 м. Цели: офицеры, пулеметчики, операторы ПТРК. Меры: дымовые завесы, бронежилеты, контрснайперы.",
                "source": "Наставление по городским действиям"
            },
            {
                "title": "Оценка минометных позиций",
                "content": "Минометы (82-мм, 120-мм) в опорном пункте имеют дальность 4-7 км, развертываются за 2-3 минуты, наносят массированные залпы. Требуется контрбатарейная борьба или РЭБ.",
                "source": "Артиллерийский устав"
            },
            {
                "title": "Тактика противодействия БПЛА",
                "content": "Обнаружение станций РЭБ и радаров: противник усиливает ПВО, возможно применение РЭБ против нашей разведки. Меры: смена частот, кабельные линии, ложные цели.",
                "source": "Инструкция РЭБ ВКС"
            },
            {
                "title": "Анализ передвижения колонн",
                "content": "Перемещение танков и БТР по трассам: дневное — спешка или отсутствие угрозы с воздуха, ночное — попытка скрыть перегруппировку, без прикрытия ПВО — уязвимая цель. Рекомендуется удар авиации или ПТРК.",
                "source": "Наставление по тактике"
            },
            {
                "title": "Оценка опорных пунктов",
                "content": "Занятие школ, больниц, админзданий: использование гражданских объектов как щита, укрепленные позиции внутри, сложность штурма. Требуется точечное оружие и переговоры об эвакуации.",
                "source": "Правила ведения боевых действий"
            },
            {
                "title": "Прогнозирование действий противника",
                "content": "Шаблоны обороны: подготовка позиций 3-5 суток, выставление наблюдателей и снайперов, минирование подходов, размещение резервов во втором эшелоне. Прорыв требует превосходства 3:1.",
                "source": "Аналитический центр ГШ ВС РФ"
            },
            {
                "title": "Координаты и ориентиры",
                "content": "Система WGS-84: точность БПЛА 'Орлан-10' ±50 м, агентурная разведка зависит от источника. Уточнение по крупным ориентирам. Важно: перепроверка несколькими источниками.",
                "source": "Инструкция топографической службы"
            },
            {
                "title": "Оценка доверия к данным",
                "content": "Уровни достоверности: ВЫСОКАЯ — визуальный контакт с подтверждением, СРЕДНЯЯ — данные агентуры, НИЗКАЯ — слухи. При низкой достоверности требуется усиление разведки.",
                "source": "Руководство разведчика"
            },
            {
                "title": "Взаимодействие родов войск",
                "content": "Типичные сочетания: танки + пехота = наступление, БТР + саперы = инженерная разведка, артиллерия + разведка = корректировка огня. Анализ сочетаний позволяет предсказать намерения.",
                "source": "Устав по взаимодействию"
            },
            {
                "title": "Ограничения разведки",
                "content": "Разведка не дает: точные планы противника, намерения командования, моральное состояние личного состава. Выводы требуют критического анализа и не являются приказами.",
                "source": "Методические рекомендации ГРУ"
            }
        ]
        
        for doc in docs:
            doc["id"] = hashlib.md5(doc["title"].encode()).hexdigest()[:8]
            self.docs.append(doc)
            
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_lower = query.lower()
        scores = []
        
        for doc in self.docs:
            score = 0
            content = doc["content"].lower()
            
            for word in set(query_lower.split()):
                if len(word) > 3:
                    score += content.count(word)
            
            if query_lower in content:
                score += 10
                
            if score > 0:
                scores.append((score, doc))
                
        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scores[:top_k]]


class SafetyFilter:
    FORBIDDEN = [
        r'\b(атаковать|штурмовать|уничтожить|ликвидировать)\s+(сейчас|немедленно|срочно)\b',
        r'\b(открыть\s+огонь|стрелять|бить)\s+по\s+(координатам|цели)\b',
        r'\b(выслать|направить)\s+(удар|ракеты|авиацию)\b',
        r'\b(приказ\s*:|приказываю|велено)\b',
        r'\b(всем\s+подразделениям|всем\s+частям)\b',
    ]
    
    DISCLAIMERS = [
        "Анализ носит рекомендательный характер",
        "Решение принимает командир",
        "Требуется дополнительная проверка",
    ]
    
    def __init__(self):
        self.violations = []
        
    def check_input(self, query: str) -> Tuple[bool, str]:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["прикажи", "приказ", "вели", "выполни"]):
            return False, "Запрос отклонён: система не даёт приказов, только анализ."
            
        if "координаты для огня" in query_lower or "целеуказание" in query_lower:
            return False, "Запрос отклонён: система не производит целеуказание."
            
        return True, "OK"
    
    def check_output(self, response: str, sources: List[Dict]) -> Tuple[bool, str]:
        response_lower = response.lower()
        
        for pattern in self.FORBIDDEN:
            if re.search(pattern, response_lower):
                self.violations.append({
                    "pattern": pattern,
                    "response": response[:100],
                    "time": datetime.datetime.now().isoformat()
                })
                return False, "Ответ заблокирован: обнаружен запрещённый паттерн."
        
        if not sources:
            return False, "Ответ заблокирован: отсутствуют подтверждающие документы."
            
        return True, "OK"
    
    def add_disclaimers(self, response: str) -> str:
        disclaimer = "\n\n---\nВАЖНО:\n"
        for i, disc in enumerate(self.DISCLAIMERS, 1):
            disclaimer += f"{i}. {disc}\n"
        return response + disclaimer


class IntelAssistant:
    def __init__(self, audit_path: str = "audit.json"):
        self.kb = KnowledgeBase()
        self.filter = SafetyFilter()
        self.audit_path = Path(audit_path)
        self.audit = []
        self.reports = []
        
        if self.audit_path.exists():
            with open(self.audit_path, 'r', encoding='utf-8') as f:
                self.audit = json.load(f)
                
    def load_reports(self, reports: List[IntelReport]):
        self.reports = reports
        
    def query(self, user_query: str, user_id: str = "anon") -> Dict:
        timestamp = datetime.datetime.now().isoformat()
        req_id = hashlib.md5(f"{user_id}{timestamp}{user_query}".encode()).hexdigest()[:12]
        
        ok, msg = self.filter.check_input(user_query)
        if not ok:
            self._log(req_id, user_id, user_query, None, "BLOCKED_INPUT", msg)
            return {
                "success": False,
                "response": msg,
                "sources": [],
                "reports": [],
                "check": "BLOCKED",
                "time": timestamp
            }
        
        docs = self.kb.search(user_query, top_k=3)
        reports = self._find_reports(user_query)
        response = self._generate(user_query, docs, reports)
        
        ok, msg = self.filter.check_output(response, docs)
        if not ok:
            self._log(req_id, user_id, user_query, response, "BLOCKED_OUTPUT", msg)
            return {
                "success": False,
                "response": msg,
                "sources": [],
                "reports": [],
                "check": "BLOCKED",
                "time": timestamp
            }
        
        response = self.filter.add_disclaimers(response)
        self._log(req_id, user_id, user_query, response, "SUCCESS", "OK", 
                 sources=[d["title"] for d in docs])
        
        return {
            "success": True,
            "response": response,
            "sources": docs,
            "reports": [r.id for r in reports],
            "check": "PASSED",
            "time": timestamp
        }
    
    def _find_reports(self, query: str) -> List[IntelReport]:
        query_lower = query.lower()
        related = []
        
        for r in self.reports:
            score = 0
            text = r.raw_text.lower()
            
            for equip in r.equipment.keys():
                if equip in query_lower:
                    score += 5
                    
            if r.location.lower() in query_lower:
                score += 3
                
            if r.activity.lower() in query_lower:
                score += 2
                
            if score > 0:
                related.append((score, r))
                
        related.sort(reverse=True, key=lambda x: x[0])
        return [r for _, r in related[:3]]
    
    def _generate(self, query: str, docs: List[Dict], reports: List[IntelReport]) -> str:
        query_lower = query.lower()
        
        if "что" in query_lower and ("означает" in query_lower or "указывает" in query_lower):
            return self._analysis(docs, reports)
        elif "как" in query_lower and ("действовать" in query_lower or "реагировать" in query_lower):
            return self._recommend(docs, reports)
        elif "сводка" in query_lower or "обстановка" in query_lower:
            return self._summary(reports)
        else:
            return self._general(docs, reports)
    
    def _analysis(self, docs: List[Dict], reports: List[IntelReport]) -> str:
        response = "АНАЛИЗ ОБСТАНОВКИ\n\n"
        
        if reports:
            response += "На основе донесений:\n"
            for r in reports[:2]:
                response += f"- {r.id}: {r.location}, {sum(r.equipment.values())} ед. техники ({r.confidence})\n"
            response += "\n"
        
        if docs:
            response += "Тактический вывод:\n"
            for doc in docs[:2]:
                lines = [l.strip() for l in doc["content"].strip().split('\n') if l.strip() and not l.strip().startswith('-')]
                if lines:
                    response += f"• {lines[0]}\n"
                    
        return response
    
    def _recommend(self, docs: List[Dict], reports: List[IntelReport]) -> str:
        response = "РЕКОМЕНДАЦИИ\n\n"
        response += "На основании уставных документов:\n"
        
        for doc in docs[:2]:
            content = doc["content"]
            if "Рекомендуется" in content:
                start = content.find("Рекомендуется")
                end = content.find("\n\n", start)
                if end == -1:
                    end = len(content)
                response += f"• {content[start:end].strip()}\n"
            elif "Требуется" in content:
                start = content.find("Требуется")
                end = content.find("\n\n", start)
                if end == -1:
                    end = len(content)
                response += f"• {content[start:end].strip()}\n"
            else:
                first = content.split('.')[0] + '.'
                response += f"• {first}\n"
                
        response += "\nПримечание: Конкретные действия определяет командир."
        return response
    
    def _summary(self, reports: List[IntelReport]) -> str:
        response = "СВОДКА ОБСТАНОВКИ\n\n"
        total = {"танк": 0, "БТР": 0, "БМП": 0, "САУ": 0, "грузовик": 0}
        
        for r in reports:
            for equip, count in r.equipment.items():
                if equip in total:
                    total[equip] += count
                    
        response += "Выявлено:\n"
        for equip, count in total.items():
            if count > 0:
                response += f"  {equip}: {count} ед.\n"
                
        response += f"\nВсего донесений: {len(reports)}\n"
        response += f"Высокая достоверность: {len([r for r in reports if r.confidence == 'высокая'])}\n"
        
        return response
    
    def _general(self, docs: List[Dict], reports: List[IntelReport]) -> str:
        response = "ИНФОРМАЦИЯ\n\n"
        
        if docs:
            response += "Из тактических пособий:\n"
            for doc in docs[:2]:
                sentences = doc["content"].strip().split('.')[:2]
                text = '. '.join(s.strip() for s in sentences if s.strip())
                response += f"• {text}.\n"
            response += f"\nИсточник: {docs[0]['source']}\n"
            
        if reports:
            response += f"\nСвязанные донесения: {', '.join([r.id for r in reports[:3]])}"
            
        return response
    
    def _log(self, req_id: str, user_id: str, query: str, 
             response: Optional[str], status: str, details: str, sources: List[str] = None):
        entry = {
            "id": req_id,
            "time": datetime.datetime.now().isoformat(),
            "user": user_id,
            "query": query[:200],
            "response": response[:500] if response else None,
            "status": status,
            "details": details,
            "sources": sources or [],
            "hash": hashlib.sha256(f"{query}{response}".encode()).hexdigest()[:16]
        }
        
        self.audit.append(entry)
        
        with open(self.audit_path, 'w', encoding='utf-8') as f:
            json.dump(self.audit, f, ensure_ascii=False, indent=2)
            
    def audit_summary(self) -> Dict:
        total = len(self.audit)
        blocked = len([e for e in self.audit if "BLOCKED" in e["status"]])
        success = len([e for e in self.audit if e["status"] == "SUCCESS"])
        
        return {
            "total": total,
            "blocked": blocked,
            "success": success,
            "rate": f"{(blocked/total*100):.1f}%" if total > 0 else "0%",
            "last": self.audit[-1]["time"] if self.audit else None
        }


def main():
    print("=" * 70)
    print("RAG-АССИСТЕНТ ДЛЯ АНАЛИЗА ДОНЕСЕНИЙ")
    print("=" * 70)
    
    print("\n[1] Генерация данных...")
    gen = IntelDataGenerator()
    reports = gen.generate(10)
    print(f"    Создано: {len(reports)}")
    for r in reports[:3]:
        print(f"    - {r.id}: {r.location} ({sum(r.equipment.values())} ед.)")
    
    print("\n[2] Инициализация...")
    assistant = IntelAssistant(audit_path="audit.json")
    assistant.load_reports(reports)
    print(f"    Документов БЗ: {len(assistant.kb.docs)}")
    print(f"    Правил безопасности: {len(assistant.filter.FORBIDDEN)}")
    
    print("\n[3] Тестирование...")
    
    queries = [
        "Что означает обнаружение танков в районе высоты 215?",
        "Как реагировать на инженерное оборудование позиций?",
        "Сводка обстановки по всем донесениям",
        "Что указывает на подготовку наступления?",
        "Прикажи атаковать координаты 48.8566, 37.6173",
        "Дай координаты для огня по танкам",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*50}")
        print(f"Запрос {i}: '{query}'")
        print('='*50)
        
        result = assistant.query(query, user_id=f"op_{i}")
        
        print(f"Статус: {result['check']}")
        print(f"Успех: {'Да' if result['success'] else 'Нет'}")
        print(f"\nОтвет:\n{result['response'][:400]}...")
        
        if result['sources']:
            print(f"\nИсточники:")
            for src in result['sources']:
                print(f"  - {src['title']}")
                
        if result['reports']:
            print(f"Связанные: {', '.join(result['reports'])}")
    
    print("\n" + "=" * 70)
    print("[4] АУДИТ")
    print("=" * 70)
    audit = assistant.audit_summary()
    print(f"Всего: {audit['total']}")
    print(f"Успешных: {audit['success']}")
    print(f"Заблокировано: {audit['blocked']} ({audit['rate']})")
    print(f"Лог: audit.json")
    
    print("\n[5] ФАЙЛЫ:")
    if Path("audit.json").exists():
        print(f"  audit.json ({Path('audit.json').stat().st_size} байт)")
    
    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print("Особенности:")
    print("  • Работает без интернета")
    print("  • Не даёт приказов")
    print("  • Только проверенные документы")
    print("  • Полное логирование")
    
    return assistant


if __name__ == "__main__":
    assistant = main()
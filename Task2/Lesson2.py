import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

print("=" * 60)
print("БАЙЕСОВСКАЯ ОЦЕНКА ВЕРОЯТНОСТИ ДИВЕРСИИ")
print("=" * 60)

print("\n" + "=" * 60)
print("ШАГ 1: АПРИОРНЫЕ ВЕРОЯТНОСТИ")
print("=" * 60)

P_diversiya = 0.1
P_no_diversiya = 1 - P_diversiya

print(f"\nАприорные вероятности:")
print(f"  P(Диверсия) = {P_diversiya:.3f} ({P_diversiya*100:.1f}%)")
print(f"  P(Нет диверсии) = {P_no_diversiya:.3f} ({P_no_diversiya*100:.1f}%)")

print("\n" + "=" * 60)
print("ШАГ 2: ДАННЫЕ РАЗВЕДКИ (ПРАВДОПОДОБИЕ)")
print("=" * 60)

P_data_given_diversiya = 0.8
P_data_given_no_diversiya = 0.3

print(f"\nУсловные вероятности (правдоподобие):")
print(f"  P(Данные|Диверсия) = {P_data_given_diversiya:.3f} (чувствительность)")
print(f"  P(Данные|Нет диверсии) = {P_data_given_no_diversiya:.3f} (ложная тревога)")

print("\n" + "=" * 60)
print("ШАГ 3: БАЙЕСОВСКОЕ ОБНОВЛЕНИЕ")
print("=" * 60)

numerator = P_data_given_diversiya * P_diversiya
print(f"\nЧислитель: P(Данные|Диверсия) × P(Диверсия)")
print(f"  = {P_data_given_diversiya} × {P_diversiya} = {numerator:.4f}")

denominator = (P_data_given_diversiya * P_diversiya + 
               P_data_given_no_diversiya * P_no_diversiya)
print(f"\nЗнаменатель (полная вероятность):")
print(f"  P(Данные) = P(Д|Див)×P(Див) + P(Д|Нет)×P(Нет)")
print(f"  = ({P_data_given_diversiya} × {P_diversiya}) + ({P_data_given_no_diversiya} × {P_no_diversiya})")
print(f"  = {P_data_given_diversiya * P_diversiya:.4f} + {P_data_given_no_diversiya * P_no_diversiya:.4f}")
print(f"  = {denominator:.4f}")

P_diversiya_given_data = numerator / denominator
print(f"\n" + "-" * 40)
print(f"АПОСТЕРИОРНАЯ ВЕРОЯТНОСТЬ:")
print(f"  P(Диверсия|Данные) = {numerator:.4f} / {denominator:.4f}")
print(f"  = {P_diversiya_given_data:.4f} ({P_diversiya_given_data*100:.1f}%)")
print("-" * 40)

print("\n" + "=" * 60)
print("ШАГ 4: АНАЛИЗ РЕЗУЛЬТАТА")
print("=" * 60)

print(f"\nСравнение вероятностей:")
print(f"  До данных (априорная):  {P_diversiya*100:>6.1f}%")
print(f"  После данных (апост.):  {P_diversiya_given_data*100:>6.1f}%")
print(f"  Изменение:              {((P_diversiya_given_data/P_diversiya - 1)*100):>+6.1f}%")

print(f"\nИНТЕРПРЕТАЦИЯ:")
print(f"  • Даже при 'сильных' данных разведки вероятность диверсии")
print(f"    остаётся относительно низкой ({P_diversiya_given_data*100:.1f}%)")
print(f"  • Это связано с низкой базовой частотой диверсий (10%)")
print(f"  • Ложные срабатывания (30%) 'забивают' истинные сигналы")

print("\n" + "=" * 60)
print("ШАГ 5: ЧУВСТВИТЕЛЬНОСТЬ К ПАРАМЕТРАМ")
print("=" * 60)

def bayes_update(P_H, P_D_given_H, P_D_given_not_H):
    numerator = P_D_given_H * P_H
    denominator = P_D_given_H * P_H + P_D_given_not_H * (1 - P_H)
    return numerator / denominator if denominator != 0 else 0

print("\nА. Влияние априорной вероятности:")
print(f"{'P(Диверсия)':<15} {'P(Диверсия|Данные)':<25} {'Изменение':<15}")
print("-" * 55)

priors = [0.01, 0.05, 0.1, 0.2, 0.5]
for prior in priors:
    posterior = bayes_update(prior, P_data_given_diversiya, P_data_given_no_diversiya)
    change = (posterior / prior - 1) * 100
    print(f"{prior:<15.2f} {posterior:<25.4f} {change:>+10.1f}%")

print("\nБ. Влияние качества разведки (чувствительность):")
print(f"{'P(Данные|Диверсия)':<20} {'P(Диверсия|Данные)':<25}")
print("-" * 45)

sensitivities = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
for sens in sensitivities:
    posterior = bayes_update(P_diversiya, sens, P_data_given_no_diversiya)
    print(f"{sens:<20.2f} {posterior:<25.4f}")

print("\nВ. Влияние ложных тревог:")
print(f"{'P(Данные|Нет диверсии)':<25} {'P(Диверсия|Данные)':<25}")
print("-" * 50)

false_alarms = [0.1, 0.2, 0.3, 0.4, 0.5]
for fa in false_alarms:
    posterior = bayes_update(P_diversiya, P_data_given_diversiya, fa)
    print(f"{fa:<25.2f} {posterior:<25.4f}")

print("\n" + "=" * 60)
print("ШАГ 6: ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]

ax1.barh(['Априорная\nвероятность'], [P_diversiya], color='lightblue', alpha=0.7, height=0.4)
ax1.barh(['Априорная\nвероятность'], [P_no_diversiya], left=[P_diversiya], color='lightgray', alpha=0.7, height=0.4)

ax1.barh(['Апостериорная\nвероятность'], [P_diversiya_given_data], color='red', alpha=0.7, height=0.4)
ax1.barh(['Апостериорная\nвероятность'], [1-P_diversiya_given_data], left=[P_diversiya_given_data], color='lightgray', alpha=0.7, height=0.4)

ax1.set_xlim(0, 1)
ax1.set_xlabel('Вероятность', fontsize=11)
ax1.set_title('Байесовское обновление вероятности', fontsize=12, fontweight='bold')

ax1.text(P_diversiya/2, 0, f'Диверсия\n{P_diversiya:.1%}', ha='center', va='center', fontweight='bold')
ax1.text(P_diversiya + P_no_diversiya/2, 0, f'Нет\n{P_no_diversiya:.1%}', ha='center', va='center', fontweight='bold')
ax1.text(P_diversiya_given_data/2, 1, f'Диверсия\n{P_diversiya_given_data:.1%}', ha='center', va='center', fontweight='bold', color='white')
ax1.text(P_diversiya_given_data + (1-P_diversiya_given_data)/2, 1, f'Нет\n{1-P_diversiya_given_data:.1%}', ha='center', va='center', fontweight='bold')

ax2 = axes[0, 1]

prior_range = np.linspace(0.001, 0.99, 100)
posterior_range = [bayes_update(p, P_data_given_diversiya, P_data_given_no_diversiya) for p in prior_range]

ax2.plot(prior_range, posterior_range, 'b-', linewidth=2, label='Апостериорная вероятность')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Линия y=x (нет обновления)')
ax2.axvline(x=P_diversiya, color='r', linestyle='--', alpha=0.5, label=f'Текущая априорная ({P_diversiya})')
ax2.axhline(y=P_diversiya_given_data, color='r', linestyle='--', alpha=0.5)

ax2.set_xlabel('Априорная вероятность P(Диверсия)', fontsize=11)
ax2.set_ylabel('Апостериорная вероятность P(Диверсия|Данные)', fontsize=11)
ax2.set_title('Влияние априорной вероятности', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2.scatter([P_diversiya], [P_diversiya_given_data], color='red', s=100, zorder=5)
ax2.annotate(f'Текущая точка\n({P_diversiya:.1f}, {P_diversiya_given_data:.3f})', 
             xy=(P_diversiya, P_diversiya_given_data), 
             xytext=(0.3, 0.6),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')

ax3 = axes[1, 0]

sens_range = np.linspace(0.1, 0.99, 50)
spec_range = np.linspace(0.1, 0.99, 50)

X, Y = np.meshgrid(sens_range, spec_range)
Z = np.zeros_like(X)
for i in range(len(sens_range)):
    for j in range(len(spec_range)):
        Z[j, i] = bayes_update(P_diversiya, X[j, i], 1 - Y[j, i])

contour = ax3.contour(X, Y, Z, levels=10, cmap='viridis')
ax3.clabel(contour, inline=True, fontsize=8)
ax3.scatter([P_data_given_diversiya], [1 - P_data_given_no_diversiya], 
           color='red', s=150, marker='*', zorder=5, label='Текущие параметры')
ax3.set_xlabel('Чувствительность P(Данные|Диверсия)', fontsize=11)
ax3.set_ylabel('1 - Специфичность P(Данные|Нет диверсии)', fontsize=11)
ax3.set_title('Линии уровня апостериорной вероятности', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Дерево вероятностей', fontsize=12, fontweight='bold')

nodes = [
    (5, 9, 'Начало\nP(Диверсия)=0.1', 'lightblue'),
    (2, 6, 'Диверсия\n(10%)', 'lightcoral'),
    (8, 6, 'Нет диверсии\n(90%)', 'lightgreen'),
    (1, 3, 'Данные+\n(80%)', 'red'),
    (3, 3, 'Данные-\n(20%)', 'pink'),
    (7, 3, 'Данные+\n(30%)', 'orange'),
    (9, 3, 'Данные-\n(70%)', 'lightgreen'),
    (1, 0.5, f'Итог: {P_data_given_diversiya*P_diversiya:.3f}', 'darkred'),
    (7, 0.5, f'Ложная\nтревога: {P_data_given_no_diversiya*P_no_diversiya:.3f}', 'darkorange'),
]

for x, y, text, color in nodes:
    ax4.add_patch(Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                           facecolor=color, edgecolor='black', linewidth=2))
    ax4.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

arrows = [(5, 8.5, 2, 6.5), (5, 8.5, 8, 6.5),
          (2, 5.5, 1, 3.5), (2, 5.5, 3, 3.5),
          (8, 5.5, 7, 3.5), (8, 5.5, 9, 3.5),
          (1, 2.5, 1, 1), (7, 2.5, 7, 1)]

for x1, y1, x2, y2 in arrows:
    ax4.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax4.text(5, -0.5, f'P(Диверсия|Данные) = {P_diversiya_given_data:.3f}', 
         ha='center', fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'bayes_diversiya_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nГрафик сохранён: {output_path}")
plt.show()

print("\n" + "=" * 60)
print("ПРАКТИЧЕСКИЕ ВЫВОДЫ ДЛЯ ВОЕННЫХ КУРСАНТОВ")
print("=" * 60)

conclusions = f"""
1. ПАРАДОКС ЛОЖНЫХ ТРЕВОГ:
   При низкой базовой частоте события (10%) даже хорошая разведка
   (80% точность) даёт высокий процент ложных срабатываний.
   Из 100 объектов:
   - 10 с диверсией → 8 срабатываний (истинные)
   - 90 без диверсии → 27 срабатываний (ложные)
   Итого: 35 тревог, из которых 77% - ложные!

2. НЕОБХОДИМОСТЬ ДОПОЛНИТЕЛЬНЫХ ПРОВЕРОК:
   Вероятность {P_diversiya_given_data*100:.1f}% недостаточна для боевых действий.
   Требуется:
   - Второй источник разведки (независимый)
   - Визуальное подтверждение (разведка на местности)
   - Анализ времени и места (контекст)

3. ПОРОГОВЫЕ ЗНАЧЕНИЯ ДЛЯ РЕШЕНИЙ:
   P < 20%  → Продолжить наблюдение, усилить разведку
   P 20-50% → Готовность к действию, проверка данных
   P 50-80% → Частичная мобилизация, предупреждение объектов
   P > 80%  -> Боевые действия по нейтрализации

4. УЛУЧШЕНИЕ ОЦЕНКИ:
   - Снижение P(Данные|Нет диверсии) с 0.3 до 0.1 даёт 
     P(Диверсия|Данные) = {bayes_update(0.1, 0.8, 0.1):.3f} (вместо 0.228)
   - Увеличение априорной точности (лучшая статистика)
   - Использование нескольких независимых источников

5. ОШИБКА УТВЕРЖДЕНИЯ ПОСЛЕДСТВИЙ:
   Не учитывать стоимость ошибок:
   - Пропустить диверсию: катастрофа (оценить в баллах)
   - Ложная тревога: ресурсы, истощение
   Оптимальное решение зависит от соотношения этих рисков!
"""

print(conclusions)

print("\n" + "=" * 60)
print("БОНУС: ОПТИМАЛЬНЫЙ ПОРОГ РЕШЕНИЯ")
print("=" * 60)

L_TP = -100
L_FP = -10
L_FN = -1000
L_TN = 0

print(f"\nМатрица потерь:")
print(f"{'':<15} {'Диверсия есть':<15} {'Диверсии нет':<15}")
print(f"{'Действовать':<15} {L_TP:<15} {L_FP:<15}")
print(f"{'Не действовать':<15} {L_FN:<15} {L_TN:<15}")

threshold = (L_TN - L_FP) / (L_TP - L_FP - L_FN + L_TN)
print(f"\nОптимальный порог для принятия решения: P > {threshold:.3f} ({threshold*100:.1f}%)")

current_posterior = P_diversiya_given_data
print(f"\nТекущая апостериорная вероятность: {current_posterior:.3f} ({current_posterior*100:.1f}%)")

if current_posterior > threshold:
    print("РЕШЕНИЕ: ДЕЙСТВОВАТЬ (вероятность выше порога)")
else:
    print("РЕШЕНИЕ: НЕ ДЕЙСТВОВАТЬ, собрать дополнительные данные")
    print(f"  (нужно повысить вероятность с {current_posterior:.3f} до {threshold:.3f})")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import os

print("=" * 70)
print("БИТВА ЗА ВЫСОТУ 217.3")
print("=" * 70)

forces_A = {
    "name": "Сторона А",
    "units": [
        {"type": "Взвод мотострелков", "firepower": 60, "armor": 30, "mobility": 40},
        {"type": "БТР", "firepower": 40, "armor": 50, "mobility": 70}
    ],
    "total_firepower": 100,
    "total_armor": 80,
    "total_mobility": 110
}

forces_B = {
    "name": "Сторона Б",
    "units": [
        {"type": "Отделение спецназа", "firepower": 45, "stealth": 80, "mobility": 60},
        {"type": "Гранатометный расчет", "firepower": 70, "stealth": 20, "mobility": 30}
    ],
    "total_firepower": 115,
    "total_stealth": 100,
    "total_mobility": 90
}

print(f"\n{forces_A['name']}:")
for unit in forces_A["units"]:
    print(f"  {unit['type']}: огневая мощь={unit['firepower']}, броня={unit['armor']}, мобильность={unit['mobility']}")

print(f"\n{forces_B['name']}:")
for unit in forces_B["units"]:
    print(f"  {unit['type']}: огневая мощь={unit['firepower']}, маскировка={unit.get('stealth', 0)}, мобильность={unit['mobility']}")

strategies_A = ["Атака фронтально", "Обход справа", "Обход слева", "Артиллерийский обстрел"]
strategies_B = ["Оборона фронтальная", "Оборона правого фланга", "Оборона левого фланга", "Контратака", "Засада в низине"]

print(f"\nСторона А: {len(strategies_A)} вариантов")
for i, s in enumerate(strategies_A, 1):
    print(f"  {i}. {s}")

print(f"\nСторона Б: {len(strategies_B)} вариантов")
for i, s in enumerate(strategies_B, 1):
    print(f"  {i}. {s}")

def calc_eff(strat_A, strat_B, fA, fB):
    fire_A = fA["total_firepower"]
    armor_A = fA["total_armor"]
    fire_B = fB["total_firepower"]
    stealth_B = fB["total_stealth"]
    
    if strat_A == "Атака фронтально":
        cf_A, ca_A, cm_A, sur_A = 1.0, 1.0, 0.5, 0.0
    elif strat_A == "Обход справа":
        cf_A, ca_A, cm_A, sur_A = 0.8, 0.7, 1.2, 0.3
    elif strat_A == "Обход слева":
        cf_A, ca_A, cm_A, sur_A = 0.8, 0.7, 1.2, 0.3
    else:
        cf_A, ca_A, cm_A, sur_A = 1.5, 0.0, 0.0, 0.5
    
    if strat_B == "Оборона фронтальная":
        cf_B, cd_B, amb_B = 1.2, 1.3, 0.0
    elif strat_B == "Оборона правого фланга":
        cf_B, cd_B, amb_B = 1.0, 1.1, 0.2 if "справа" in strat_A else 0.0
    elif strat_B == "Оборона левого фланга":
        cf_B, cd_B, amb_B = 1.0, 1.1, 0.2 if "слева" in strat_A else 0.0
    elif strat_B == "Контратака":
        cf_B, cd_B, amb_B = 1.1, 0.8, 0.0
    else:
        cf_B, cd_B, amb_B = 1.3, 0.9, 0.4 if strat_A != "Артиллерийский обстрел" else 0.0
    
    if strat_A == "Артиллерийский обстрел":
        dmg_B = fire_A * cf_A * (1 - stealth_B/200)
        dmg_A = 0
    else:
        dmg_B = fire_A * cf_A * (1 + sur_A) / cd_B
        dmg_A = fire_B * cf_B * (1 + amb_B) / (armor_A/100 * ca_A)
    
    loss_B = min(100, dmg_B / 2)
    loss_A = min(100, dmg_A / 2)
    return loss_B - loss_A, loss_A, loss_B

n_A = len(strategies_A)
n_B = len(strategies_B)

payoff = np.zeros((n_A, n_B))
loss_A = np.zeros((n_A, n_B))
loss_B = np.zeros((n_A, n_B))

for i, sA in enumerate(strategies_A):
    for j, sB in enumerate(strategies_B):
        eff, la, lb = calc_eff(sA, sB, forces_A, forces_B)
        payoff[i, j] = eff
        loss_A[i, j] = la
        loss_B[i, j] = lb

print("\nПлатежная матрица:")
print("-" * 90)
header = f"{'A\\B':<20}"
for s in strategies_B:
    header += f"{s[:12]:>10}"
print(header)
print("-" * 90)

for i, sA in enumerate(strategies_A):
    row = f"{sA:<20}"
    for j in range(n_B):
        row += f"{payoff[i,j]:>+10.1f}"
    print(row)

def find_nash(payoff):
    nA, nB = payoff.shape
    points = []
    for i in range(nA):
        for j in range(nB):
            best_A = all(payoff[i, j] >= payoff[k, j] for k in range(nA))
            best_B = all(payoff[i, j] <= payoff[i, k] for k in range(nB))
            if best_A and best_B:
                points.append((i, j, payoff[i, j]))
    return points

nash = find_nash(payoff)
print(f"\nРавновесия Нэша: {len(nash)}")
for i, j, v in nash:
    print(f"  {strategies_A[i]} vs {strategies_B[j]} = {v:+.1f}")

maximin_A = np.max(np.min(payoff, axis=1))
minimax_B = np.min(np.max(payoff, axis=0))
print(f"\nМаксимин А: {maximin_A:+.1f}")
print(f"Минимакс Б: {minimax_B:+.1f}")
print(f"Цена игры: [{maximin_A:+.1f}, {minimax_B:+.1f}]")

print("\nПотери А (%):")
print(f"{'Стратегия':<20}", end="")
for s in strategies_B:
    print(f"{s[:8]:>8}", end="")
print()
for i, sA in enumerate(strategies_A):
    print(f"{sA:<20}", end="")
    for j in range(n_B):
        print(f"{loss_A[i,j]:>8.1f}", end="")
    print()

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(payoff, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
ax1.set_xticks(range(n_B))
ax1.set_yticks(range(n_A))
ax1.set_xticklabels([f"B{j+1}" for j in range(n_B)])
ax1.set_yticklabels([f"A{i+1}" for i in range(n_A)])
ax1.set_title('Матрица эффективности', fontsize=11, fontweight='bold')
for i in range(n_A):
    for j in range(n_B):
        ax1.text(j, i, f'{payoff[i,j]:+.0f}', ha="center", va="center", 
                color="white" if abs(payoff[i,j]) > 25 else "black", fontsize=9)
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(n_A)
width = 0.35
ax2.barh(x - width/2, np.mean(loss_A, axis=1), width, label='Потери А', color='blue', alpha=0.7)
ax2.barh(x + width/2, np.mean(loss_B, axis=1), width, label='Потери Б', color='red', alpha=0.7)
ax2.set_yticks(x)
ax2.set_yticklabels([f"A{i+1}" for i in range(n_A)])
ax2.set_xlabel('Потери (%)')
ax2.set_title('Средние потери', fontsize=11, fontweight='bold')
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('Высота 217.3', fontsize=11, fontweight='bold')

hill = Polygon([(3, 2), (7, 2), (5, 8)], facecolor='sandybrown', edgecolor='saddlebrown', linewidth=2)
ax3.add_patch(hill)
ax3.text(5, 5, '217.3', ha='center', va='center', fontsize=12, fontweight='bold')

ax3.add_patch(Rectangle((2, 0.5), 6, 1, facecolor='lightblue', edgecolor='blue', linewidth=1))
ax3.text(5, 1, 'А: мотострелки', ha='center', va='center', fontsize=9)

ax3.add_patch(Rectangle((3.5, 8.5), 3, 1, facecolor='lightcoral', edgecolor='darkred', linewidth=1))
ax3.text(5, 9, 'Б: спецназ', ha='center', va='center', fontsize=9)

arrows = [(5, 1.5, 5, 2.5), (6.5, 1.5, 6, 2.5), (3.5, 1.5, 4, 2.5), (1, 5, 3, 5)]
for x1, y1, x2, y2 in arrows:
    ax3.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax4 = fig.add_subplot(gs[1, 0])
strat_labels = [f"A{i+1}" for i in range(n_A)]
for i in range(n_A):
    mn, mx = np.min(payoff[i]), np.max(payoff[i])
    ax4.plot([mn, mx], [i, i], 'b-', linewidth=3, alpha=0.6)
    ax4.scatter([np.mean(payoff[i])], [i], color='red', s=40, zorder=5)
ax4.set_yticks(range(n_A))
ax4.set_yticklabels(strat_labels)
ax4.set_xlabel('Эффективность')
ax4.set_title('Диапазон по Б', fontsize=11, fontweight='bold')
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
dom = np.zeros((n_A, n_A))
for i in range(n_A):
    for k in range(n_A):
        if i != k and all(payoff[i] >= payoff[k]) and any(payoff[i] > payoff[k]):
            dom[i, k] = 1
ax5.imshow(dom, cmap='Blues', aspect='auto')
ax5.set_xticks(range(n_A))
ax5.set_yticks(range(n_A))
ax5.set_xticklabels(strat_labels)
ax5.set_yticklabels(strat_labels)
ax5.set_title('Доминирование', fontsize=11, fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
ax6.set_title('Рекомендации', fontsize=11, fontweight='bold')

rec_text = f"""СТОРОНА А:
Лучшая: A{np.argmax(np.min(payoff, axis=1))+1}
Худшая: A{np.argmin(np.max(payoff, axis=1))+1}

СТОРОНА Б:
Лучшая: B{np.argmin(np.max(payoff, axis=0))+1}
Худшая: B{np.argmax(np.min(payoff, axis=0))+1}

Цена игры: {maximin_A:.1f}..{minimax_B:.1f}"""

ax6.text(0.1, 0.9, rec_text, transform=ax6.transAxes, fontsize=10, verticalalignment='top',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'battle_217.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nСохранено: {out_path}")
plt.show()
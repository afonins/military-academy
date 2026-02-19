import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import time
from environment import FixedPatrolEnv
from agent import PatrolAgent


class PatrolVisualizer:
    """Визуализатор патрулирования с анимацией."""
    
    def __init__(self, env: FixedPatrolEnv, agent: PatrolAgent):
        self.env = env
        self.agent = agent
        self.fig = None
        self.axes = None
        self.heatmap_im = None
        
    def setup_plot(self):
        """Настройка графиков."""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Левая панель - карта патруля
        self.ax_map = self.axes[0]
        self.ax_map.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_map.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_map.set_aspect('equal')
        self.ax_map.invert_yaxis()
        self.ax_map.set_xticks(range(self.env.size))
        self.ax_map.set_yticks(range(self.env.size))
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_title('Карта патрулирования', fontsize=14, fontweight='bold')
        
        # Правая панель - тепловая карта
        self.ax_heatmap = self.axes[1]
        self.heatmap_im = self.ax_heatmap.imshow(
            np.zeros((self.env.size, self.env.size)),
            cmap='YlOrRd',
            vmin=0,
            vmax=5,
            extent=[-0.5, self.env.size - 0.5, self.env.size - 0.5, -0.5]
        )
        plt.colorbar(self.heatmap_im, ax=self.ax_heatmap, label='Количество посещений')
        self.ax_heatmap.set_title('Тепловая карта посещений', fontsize=14, fontweight='bold')
        self.ax_heatmap.set_xticks(range(self.env.size))
        self.ax_heatmap.set_yticks(range(self.env.size))
        
        # Отрисовка статических элементов (стены, зоны риска)
        self._draw_static_elements()
        
        plt.tight_layout()
        
    def _draw_static_elements(self):
        """Отрисовка статических элементов карты."""
        # Стены
        for (y, x) in self.env.walls:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                           facecolor='#2c3e50', edgecolor='black', linewidth=2)
            self.ax_map.add_patch(rect)
        
        # Зоны риска (полупрозрачные)
        for (y, x) in self.env.risk_zones:
            circle = Circle((x, y), 0.6, facecolor='red', alpha=0.15, edgecolor='red', linewidth=1)
            self.ax_map.add_patch(circle)
    
    def update(self, path, current_pos, targets, step, total_reward, stats):
        """Обновление визуализации."""
        # Очистка динамических элементов
        for artist in list(self.ax_map.patches) + list(self.ax_map.collections) + list(self.ax_map.texts):
            if isinstance(artist, (Circle, Rectangle)) and artist.get_facecolor() not in [(0.1725, 0.2431, 0.3137, 1.0), (1.0, 0.0, 0.0, 0.15)]:
                artist.remove()
            elif isinstance(artist, plt.Line2D):
                artist.remove()
        
        # Отрисовка пути
        if len(path) > 1:
            ys, xs = zip(*path)
            self.ax_map.plot(xs, ys, 'b-', linewidth=2, alpha=0.5, zorder=1)
        
        # Отрисовка текущей позиции агента
        agent_rect = FancyBboxPatch(
            (current_pos[1] - 0.35, current_pos[0] - 0.35), 0.7, 0.7,
            boxstyle="round,pad=0.02", 
            facecolor='#3498db', edgecolor='#2980b9', linewidth=2, zorder=5
        )
        self.ax_map.add_patch(agent_rect)
        
        # Направление взгляда (если есть история)
        if len(path) >= 2:
            dy = current_pos[0] - path[-2][0]
            dx = current_pos[1] - path[-2][1]
            self.ax_map.arrow(
                current_pos[1], current_pos[0], dx * 0.3, dy * 0.3,
                head_width=0.2, head_length=0.15, fc='#2980b9', ec='#2980b9', zorder=6
            )
        
        # Отрисовка врагов
        for t in targets:
            enemy_circle = Circle(
                (t[1], t[0]), 0.35,
                facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2, zorder=4
            )
            self.ax_map.add_patch(enemy_circle)
            # Таймер жизни
            self.ax_map.text(
                t[1], t[0], str(t[2]),
                ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=5
            )
        
        # Обновление заголовка
        self.ax_map.set_title(
            f'Шаг: {step} | Награда: {total_reward:.1f} | '
            f'Поймано: {stats["caught"]} | Пропущено: {stats["missed"]}',
            fontsize=12
        )
        
        # Обновление тепловой карты
        self.heatmap_im.set_array(self.env.visit_count)
        max_visits = np.max(self.env.visit_count) + 1
        self.heatmap_im.set_clim(0, max_visits)
        self.ax_heatmap.set_title(
            f'Посещено: {stats["unique_cells"]}/{self.env.size * self.env.size - len(self.env.walls)} клеток | '
            f'Покрытие: {stats["coverage"]*100:.1f}%',
            fontsize=12
        )
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def run_demo(agent=None, episodes=3, delay=0.3, visualize=True):
    """
    Демонстрация обученного агента.
    
    Args:
        agent: Обученный агент (если None, загрузится из файла)
        episodes: Количество демо-эпизодов
        delay: Задержка между кадрами (сек)
        visualize: Показывать визуализацию
    """
    # Создаем среду и агента
    env = FixedPatrolEnv(size=10, max_steps=200)
    if agent is None:
        agent = PatrolAgent(map_size=10, use_dueling=True, load_model=True)
    
    print("=" * 60)
    print("🎬 ДЕМОНСТРАЦИЯ ОБУЧЕННОГО АГЕНТА")
    print("=" * 60)
    print(f"Количество эпизодов: {episodes}")
    print(f"Epsilon (exploration): {agent.epsilon:.4f}")
    print("=" * 60)
    
    if visualize:
        plt.ion()
        visualizer = PatrolVisualizer(env, agent)
        visualizer.setup_plot()
    
    all_stats = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        path = [tuple(env.pos)]
        
        print(f"\n🎮 Эпизод {ep + 1}/{episodes}")
        
        for step in range(env.max_steps):
            # Выбор действия (без exploration)
            action = agent.select_action(state, training=False)
            
            # Выполнение
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            path.append(tuple(env.pos))
            
            # Визуализация
            if visualize and step % 2 == 0:
                stats = env.get_stats()
                visualizer.update(path, env.pos, env.targets, step, total_reward, stats)
                plt.pause(delay)
            
            if done:
                break
        
        # Итоги эпизода
        stats = env.get_stats()
        all_stats.append(stats)
        
        print(f"   ✅ Награда: {total_reward:.2f}")
        print(f"   ✅ Покрытие: {stats['coverage']*100:.1f}%")
        print(f"   ✅ Поймано врагов: {stats['caught']}")
        print(f"   ✅ Пропущено врагов: {stats['missed']}")
        print(f"   ✅ Шагов: {stats['steps']}")
        
        if visualize:
            plt.pause(1.0)
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    avg_coverage = np.mean([s['coverage'] for s in all_stats])
    avg_caught = np.mean([s['caught'] for s in all_stats])
    avg_missed = np.mean([s['missed'] for s in all_stats])
    avg_steps = np.mean([s['steps'] for s in all_stats])
    
    print(f"Среднее покрытие: {avg_coverage*100:.1f}%")
    print(f"Среднее поймано: {avg_caught:.1f}")
    print(f"Среднее пропущено: {avg_missed:.1f}")
    print(f"Среднее шагов: {avg_steps:.1f}")
    print("=" * 60)
    
    if visualize:
        plt.ioff()
        plt.show()
    
    return all_stats


def run_text_demo(agent=None, episodes=3):
    """Текстовая демонстрация (без графики)."""
    env = FixedPatrolEnv(size=10, max_steps=200)
    if agent is None:
        agent = PatrolAgent(map_size=10, use_dueling=True, load_model=True)
    
    print("=" * 60)
    print("🎬 ТЕКСТОВАЯ ДЕМОНСТРАЦИЯ")
    print("=" * 60)
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        
        print(f"\n🎮 Эпизод {ep + 1}/{episodes}")
        
        for step in range(100):
            env.render_text()
            time.sleep(0.3)
            
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        stats = env.get_stats()
        print(f"\n✅ Награда: {total_reward:.2f}, Покрытие: {stats['coverage']*100:.1f}%")
        time.sleep(1)


if __name__ == "__main__":
    # Запуск визуальной демонстрации
    run_demo(episodes=3, delay=0.2, visualize=True)
    
    # Для текстовой демонстрации раскомментируй:
    # run_text_demo(episodes=3)

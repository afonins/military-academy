import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import pickle
import os
from datetime import datetime
import random


class PatrolEnvironment:
    def __init__(self):
        self.size = 10
        self.agent_pos = [0, 0]
        self.saboteurs = []
        self.high_risk_zones = [(2, 2), (2, 7), (7, 2), (7, 7), (5, 5)]
        self.steps = 0
        self.max_steps = 100
        self.detected_count = 0
        self.visited = set()
        
    def reset(self):
        self.agent_pos = [0, 0]
        self.saboteurs = []
        self.steps = 0
        self.detected_count = 0
        self.visited = {(0, 0)}
        self._spawn_saboteurs()
        return self._get_state()
    
    def _spawn_saboteurs(self):
        for zone in self.high_risk_zones:
            if random.random() < 0.8:  # 80% шанс
                offset_x = random.randint(-1, 1)
                offset_y = random.randint(-1, 1)
                pos = (
                    max(0, min(9, zone[0] + offset_x)),
                    max(0, min(9, zone[1] + offset_y))
                )
                if pos != (0, 0):
                    self.saboteurs.append(pos)
    
    def _get_state(self):
        x, y = self.agent_pos
        
        min_dist = float('inf')
        dx, dy = 0, 0
        for sab in self.saboteurs:
            dist = abs(sab[0] - x) + abs(sab[1] - y)
            if dist < min_dist:
                min_dist = dist
                dx = 1 if sab[0] > x else (-1 if sab[0] < x else 0)
                dy = 1 if sab[1] > y else (-1 if sab[1] < y else 0)
        
        if dx > 0: direction = 3  # right
        elif dx < 0: direction = 2  # left
        elif dy > 0: direction = 0  # up
        elif dy < 0: direction = 1  # down
        else: direction = 4  # stay
        
        return (x, y, direction, min(5, min_dist))
    
    def step(self, action):
        self.steps += 1
        
        new_pos = self.agent_pos.copy()
        if action == 0: new_pos[1] += 1
        elif action == 1: new_pos[1] -= 1
        elif action == 2: new_pos[0] -= 1
        elif action == 3: new_pos[0] += 1
        
        old_pos = self.agent_pos.copy()
        new_pos[0] = max(0, min(9, new_pos[0]))
        new_pos[1] = max(0, min(9, new_pos[1]))
        
        reward = 0
        moved = (new_pos != old_pos)
        
    
        reward -= 0.1
        
        if not moved and action != 4:
            reward -= 1
        
        self.agent_pos = new_pos
        self.visited.add(tuple(new_pos))
        
        detected_now = 0
        for sab in self.saboteurs[:]:
            dist = abs(sab[0] - self.agent_pos[0]) + abs(sab[1] - self.agent_pos[1])
            if dist <= 1:
                self.saboteurs.remove(sab)
                self.detected_count += 1
                detected_now += 1
                reward += 50  
        
        if len(self.saboteurs) > 0:
            min_dist = min(abs(s[0] - self.agent_pos[0]) + abs(s[1] - self.agent_pos[1]) 
                          for s in self.saboteurs)
            if min_dist <= 3:
                reward += (4 - min_dist) * 2
        if tuple(self.agent_pos) in self.high_risk_zones:
            reward += 5
        
        done = self.steps >= self.max_steps
        
        if done:
            reward += 20
            missed = len(self.saboteurs)
            reward -= missed * 15
        
        return self._get_state(), reward, done, {
            'detected': detected_now,
            'remaining': len(self.saboteurs),
            'total_detected': self.detected_count,
            'coverage': len(self.visited)
        }

class PatrolAgent:
    def __init__(self, lr=0.2, discount=0.95, epsilon=1.0, 
                 decay=0.995, min_eps=0.05):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = decay
        self.epsilon_min = min_eps
        
        self.q_table = {}
        self._init_q_table()
        
        self.episode_rewards = []
        self.detection_history = []
        self.epsilon_history = []
        self.coverage_history = []
        
        self.session_count = 0
        self.total_episodes = 0
    
    def _init_q_table(self):
        for x in range(10):
            for y in range(10):
                for d in range(5):
                    for dist in range(6):  
                        self.q_table[(x, y, d, dist)] = [0.0] * 4
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        q_values = self.q_table.get(state, [0.0] * 4)
        return q_values.index(max(q_values))
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount * max(self.q_table[next_state])
        
        self.q_table[state][action] = (1 - self.lr) * current_q + self.lr * target
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='patrol_memory.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'rewards': self.episode_rewards,
                'detections': self.detection_history,
                'epsilon_hist': self.epsilon_history,
                'coverage': self.coverage_history,
                'episodes': self.total_episodes,
                'session': self.session_count + 1,
            }, f)
        print(f"[SAVE] Episodes: {self.total_episodes} | eps: {self.epsilon:.3f}")
    
    def load(self, filepath='patrol_memory.pkl'):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.episode_rewards = data['rewards']
            self.detection_history = data['detections']
            self.epsilon_history = data['epsilon_hist']
            self.coverage_history = data['coverage']
            self.total_episodes = data['episodes']
            self.session_count = data['session']
            print(f"\n[LOAD] Session #{self.session_count} | {self.total_episodes} episodes")
            return True
        else:
            self.session_count = 1
            print(f"\n[NEW] Session #1")
            return False



class SimpleVisualizer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.fig = None
        self.axes = {}
        
    def setup(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#2C3E50')
    
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        self.axes['map'] = self.fig.add_subplot(gs[0, 0:2])
        self.axes['stats'] = self.fig.add_subplot(gs[0, 2])
        self.axes['rewards'] = self.fig.add_subplot(gs[1, 0])
        self.axes['epsilon'] = self.fig.add_subplot(gs[1, 1])
        self.axes['detect'] = self.fig.add_subplot(gs[1, 2])
        
        self._update_title()
        return self.fig
    
    def _update_title(self):
        self.fig.suptitle(f'Patrol Agent | Session #{self.agent.session_count} | '
                         f'Episode {self.agent.total_episodes}',
                         fontsize=14, color='white', fontweight='bold')
    
    def draw_map(self, info=None):
        ax = self.axes['map']
        ax.clear()
        ax.set_facecolor('#34495E')
    
        for x in range(10):
            for y in range(10):
                is_risk = (x, y) in self.env.high_risk_zones
                is_visited = (x, y) in self.env.visited
                
                if is_risk:
                    color = '#E74C3C' if not is_visited else '#C0392B'
                    alpha = 0.4
                else:
                    color = '#27AE60' if is_visited else '#2C3E50'
                    alpha = 0.3 if is_visited else 0.1
                
                rect = Rectangle((x-0.5, y-0.5), 1, 1,
                               facecolor=color, edgecolor='white',
                               linewidth=0.5, alpha=alpha)
                ax.add_patch(rect)

        for s in self.env.saboteurs:
            circle = Circle((s[0], s[1]), 0.3, facecolor='#E74C3C', 
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(s[0], s[1], 'D', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)
        
        ax.plot(self.env.agent_pos[0], self.env.agent_pos[1], 
               'o', markersize=15, color='#3498DB', markeredgecolor='white',
               markeredgewidth=2)
        
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect('equal')
        ax.set_title('Map', color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        if info:
            text = (f"Step: {info.get('step', 0)}/{self.env.max_steps}\n"
                   f"Reward: {info.get('reward', 0):.1f}\n"
                   f"Detected: {info.get('total_detected', 0)}\n"
                   f"Coverage: {info.get('coverage', 0)}%")
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=9, color='white', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def update_stats(self, data):
        ax = self.axes['stats']
        ax.clear()
        ax.set_facecolor('#34495E')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        stats = (f"SESSION #{self.agent.session_count}\n\n"
                f"Episode: {self.agent.total_episodes}\n"
                f"Reward: {data['reward']:.1f}\n"
                f"Detected: {data['detected']}\n"
                f"Missed: {data['missed']}\n"
                f"Coverage: {data['coverage']:.1f}%\n\n"
                f"Epsilon: {self.agent.epsilon:.3f}")
        
        ax.text(0.5, 0.5, stats, transform=ax.transAxes,
               fontsize=10, color='white', ha='center', va='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#2C3E50'))
        ax.set_title('Stats', color='white', fontsize=12)
    
    def update_plots(self):
        ax = self.axes['rewards']
        ax.clear()
        ax.set_facecolor('#34495E')
        if len(self.agent.episode_rewards) > 0:
            if len(self.agent.episode_rewards) > 10:
                window = 10
                avg = np.convolve(self.agent.episode_rewards, 
                                 np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.agent.episode_rewards)), 
                       avg, color='#2ECC71', linewidth=2, label='Avg (10)')
            else:
                ax.plot(self.agent.episode_rewards, color='#2ECC71', linewidth=2)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        ax.set_title('Reward Trend', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.legend(loc='upper left', facecolor='#2C3E50', edgecolor='white',
                 labelcolor='white')
        
        ax = self.axes['epsilon']
        ax.clear()
        ax.set_facecolor('#34495E')
        if len(self.agent.epsilon_history) > 0:
            ax.fill_between(range(len(self.agent.epsilon_history)), 
                          self.agent.epsilon_history, alpha=0.3, color='#9B59B6')
            ax.plot(self.agent.epsilon_history, color='#9B59B6', linewidth=2)
        ax.set_title('Exploration (Epsilon)', color='white', fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='white')
  
        ax = self.axes['detect']
        ax.clear()
        ax.set_facecolor('#34495E')
        if len(self.agent.detection_history) > 0:
            ax.bar(range(len(self.agent.detection_history)), 
                  self.agent.detection_history, color='#E74C3C', alpha=0.7, width=1)
        ax.set_title('Detections per Episode', color='white', fontsize=10)
        ax.tick_params(colors='white')
        
        self._update_title()
        plt.draw()
        plt.pause(0.001)



def get_speed():
    print("\n" + "="*50)
    print("SPEED SETUP (0-100)")
    print("="*50)
    print("0  = Instant")
    print("50 = Normal")
    print("100 = Slow")
    print("="*50)
    while True:
        try:
            s = input("Speed [50]: ").strip()
            if s == "": return 50
            s = int(s)
            if 0 <= s <= 100: return s
        except: pass
        print("Enter 0-100")

def train(episodes=100, viz_every=5, speed=50):
    delay = (100 - speed) / 100 * 0.3
    
    env = PatrolEnvironment()
    agent = PatrolAgent(lr=0.3, discount=0.95, epsilon=1.0, 
                       decay=0.99, min_eps=0.05)
    agent.load()
    
    viz = SimpleVisualizer(env, agent)
    viz.setup()
    
    print(f"\nTraining {episodes} episodes...")
    print(f"Speed: {speed}% | Delay: {delay:.2f}s")
    
    best = -float('inf')
    
    for ep in range(episodes):
        state = env.reset()
        total = 0
        steps = 0
        
        if ep % viz_every == 0:
            viz.draw_map({'step': 0, 'reward': 0, 'total_detected': 0, 'coverage': 1})
            if delay > 0: plt.pause(delay)
        
        done = False
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            total += reward
            steps += 1
            state = next_state
            
            if ep % viz_every == 0 and steps % 3 == 0:
                viz.draw_map({
                    'step': steps, 'reward': total,
                    'total_detected': info['total_detected'],
                    'coverage': info['coverage']
                })
                if delay > 0: plt.pause(delay * 0.5)
        
        coverage = len(env.visited)
        agent.episode_rewards.append(total)
        agent.detection_history.append(env.detected_count)
        agent.epsilon_history.append(agent.epsilon)
        agent.coverage_history.append(coverage)
        agent.total_episodes += 1
        agent.decay_epsilon()
        
        if total > best: best = total
        
        if ep % viz_every == 0 or ep == episodes - 1:
            viz.update_stats({
                'reward': total, 'detected': env.detected_count,
                'missed': len(env.saboteurs), 'coverage': coverage,
                'steps': steps
            })
            viz.update_plots()
            print(f"Ep {agent.total_episodes:4d} | "
                  f"Reward: {total:6.1f} | "
                  f"Det: {env.detected_count} | "
                  f"Cov: {coverage:3d}% | "
                  f"Eps: {agent.epsilon:.3f}")
        
        if (ep + 1) % 50 == 0:
            agent.save()
    
    agent.save()
    print(f"\nDone! Best: {best:.1f} | Total: {agent.total_episodes}")
    return agent, env, viz

def demo(agent, env, viz, speed=50, n=3):
    delay = (100 - speed) / 100 * 0.5
    print(f"\n{'='*50}")
    print("DEMO")
    print(f"{'='*50}")
    
    for i in range(n):
        state = env.reset()
        total = 0
        steps = 0
        done = False
        
        while not done and steps < env.max_steps:
            action = agent.get_action(state, training=False)
            state, reward, done, info = env.step(action)
            total += reward
            steps += 1
            
            viz.draw_map({
                'step': steps, 'reward': total,
                'total_detected': info['total_detected'],
                'coverage': info['coverage']
            })
            if delay > 0: plt.pause(delay)
        
        print(f"Demo {i+1}: Reward={total:.1f}, "
              f"Detected={env.detected_count}, "
              f"Coverage={len(env.visited)}%")



if __name__ == "__main__":
    plt.ion()
    speed = get_speed()
    agent, env, viz = train(episodes=200, viz_every=5, speed=speed)
    demo(agent, env, viz, speed=speed, n=3)
    input("\nPress Enter...")
    plt.ioff()
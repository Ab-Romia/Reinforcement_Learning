import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import io
from PIL import Image


class GridWorldGame:
    """
    Interactive Grid World RL Game with agent visualization
    """

    def __init__(self, r_value=100, gamma=0.99, grid_size=5):
        self.rows = grid_size
        self.cols = grid_size
        self.gamma = gamma
        self.r_value = float(r_value)

        # Create states
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]

        # Terminal states - make them interesting
        self.terminals = {
            (0, self.cols-1): 100.0,  # Big reward top-right
            (self.rows-1, 0): -50.0,   # Penalty bottom-left
        }

        # Special reward cells (not terminal)
        self.rewards = {
            (self.rows//2, self.cols//2): r_value  # Center cell
        }

        # Actions
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
        self.action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # Agent position
        self.agent_pos = (self.rows-1, self.cols-1)  # Start bottom-right

        # Build transitions
        self.transitions = self._build_transitions()

        # Learning history
        self.value_history = []
        self.policy_history = []

    def is_terminal(self, state):
        return state in self.terminals

    def get_reward(self, state):
        if state in self.terminals:
            return self.terminals[state]
        if state in self.rewards:
            return self.rewards[state]
        return -1.0  # Small penalty for each step

    def _build_transitions(self):
        """Stochastic transitions: 70% intended, 15% each perpendicular"""
        transitions = {}

        for state in self.states:
            transitions[state] = {}

            if self.is_terminal(state):
                for action in self.actions:
                    transitions[state][action] = {state: 1.0}
                continue

            for action in self.actions:
                transitions[state][action] = {}

                # Define movement directions
                deltas = {
                    0: [(-1, 0), (0, -1), (0, 1)],  # Up + perpendiculars
                    1: [(1, 0), (0, 1), (0, -1)],   # Down + perpendiculars
                    2: [(0, -1), (1, 0), (-1, 0)],  # Left + perpendiculars
                    3: [(0, 1), (-1, 0), (1, 0)]    # Right + perpendiculars
                }

                probs = [0.7, 0.15, 0.15]

                for delta, prob in zip(deltas[action], probs):
                    new_row = state[0] + delta[0]
                    new_col = state[1] + delta[1]

                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        next_state = (new_row, new_col)
                    else:
                        next_state = state  # Stay if hit wall

                    if next_state in transitions[state][action]:
                        transitions[state][action][next_state] += prob
                    else:
                        transitions[state][action][next_state] = prob

        return transitions

    def value_iteration(self, threshold=1e-6, max_iter=1000):
        """Run value iteration and track progress"""
        V = {s: 0.0 for s in self.states}

        for state in self.terminals:
            V[state] = self.terminals[state]

        self.value_history = [V.copy()]

        for iteration in range(max_iter):
            V_new = V.copy()
            max_change = 0.0

            for state in self.states:
                if self.is_terminal(state):
                    continue

                q_values = []
                for action in self.actions:
                    q = self.get_reward(state)
                    for next_state, prob in self.transitions[state][action].items():
                        q += prob * self.gamma * V[next_state]
                    q_values.append(q)

                V_new[state] = max(q_values)
                max_change = max(max_change, abs(V_new[state] - V[state]))

            V = V_new
            self.value_history.append(V.copy())

            if max_change < threshold:
                break

        return V, iteration + 1

    def extract_policy(self, V):
        """Get optimal policy from values"""
        policy = {}

        for state in self.states:
            if self.is_terminal(state):
                policy[state] = '*'
                continue

            best_action = None
            best_q = float('-inf')

            for action in self.actions:
                q = self.get_reward(state)
                for next_state, prob in self.transitions[state][action].items():
                    q += prob * (self.gamma * V[next_state])

                if q > best_q:
                    best_q = q
                    best_action = action

            policy[state] = best_action

        return policy

    def reset_agent(self):
        """Reset agent to start position"""
        self.agent_pos = (self.rows-1, self.cols-1)
        return self.agent_pos

    def step(self, action, stochastic=True):
        """Take action and move agent"""
        if self.is_terminal(self.agent_pos):
            return self.agent_pos, 0, True

        if stochastic:
            # Sample next state based on transition probabilities
            next_states = list(self.transitions[self.agent_pos][action].keys())
            probs = list(self.transitions[self.agent_pos][action].values())
            next_state = random.choices(next_states, weights=probs)[0]
        else:
            # Deterministic movement
            delta = self.action_deltas[action]
            new_row = self.agent_pos[0] + delta[0]
            new_col = self.agent_pos[1] + delta[1]

            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                next_state = (new_row, new_col)
            else:
                next_state = self.agent_pos

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.agent_pos = next_state

        return next_state, reward, done

    def visualize_grid(self, values=None, policy=None, show_agent=True, title="Grid World"):
        """Create beautiful visualization of the grid world"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid cells
        for i in range(self.rows):
            for j in range(self.cols):
                state = (i, j)

                # Determine cell color
                if state in self.terminals:
                    if self.terminals[state] > 0:
                        color = '#90EE90'  # Light green for positive terminal
                    else:
                        color = '#FFB6C6'  # Light red for negative terminal
                elif state in self.rewards:
                    color = '#FFD700'  # Gold for reward cells
                else:
                    # Color based on value if available
                    if values is not None:
                        v = values[state]
                        max_v = max(abs(min(values.values())), abs(max(values.values())))
                        if max_v > 0:
                            intensity = abs(v) / max_v
                            if v >= 0:
                                color = (1 - 0.5*intensity, 1, 1 - 0.5*intensity)
                            else:
                                color = (1, 1 - 0.5*intensity, 1 - 0.5*intensity)
                        else:
                            color = 'white'
                    else:
                        color = 'white'

                rect = patches.Rectangle((j, i), 1, 1, linewidth=2,
                                        edgecolor='black', facecolor=color)
                ax.add_patch(rect)

                # Add value text
                if values is not None:
                    ax.text(j + 0.5, i + 0.3, f'{values[state]:.1f}',
                           ha='center', va='center', fontsize=10, weight='bold')

                # Add policy arrow
                if policy is not None and policy[state] != '*':
                    action = policy[state]
                    arrow_map = {
                        0: '↑', 1: '↓', 2: '←', 3: '→'
                    }
                    ax.text(j + 0.5, i + 0.7, arrow_map[action],
                           ha='center', va='center', fontsize=20, color='blue')

                # Mark terminal states
                if state in self.terminals:
                    reward = self.terminals[state]
                    ax.text(j + 0.5, i + 0.5, f'★\n{reward:.0f}',
                           ha='center', va='center', fontsize=16, weight='bold',
                           color='darkgreen' if reward > 0 else 'darkred')

                # Mark reward states
                elif state in self.rewards:
                    ax.text(j + 0.5, i + 0.5, f'◆\n{self.rewards[state]:.0f}',
                           ha='center', va='center', fontsize=12, weight='bold',
                           color='orange')

        # Draw agent
        if show_agent and self.agent_pos is not None:
            i, j = self.agent_pos
            circle = patches.Circle((j + 0.5, i + 0.5), 0.3,
                                   linewidth=3, edgecolor='red',
                                   facecolor='yellow', zorder=10)
            ax.add_patch(circle)
            ax.text(j + 0.5, i + 0.5, '🤖', ha='center', va='center',
                   fontsize=24, zorder=11)

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def run_episode(self, policy, max_steps=50, stochastic=True):
        """Run one episode following the policy"""
        self.reset_agent()
        trajectory = [self.agent_pos]
        total_reward = 0

        for step in range(max_steps):
            if self.is_terminal(self.agent_pos):
                break

            action = policy[self.agent_pos]
            if action == '*':
                break

            next_state, reward, done = self.step(action, stochastic)
            trajectory.append(next_state)
            total_reward += reward

            if done:
                break

        return trajectory, total_reward

    def get_stats(self):
        """Get game statistics"""
        return {
            'grid_size': f'{self.rows}x{self.cols}',
            'num_states': len(self.states),
            'terminals': len(self.terminals),
            'reward_cells': len(self.rewards),
            'gamma': self.gamma
        }

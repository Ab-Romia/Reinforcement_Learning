import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import io
from PIL import Image
import time


class GridWorldGame:
    """
    Interactive Grid World RL Game with comprehensive features and error handling
    """

    def __init__(self, r_value=100, gamma=0.99, grid_size=5):
        # Input validation
        if not isinstance(grid_size, int) or grid_size < 3 or grid_size > 10:
            raise ValueError("Grid size must be an integer between 3 and 10")
        if not 0 <= gamma < 1:
            raise ValueError("Gamma must be between 0 and 1")

        self.rows = grid_size
        self.cols = grid_size
        self.gamma = float(gamma)
        self.r_value = float(r_value)

        # Create states
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]

        # Terminal states - strategic placement
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
        self.start_pos = self.agent_pos

        # Build transitions
        self.transitions = self._build_transitions()

        # Learning history
        self.value_history = []
        self.policy_history = []
        self.convergence_data = []

    def is_terminal(self, state):
        """Check if state is terminal"""
        return state in self.terminals

    def get_reward(self, state):
        """Get reward for a state"""
        try:
            if state in self.terminals:
                return self.terminals[state]
            if state in self.rewards:
                return self.rewards[state]
            return -1.0  # Step cost
        except Exception as e:
            print(f"Error getting reward: {e}")
            return 0.0

    def _build_transitions(self):
        """Build stochastic transition model: 70% intended, 15% each perpendicular"""
        transitions = {}

        try:
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

        except Exception as e:
            print(f"Error building transitions: {e}")
            raise

        return transitions

    def value_iteration(self, threshold=1e-6, max_iter=2000):
        """Run value iteration with progress tracking"""
        try:
            V = {s: 0.0 for s in self.states}

            for state in self.terminals:
                V[state] = self.terminals[state]

            self.value_history = [V.copy()]
            self.convergence_data = []

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

                # Record convergence progress every 10 iterations
                if iteration % 10 == 0 or max_change < threshold:
                    self.convergence_data.append({
                        'iteration': iteration + 1,
                        'max_change': max_change,
                        'avg_value': np.mean(list(V.values()))
                    })

                if iteration % 50 == 0:
                    self.value_history.append(V.copy())

                if max_change < threshold:
                    break

            return V, iteration + 1

        except Exception as e:
            print(f"Error in value iteration: {e}")
            raise

    def policy_iteration(self, max_iter=100):
        """Run policy iteration algorithm"""
        try:
            # Initialize random policy
            policy = {}
            for state in self.states:
                if self.is_terminal(state):
                    policy[state] = '*'
                else:
                    policy[state] = random.choice(self.actions)

            self.policy_history = [policy.copy()]

            for iteration in range(max_iter):
                # Policy Evaluation
                V = self._policy_evaluation(policy)

                # Policy Improvement
                policy_stable = True
                new_policy = {}

                for state in self.states:
                    if self.is_terminal(state):
                        new_policy[state] = '*'
                        continue

                    old_action = policy[state]

                    # Find best action
                    best_action = None
                    best_q = float('-inf')

                    for action in self.actions:
                        q = self.get_reward(state)
                        for next_state, prob in self.transitions[state][action].items():
                            q += prob * self.gamma * V[next_state]

                        if q > best_q:
                            best_q = q
                            best_action = action

                    new_policy[state] = best_action

                    if old_action != best_action:
                        policy_stable = False

                policy = new_policy
                self.policy_history.append(policy.copy())

                if policy_stable:
                    break

            # Final evaluation
            V = self._policy_evaluation(policy)
            return policy, V, iteration + 1

        except Exception as e:
            print(f"Error in policy iteration: {e}")
            raise

    def _policy_evaluation(self, policy, threshold=1e-6, max_iter=1000):
        """Evaluate a given policy"""
        try:
            V = {s: 0.0 for s in self.states}

            for state in self.terminals:
                V[state] = self.terminals[state]

            for iteration in range(max_iter):
                V_new = V.copy()
                max_change = 0.0

                for state in self.states:
                    if self.is_terminal(state):
                        continue

                    action = policy[state]
                    if action == '*':
                        continue

                    v = self.get_reward(state)
                    for next_state, prob in self.transitions[state][action].items():
                        v += prob * self.gamma * V[next_state]

                    V_new[state] = v
                    max_change = max(max_change, abs(V_new[state] - V[state]))

                V = V_new

                if max_change < threshold:
                    break

            return V

        except Exception as e:
            print(f"Error in policy evaluation: {e}")
            return {s: 0.0 for s in self.states}

    def extract_policy(self, V):
        """Extract optimal policy from value function"""
        try:
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

        except Exception as e:
            print(f"Error extracting policy: {e}")
            return {s: 0 for s in self.states}

    def reset_agent(self):
        """Reset agent to start position"""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action, stochastic=True):
        """Take action and move agent"""
        try:
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

        except Exception as e:
            print(f"Error in step: {e}")
            return self.agent_pos, 0, True

    def visualize_grid(self, values=None, policy=None, show_agent=True,
                       title="Grid World", highlight_path=None):
        """Create beautiful visualization of the grid world"""
        try:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Calculate cell size for text scaling
            cell_size = min(12 / self.rows, 12 / self.cols)

            # Draw grid cells
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)

                    # Determine cell color
                    if state in self.terminals:
                        if self.terminals[state] > 0:
                            color = '#4CAF50'  # Green for positive terminal
                        else:
                            color = '#F44336'  # Red for negative terminal
                    elif state in self.rewards:
                        color = '#FFD700'  # Gold for reward cells
                    elif highlight_path and state in highlight_path:
                        color = '#B3E5FC'  # Light blue for path
                    else:
                        # Color based on value if available
                        if values is not None:
                            v = values[state]
                            all_vals = [values[s] for s in self.states if not self.is_terminal(s)]
                            if all_vals:
                                max_v = max(abs(min(all_vals)), abs(max(all_vals)))
                                if max_v > 0:
                                    intensity = min(abs(v) / max_v, 1.0)
                                    if v >= 0:
                                        color = (0.9 - 0.4*intensity, 0.95, 0.9 - 0.4*intensity)
                                    else:
                                        color = (0.95, 0.9 - 0.4*intensity, 0.9 - 0.4*intensity)
                                else:
                                    color = '#FFFFFF'
                            else:
                                color = '#FFFFFF'
                        else:
                            color = '#FFFFFF'

                    rect = patches.Rectangle((j, i), 1, 1, linewidth=2.5,
                                            edgecolor='#333333', facecolor=color, alpha=0.9)
                    ax.add_patch(rect)

                    # Add value text
                    if values is not None and not self.is_terminal(state):
                        fontsize = max(8, min(12, 120 / self.rows))
                        ax.text(j + 0.5, i + 0.25, f'{values[state]:.1f}',
                               ha='center', va='center', fontsize=fontsize,
                               weight='bold', color='#333333')

                    # Add policy arrow
                    if policy is not None and state in policy and policy[state] != '*':
                        action = policy[state]
                        arrow_map = {
                            0: '↑', 1: '↓', 2: '←', 3: '→'
                        }
                        arrow_fontsize = max(16, min(24, 200 / self.rows))
                        ax.text(j + 0.5, i + 0.7, arrow_map[action],
                               ha='center', va='center', fontsize=arrow_fontsize,
                               color='#1976D2', weight='bold')

                    # Mark terminal states
                    if state in self.terminals:
                        reward = self.terminals[state]
                        symbol_fontsize = max(14, min(20, 180 / self.rows))
                        text_fontsize = max(10, min(14, 140 / self.rows))
                        ax.text(j + 0.5, i + 0.4, '★',
                               ha='center', va='center', fontsize=symbol_fontsize,
                               color='white', weight='bold')
                        ax.text(j + 0.5, i + 0.7, f'{reward:.0f}',
                               ha='center', va='center', fontsize=text_fontsize,
                               color='white', weight='bold')

                    # Mark reward states
                    elif state in self.rewards:
                        symbol_fontsize = max(12, min(16, 150 / self.rows))
                        text_fontsize = max(9, min(12, 120 / self.rows))
                        ax.text(j + 0.5, i + 0.4, '◆',
                               ha='center', va='center', fontsize=symbol_fontsize,
                               color='#FF6F00', weight='bold')
                        ax.text(j + 0.5, i + 0.7, f'{self.rewards[state]:.0f}',
                               ha='center', va='center', fontsize=text_fontsize,
                               color='#FF6F00', weight='bold')

            # Draw agent
            if show_agent and self.agent_pos is not None:
                i, j = self.agent_pos
                circle_radius = min(0.35, 4 / self.rows)
                circle = patches.Circle((j + 0.5, i + 0.5), circle_radius,
                                       linewidth=3, edgecolor='#D32F2F',
                                       facecolor='#FFD54F', zorder=10)
                ax.add_patch(circle)

                agent_fontsize = max(20, min(30, 250 / self.rows))
                ax.text(j + 0.5, i + 0.5, '🤖', ha='center', va='center',
                       fontsize=agent_fontsize, zorder=11)

            ax.set_xlim(0, self.cols)
            ax.set_ylim(0, self.rows)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(title, fontsize=18, weight='bold', pad=20, color='#333333')
            ax.axis('off')

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error in visualization: {e}")
            # Return blank figure on error
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, f'Visualization Error: {str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

    def run_episode(self, policy, max_steps=100, stochastic=True):
        """Run one episode following the policy"""
        try:
            self.reset_agent()
            trajectory = [self.agent_pos]
            total_reward = 0
            rewards_collected = []

            for step in range(max_steps):
                if self.is_terminal(self.agent_pos):
                    break

                action = policy.get(self.agent_pos, 0)
                if action == '*':
                    break

                next_state, reward, done = self.step(action, stochastic)
                trajectory.append(next_state)
                total_reward += reward
                rewards_collected.append(reward)

                if done:
                    break

            return trajectory, total_reward, rewards_collected

        except Exception as e:
            print(f"Error running episode: {e}")
            return [self.start_pos], 0, []

    def get_stats(self):
        """Get comprehensive game statistics"""
        try:
            return {
                'grid_size': f'{self.rows}x{self.cols}',
                'num_states': len(self.states),
                'terminals': len(self.terminals),
                'reward_cells': len(self.rewards),
                'gamma': self.gamma,
                'center_reward': self.r_value,
                'start_pos': self.start_pos,
                'terminal_rewards': list(self.terminals.values())
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def compare_algorithms(self):
        """Compare Value Iteration and Policy Iteration"""
        try:
            # Value Iteration
            start_time = time.time()
            vi_values, vi_iters = self.value_iteration()
            vi_time = time.time() - start_time
            vi_policy = self.extract_policy(vi_values)

            # Policy Iteration
            start_time = time.time()
            pi_policy, pi_values, pi_iters = self.policy_iteration()
            pi_time = time.time() - start_time

            # Check if policies match
            policies_match = all(
                vi_policy.get(s) == pi_policy.get(s)
                for s in self.states if not self.is_terminal(s)
            )

            return {
                'vi_iterations': vi_iters,
                'vi_time': vi_time,
                'vi_policy': vi_policy,
                'vi_values': vi_values,
                'pi_iterations': pi_iters,
                'pi_time': pi_time,
                'pi_policy': pi_policy,
                'pi_values': pi_values,
                'policies_match': policies_match
            }

        except Exception as e:
            print(f"Error comparing algorithms: {e}")
            return None

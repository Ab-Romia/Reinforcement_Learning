import numpy as np
import random
import matplotlib.pyplot as plt


class GridMDP:
    """
    Simple 3x3 Grid World MDP solver using Value Iteration and Policy Iteration.
    """

    def __init__(self, r_value, gamma=0.99):
        self.rows = 3
        self.cols = 3
        self.gamma = gamma  # discount factor
        self.r_value = float(r_value)

        # Create all states
        self.states = [(i, j) for i in range(3) for j in range(3)]

        # Define terminal states and their rewards
        self.terminals = { (0,0): self.r_value, (0, 2): 10.0}

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = [0, 1, 2, 3]
        self.action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

        # Precompute transition probabilities
        self.transitions = self._build_transitions()

    def is_terminal(self, state):
        return state in self.terminals

    def get_reward(self, state):
        """Get immediate reward for being in a state"""
        if state in self.terminals:
            return self.terminals[state]
        return -1.0

    def _build_transitions(self):
        """Build transition probability matrix P(s'|s,a)"""
        transitions = {}

        for state in self.states:
            transitions[state] = {}

            if self.is_terminal(state):
                # Terminal states stay in place
                for action in self.actions:
                    transitions[state][action] = {state: 1.0}
                continue

            for action in self.actions:
                transitions[state][action] = {}

                # Define movement deltas for each action
                deltas = {
                    0: [(-1, 0), (0, -1), (0, 1)],  # Up: main=up, sides=left,right
                    1: [(1, 0), (0, 1), (0, -1)],  # Down: main=down, sides=right,left
                    2: [(0, -1), (1, 0), (-1, 0)],  # Left: main=left, sides=down,up
                    3: [(0, 1), (-1, 0), (1, 0)]  # Right: main=right, sides=up,down
                }

                # Probabilities: 80% intended, 10% each side
                probs = [0.8, 0.1, 0.1]

                for delta, prob in zip(deltas[action], probs):
                    new_row = state[0] + delta[0]
                    new_col = state[1] + delta[1]

                    # Check bounds
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        next_state = (new_row, new_col)
                    else:
                        next_state = state  # Hit wall, stay in place

                    # Add probability (handle multiple ways to reach same state)
                    if next_state in transitions[state][action]:
                        transitions[state][action][next_state] += prob
                    else:
                        transitions[state][action][next_state] = prob

        return transitions

    def value_iteration(self, threshold=1e-6, max_iter=1000):
        """Solve MDP using Value Iteration"""
        # Initialize values
        V = {s: self.get_reward(s) for s in self.states}

        # Set terminal state values
        for state, reward in self.terminals.items():
            V[state] = reward

        for iteration in range(max_iter):
            V_new = V.copy()
            max_change = 0.0

            for state in self.states:
                if self.is_terminal(state):
                    continue

                # Compute Q-values for all actions
                q_values = []
                for action in self.actions:
                    q_val = 0.0
                    for next_state, prob in self.transitions[state][action].items():
                        reward = self.get_reward(state)
                        q_val += prob * (reward + self.gamma * V[next_state])
                    q_values.append(q_val)

                # Take maximum Q-value
                V_new[state] = max(q_values)
                max_change = max(max_change, abs(V_new[state] - V[state]))

            V = V_new

            if max_change < threshold:
                print(f"Value Iteration converged in {iteration + 1} iterations")
                break

        return V

    def extract_policy(self, V):
        """Extract optimal policy from value function"""
        policy = {}

        for state in self.states:
            if self.is_terminal(state):
                policy[state] = '*'  # Terminal marker
                continue

            # Find best action
            best_action = None
            best_value = float('-inf')

            for action in self.actions:
                q_val = 0.0
                for next_state, prob in self.transitions[state][action].items():
                    reward = self.get_reward(state)
                    q_val += prob * (reward + self.gamma * V[next_state])

                if q_val > best_value:
                    best_value = q_val
                    best_action = action

            policy[state] = self.action_names[best_action]

        return policy

    def policy_evaluation(self, policy, threshold=1e-6, max_iter=1000):
        """Evaluate a given policy"""
        V = {s: self.get_reward(s) for s in self.states}

        # Set terminal values
        for state, reward in self.terminals.items():
            V[state] = reward

        for iteration in range(max_iter):
            V_new = V.copy()
            max_change = 0.0

            for state in self.states:
                if self.is_terminal(state):
                    continue

                # Get action from policy
                action_name = policy[state]
                action = None
                for a, name in self.action_names.items():
                    if name == action_name:
                        action = a
                        break

                if action is None:
                    continue

                # Compute value under this policy
                v_val = 0.0
                for next_state, prob in self.transitions[state][action].items():
                    reward = self.get_reward(state)
                    v_val += prob * (reward + self.gamma * V[next_state])

                V_new[state] = v_val
                max_change = max(max_change, abs(V_new[state] - V[state]))

            V = V_new

            if max_change < threshold:
                break

        return V

    def policy_iteration(self, max_iter=100):
        """Solve MDP using Policy Iteration"""
        # Initialize random policy
        policy = {}
        for state in self.states:
            if self.is_terminal(state):
                policy[state] = '*'
            else:
                policy[state] = self.action_names[random.choice(self.actions)]

        for iteration in range(max_iter):
            # Policy Evaluation
            V = self.policy_evaluation(policy)

            # Policy Improvement
            policy_changed = False
            new_policy = {}

            for state in self.states:
                if self.is_terminal(state):
                    new_policy[state] = '*'
                    continue

                # Find best action
                best_action = None
                best_value = float('-inf')

                for action in self.actions:
                    q_val = 0.0
                    for next_state, prob in self.transitions[state][action].items():
                        reward = self.get_reward(state)
                        q_val += prob * (reward + self.gamma * V[next_state])

                    if q_val > best_value:
                        best_value = q_val
                        best_action = action

                new_policy[state] = self.action_names[best_action]

                if new_policy[state] != policy[state]:
                    policy_changed = True

            policy = new_policy

            if not policy_changed:
                print(f"Policy Iteration converged in {iteration + 1} iterations")
                break

        # Final evaluation
        V = self.policy_evaluation(policy)
        return policy, V


class GridVisualizer:
    """Handles visualization of grid world results"""

    def __init__(self):
        self.action_symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', '*': '*'}

    def plot_values(self, V, title, terminals=None):
        """Plot state values as a heatmap"""
        # Convert to matrix
        values = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                values[i, j] = V[(i, j)]

        fig, ax = plt.subplots(figsize=(6, 5))

        # Create heatmap
        im = ax.imshow(values, cmap='coolwarm')
        plt.colorbar(im, ax=ax, label='State Value')

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text_color = 'white' if abs(values[i, j]) > np.max(np.abs(values)) * 0.5 else 'black'
                weight = 'bold' if terminals and (i, j) in terminals else 'normal'
                ax.text(j, i, f'{values[i, j]:.2f}',
                        ha='center', va='center', color=text_color, weight=weight)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.tight_layout()
        return fig

    def plot_policy(self, policy, title, terminals=None):
        """Plot policy as arrows"""
        fig, ax = plt.subplots(figsize=(6, 5))

        # White background
        ax.imshow(np.ones((3, 3, 3)), vmin=0, vmax=1)

        # Add policy symbols
        for i in range(3):
            for j in range(3):
                symbol = self.action_symbols.get(policy[(i, j)], policy[(i, j)])
                color = 'darkblue' if terminals and (i, j) in terminals else 'black'
                weight = 'bold' if terminals and (i, j) in terminals else 'normal'
                size = 18 if terminals and (i, j) in terminals else 20

                ax.text(j, i, symbol, ha='center', va='center',
                        fontsize=size, color=color, weight=weight)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.tight_layout()
        return fig


def run_experiment():
    """Run the complete experiment for different r values"""
    r_values = [100, 3, 0, -3]
    visualizer = GridVisualizer()

    print("Grid World MDP Experiment")
    print("=" * 40)
    print("Grid: 3x3")
    print("Terminals: (0,0)=r, (0,2)=+10")
    print("Non-terminal reward: -1")
    print("Discount factor: 0.99")
    print("Transition: 80% intended, 10% each perpendicular")
    print()

    for r in r_values:
        print(f"\n--- r = {r} ---")

        # Create MDP
        mdp = GridMDP(r)

        # Value Iteration
        print("Running Value Iteration...")
        V_vi = mdp.value_iteration()
        policy_vi = mdp.extract_policy(V_vi)

        # Policy Iteration
        print("Running Policy Iteration...")
        policy_pi, V_pi = mdp.policy_iteration()

        # Check if policies match
        policies_match = True
        for state in mdp.states:
            if not mdp.is_terminal(state):
                if policy_vi[state] != policy_pi[state]:
                    policies_match = False
                    break

        print(f"Policies match: {policies_match}")

        # Visualize results
        visualizer.plot_values(V_vi, f"Values (VI) - r={r}", mdp.terminals)
        visualizer.plot_policy(policy_vi, f"Policy (VI) - r={r}", mdp.terminals)
        visualizer.plot_values(V_pi, f"Values (PI) - r={r}", mdp.terminals)
        visualizer.plot_policy(policy_pi, f"Policy (PI) - r={r}", mdp.terminals)

        # Print policy for quick reference
        print("Optimal Policy (VI):")
        for i in range(3):
            row = ""
            for j in range(3):
                symbol = visualizer.action_symbols.get(policy_vi[(i, j)], policy_vi[(i, j)])
                row += f"{symbol:^3}"
            print(row)

    print("\n--- Policy Intuition ---")
    print("r=100: High reward at (0,0) - agent should go there despite risk")
    print("r=3:   Moderate reward at (0,0) - might still prefer it over (0,2)")
    print("r=0:   No reward at (0,0) - should prefer (0,2) with +10")
    print("r=-3:  Negative reward at (0,0) - strongly avoid, go to (0,2)")

    plt.show()


if __name__ == "__main__":
    run_experiment()
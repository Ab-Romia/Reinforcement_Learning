# Grid World MDP: Value Iteration vs Policy Iteration

A comprehensive implementation and analysis of Markov Decision Process (MDP) algorithms for solving a 3×3 Grid World environment with stochastic transitions.

## Overview

This project implements and compares two fundamental reinforcement learning algorithms:
- **Value Iteration**: Iteratively computes optimal state values using the Bellman optimality equation
- **Policy Iteration**: Alternates between policy evaluation and policy improvement

The experiment investigates how different reward values at terminal states affect optimal policies in a stochastic environment.

## Environment Description

### Grid World Setup
- **Size**: 3×3 grid
- **Terminal States**: 
  - (0,0): Variable reward `r` (tested values: 100, 3, 0, -3)
  - (0,2): Fixed reward of +10
- **Step Cost**: -1 for all non-terminal states
- **Discount Factor**: γ = 0.99

### Stochastic Transitions
The environment uses a stochastic transition model:
- **80%** probability of moving in the intended direction
- **10%** probability of moving perpendicular (left)
- **10%** probability of moving perpendicular (right)
- If movement would go outside the grid, the agent stays in place

## Mathematical Formulation

### MDP Definition
The Grid World MDP is defined as a tuple ⟨S, A, P, R, γ⟩:

- **State Space S**: All positions in the 3×3 grid
- **Action Space A**: {Up, Down, Left, Right}
- **Transition Probabilities P(s'|s,a)**: Stochastic with 80%/10%/10% distribution
- **Reward Function R(s)**:
  - r at (0,0)
  - +10 at (0,2)
  - -1 elsewhere
- **Discount Factor γ**: 0.99

### Value Iteration
Iteratively applies the Bellman optimality equation:

```
V_{k+1}(s) = max_a Σ P(s'|s,a) [R(s) + γ V_k(s')]
```

### Policy Iteration
Alternates between:
1. **Policy Evaluation**: V^π(s) = Σ P(s'|s,π(s)) [R(s) + γ V^π(s')]
2. **Policy Improvement**: π'(s) = argmax_a Σ P(s'|s,a) [R(s) + γ V^π(s')]

## Installation

### Requirements
```bash
pip install numpy matplotlib seaborn pandas
```

### Files
- `flame.py`: Main implementation file containing:
  - `GridMDP` class with both algorithms
  - `GridVisualizer` class for result visualization
  - Experimental framework and analysis tools

## Usage

### Basic Example
```python
# Create Grid World MDP
mdp = GridMDP(terminal_reward=100)

# Run Value Iteration
values_vi = mdp.value_iteration()
policy_vi = mdp.extract_policy(values_vi)

# Run Policy Iteration
policy_pi, values_pi = mdp.policy_iteration()

# Visualize results
visualizer = GridVisualizer()
visualizer.plot_policy(policy_vi, title="Value Iteration Policy")
visualizer.plot_values(values_vi, title="State Values")
```

### Running Complete Experiment
```python
# Test different reward values
reward_values = [100, 3, 0, -3]
results = {}

for r in reward_values:
    mdp = GridMDP(terminal_reward=r)
    
    # Value Iteration
    values_vi = mdp.value_iteration()
    policy_vi = mdp.extract_policy(values_vi)
    
    # Policy Iteration
    policy_pi, values_pi = mdp.policy_iteration()
    
    # Store results
    results[r] = {
        'vi_policy': policy_vi,
        'pi_policy': policy_pi,
        'values': values_vi,
        'policies_match': policies_equal(policy_vi, policy_pi)
    }
```

## Key Experimental Results

### Convergence Performance
| r Value | Policies Match | VI Iterations | PI Iterations |
|---------|----------------|---------------|---------------|
| 100     | ✓              | 31            | 4             |
| 3       | ✓              | 26            | 3             |
| 0       | ✓              | 26            | 3             |
| -3      | ✓              | 26            | 3             |

### Policy Patterns

**r = 100 (High Reward):**
```
*  ←  *
↑  ←  ↓
↑  ←  ←
```
Mixed strategy: some states move toward (0,0), others toward (0,2)

**r = 3, 0, -3 (Lower Rewards):**
```
*  →  *
→  →  ↑
→  →  ↑
```
Uniform strategy: all states direct toward (0,2) with +10 reward

## Key Findings

### Critical Insights
1. **Threshold Effect**: A critical reward threshold exists between r=100 and r=3 where optimal policies completely change
2. **Policy Robustness**: Identical policies for r=3, 0, -3 demonstrate robust decision boundaries
3. **Algorithm Efficiency**: Policy Iteration consistently converged 6-8x faster than Value Iteration
4. **Stochastic Impact**: 80%/10%/10% transitions significantly influence policy decisions

### Practical Implications
- **Reward Design**: Small changes in reward ratios can cause dramatic policy shifts
- **Algorithm Choice**: Policy Iteration preferred for smaller state spaces due to faster convergence
- **Uncertainty Handling**: Stochastic transitions effectively reduce attractiveness of distant rewards

## Visualization Features

The implementation includes comprehensive visualization tools:

### Policy Visualization
- Arrow-based policy representation
- Color-coded terminal states
- Clear grid layout with state labels

### Value Function Visualization
- Heatmap representation of state values
- Numerical value display
- Color gradient indicating value magnitude

### Convergence Analysis
- Iteration count comparison
- Convergence rate visualization
- Policy difference analysis

## Code Structure

### GridMDP Class
```python
class GridMDP:
    def __init__(self, terminal_reward=0):
        # Initialize grid world parameters
        
    def value_iteration(self, threshold=1e-6, max_iter=1000):
        # Implement Value Iteration algorithm
        
    def policy_iteration(self, max_iter=100):
        # Implement Policy Iteration algorithm
        
    def policy_evaluation(self, policy, threshold=1e-6):
        # Evaluate given policy
        
    def extract_policy(self, values):
        # Extract optimal policy from values
```

### GridVisualizer Class
```python
class GridVisualizer:
    def plot_policy(self, policy, title="Policy"):
        # Visualize policy with arrows
        
    def plot_values(self, values, title="State Values"):
        # Visualize value function as heatmap
        
    def compare_results(self, results):
        # Compare multiple experimental results
```

## Extensions and Modifications

### Customization Options
- **Grid Size**: Modify dimensions in `GridMDP.__init__()`
- **Transition Probabilities**: Adjust stochastic model parameters
- **Reward Structure**: Change terminal and step rewards
- **Convergence Criteria**: Modify threshold and iteration limits

### Potential Enhancements
- **Larger Grids**: Scale to larger state spaces
- **Additional Algorithms**: Implement Q-Learning or SARSA
- **Animation**: Add step-by-step visualization
- **Performance Metrics**: Add detailed timing and memory analysis

## Authors

- **Abdelrahman Omar Abouroumia** (ID: 8368)
- **Zeyad Hesham Elsayed** (ID: 8226)

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Bellman, R. (1957). *Dynamic Programming*
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*

## License

This project is developed for academic purposes as part of a Reinforcement Learning course assignment.

---

**Note**: This implementation serves as an educational tool for understanding fundamental MDP algorithms. The code is optimized for clarity and educational value rather than production performance.

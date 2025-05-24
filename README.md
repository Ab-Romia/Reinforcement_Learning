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

#### Terminal State (0,0)
| r Value | Policies Match | VI Iterations | PI Iterations |
|---------|----------------|---------------|---------------|
| 100     | ✓              | 31            | 4             |
| 3       | ✓              | 26            | 3             |
| 0       | ✓              | 26            | 3             |
| -3      | ✓              | 26            | 3             |

#### Non-Terminal State (0,0)
| r Value | Policies Match | VI Iterations | PI Iterations |
|---------|----------------|---------------|---------------|
| 100     | ✓              | 2508          | 4             |
| 3       | ✓              | 2153          | 4             |
| 0       | ✓              | 36            | 4             |
| -3      | ✓              | 36            | 3             |

### Policy Patterns

#### Terminal State (0,0) Cases

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

#### Non-Terminal State (0,0) Cases

**r = 100 and r = 3 (Positive Rewards):**
```
↑  ←  *
↑  ←  ↓
↑  ←  ←
```
Strategy: Direct toward (0,0) first to collect reward, then proceed to terminal state

**r = 0 (Zero Reward):**
```
→  →  *
↑  ↑  ↑
↑  ↑  ↑
```
Strategy: Bypass (0,0) and move efficiently toward (0,2)

**r = -3 (Negative Reward):**
```
→  →  *
→  →  ↑
→  →  ↑
```
Strategy: Actively avoid (0,0) while navigating to (0,2)

## Key Findings

### Critical Insights

#### Reward Threshold Effects
1. **Terminal State Threshold**: Critical threshold between r=100 and r=3 where optimal policies completely change
2. **Non-Terminal Binary Threshold**: Sharp decision boundary between positive rewards (r=100, r=3) and non-positive rewards (r=0, r=-3)
3. **Positive Reward Equivalence**: In non-terminal cases, r=100 and r=3 produce identical policies, showing magnitude insensitivity within positive rewards
4. **Zero vs Negative Distinction**: r=0 and r=-3 create different avoidance patterns in non-terminal cases

#### Algorithm Performance
1. **Policy Iteration Dominance**: PI consistently converged in 3-4 iterations across all scenarios
2. **Value Iteration Variability**: VI performance highly dependent on reward structure and state type
   - Terminal cases: 26-31 iterations
   - Non-terminal positive rewards: 2000+ iterations  
   - Non-terminal zero/negative rewards: 36 iterations
3. **Perfect Agreement**: Both algorithms produced identical optimal policies in all 8 test cases

#### State Type Impact
1. **Terminal vs Non-Terminal Complexity**: Non-terminal states enable multi-step reward collection strategies
2. **Convergence Difficulty**: Non-terminal positive reward cases required dramatically more VI iterations
3. **Policy Sophistication**: Non-terminal cases show more nuanced decision-making patterns

### Practical Implications

#### Reward Design Guidelines
- **Reward Sign Criticality**: The sign of rewards (positive vs zero vs negative) is more important than magnitude
- **Threshold Awareness**: Small reward changes can cause complete policy restructuring
- **Multi-Step Considerations**: Non-terminal reward states enable complex sequential strategies

#### Algorithm Selection
- **Policy Iteration Preferred**: Consistently superior convergence for discrete MDPs of this scale
- **Computational Predictability**: PI offers reliable iteration counts regardless of reward structure
- **Implementation Validation**: Running both algorithms provides excellent correctness verification

#### Environment Design
- **Stochastic Impact**: 80%/10%/10% transitions significantly reduce attractiveness of distant rewards
- **State Type Strategy**: Terminal vs non-terminal reward placement dramatically affects optimal behavior
- **Uncertainty Handling**: Stochastic environments favor simpler, more direct policies

## Visualization Features

The implementation includes comprehensive visualization tools for both terminal and non-terminal cases:

### Policy Visualization
- Arrow-based policy representation for all 9 states
- Color-coded terminal states (red) and reward states (green/yellow)
- Clear grid layout with coordinate labels
- Distinct visual patterns for different reward scenarios

### Value Function Visualization
- Heatmap representation showing value propagation
- Numerical value display for precise analysis
- Color gradients indicating relative value magnitudes
- Comparison views for terminal vs non-terminal cases

### Convergence Analysis
- Side-by-side iteration count comparison (VI vs PI)
- Convergence rate visualization across different reward values
- Policy difference analysis highlighting threshold effects
- Performance metrics for algorithm selection guidance

### Comparative Analysis Tools
- Terminal vs non-terminal result comparison
- Reward sensitivity analysis visualization
- Algorithm efficiency metrics
- Policy evolution tracking across iterations

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

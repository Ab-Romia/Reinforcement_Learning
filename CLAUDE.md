# CLAUDE.md - AI Assistant Guide for Grid World RL Game

This document provides comprehensive guidance for AI assistants working with this Reinforcement Learning codebase.

## Project Overview

**Name**: Grid World RL Game
**Type**: Educational Reinforcement Learning Web Application
**Purpose**: Interactive browser-based platform for visualizing and learning RL concepts through Value Iteration and Policy Iteration algorithms
**Tech Stack**: Python, Gradio, NumPy, Matplotlib
**Deployment Target**: Hugging Face Spaces
**License**: MIT

### What This Project Does

This is an educational tool that allows users to:
- Train RL agents using Value Iteration or Policy Iteration algorithms
- Visualize optimal policies and value functions on customizable grid worlds
- Watch agents navigate environments with stochastic transitions
- Compare algorithm performance and convergence characteristics
- Experiment with different parameters (grid size, rewards, discount factors)

### Target Audience

- RL students and learners
- Educators teaching Markov Decision Processes
- Developers exploring dynamic programming algorithms
- Anyone wanting visual intuition for RL concepts

---

## Codebase Structure

### File Organization

```
Reinforcement_Learning/
├── app.py              # [580 lines] Gradio web interface - PRIMARY ENTRY POINT
├── game.py             # [538 lines] Core RL logic and GridWorldGame class
├── flame.py            # [380 lines] Legacy 3x3 MDP reference implementation
├── requirements.txt    # Python dependencies (4 packages)
├── README.md           # User-facing documentation with HF metadata
├── .gitignore          # Standard Python + Gradio exclusions
└── CLAUDE.md           # This file - AI assistant guide
```

### Module Responsibilities

#### app.py - Web Interface Layer
**Role**: Provides Gradio UI for interacting with RL algorithms
**Key Functions**:
- `train_and_visualize()` - Main training endpoint (lines 38-150)
- `simulate_episode()` - Live agent simulation (lines 150-200)
- `compare_algorithms()` - VI vs PI comparison (lines 200-280)
- `fig_to_pil()` - Matplotlib to PIL conversion utility
- `safe_execute()` - Error handling decorator

**Entry Point**: `if __name__ == "__main__": demo.launch(...)`
**Server Config**: 0.0.0.0:7860, accessible at http://localhost:7860

#### game.py - Core RL Engine
**Role**: Grid world environment and RL algorithm implementations
**Main Class**: `GridWorldGame`

**Key Methods**:
- `__init__(r_value, gamma, grid_size)` - Environment setup with validation (lines 16-58)
- `value_iteration(threshold=1e-6, max_iter=2000)` - Bellman optimality implementation
- `policy_iteration(max_iter=100)` - Policy evaluation + improvement
- `extract_policy(V)` - Derives optimal policy from value function
- `run_episode(policy, max_steps, stochastic)` - Simulates agent execution
- `visualize_grid(values, policy, show_agent, title)` - Creates matplotlib visualizations
- `_build_transitions()` - Stochastic transition model (70% intended, 15% each perpendicular)
- `get_reward(state)` - Reward function (-1 step cost, terminals ±100/-50, center configurable)

**Environment Specifications**:
- Grid: 3x3 to 10x10 (validated)
- Terminal states: Top-right (+100), Bottom-left (-50)
- Reward cells: Center (r_value parameter, -50 to +50)
- Actions: 4 cardinal directions [Up=0, Down=1, Left=2, Right=3]
- Start position: Bottom-right corner
- Stochastic transitions with wall bouncing

#### flame.py - Legacy Reference
**Role**: Original 3x3 GridMDP implementation (kept for reference)
**Status**: Not used by app.py - standalone educational example
**Entry Point**: `run_experiment()` function
**Note**: Fixed 3x3 grid with different transition probabilities (80/10/10 vs 70/15/15)

---

## Architecture & Design Patterns

### Separation of Concerns

1. **game.py** = Model (RL logic, environment, algorithms)
2. **app.py** = View + Controller (UI, user interaction, workflow orchestration)
3. **flame.py** = Reference implementation (educational comparison)

### Key Design Principles

- **Input Validation**: All user inputs validated before algorithm execution
- **Error Handling**: Try-except blocks with user-friendly error messages
- **Stateful Game Objects**: GridWorldGame instances maintain environment state
- **Functional Visualization**: Pure functions convert state → matplotlib figures → PIL images
- **Convergence Tracking**: Algorithms record iteration history for analysis

### Data Flow

```
User Input (Gradio UI)
    ↓
app.py: train_and_visualize()
    ↓
game.py: GridWorldGame(params)
    ↓
game.py: value_iteration() OR policy_iteration()
    ↓
game.py: extract_policy(values)
    ↓
game.py: run_episode(policy)
    ↓
game.py: visualize_grid(values, policy)
    ↓
app.py: fig_to_pil(fig)
    ↓
Gradio Image Output
```

### State Management

**GridWorldGame State**:
- `self.states`: List of (row, col) tuples
- `self.terminals`: Dict {state: terminal_reward}
- `self.rewards`: Dict {state: intermediate_reward}
- `self.transitions`: Nested dict {state: {action: {next_state: probability}}}
- `self.agent_pos`: Current agent position (row, col)
- `self.convergence_data`: List of training metrics

**Immutability**: Value functions and policies are dicts (not modified in-place during iteration)

---

## Development Workflows

### Running Locally

```bash
# Standard workflow
git clone https://github.com/Ab-Romia/Reinforcement_Learning.git
cd Reinforcement_Learning
pip install -r requirements.txt
python app.py
# Navigate to http://localhost:7860
```

### Making Changes

#### To Modify RL Algorithms:
1. Edit `game.py` - modify algorithm implementations
2. Update convergence tracking if needed
3. Test via `python app.py` and UI interaction
4. Verify visualizations render correctly

#### To Modify UI/UX:
1. Edit `app.py` - modify Gradio components
2. Update layout, styling, or parameter ranges
3. Test responsiveness and error handling
4. Ensure backward compatibility with game.py API

#### To Add New Features:
1. **New Algorithms**: Add method to GridWorldGame class
2. **New Visualizations**: Add helper function in game.py or app.py
3. **New Parameters**: Update both game.py.__init__() and app.py UI components
4. **New Experiment Presets**: Add to `create_demo_examples()` in app.py

### Testing Approach

**Current Status**: No formal unit tests (educational project)

**Testing Strategy**:
- Manual testing via web interface
- Visual validation of results
- Input validation in code (raises ValueError for invalid params)
- Error handling with full tracebacks shown to users

**Recommended Testing** (if adding):
- Unit tests for algorithm correctness (verify VI/PI converge)
- Integration tests for UI workflows
- Edge cases: 3x3 grid, 10x10 grid, gamma near 0 and 1

### Git Workflow

**Current Branch**: `claude/claude-md-mhyblrx7qmfkjth1-01KwbFn4SQ5jT1oFToyiFiz5`

**Important**:
- Develop on the designated Claude branch
- Commit messages should be descriptive
- Push with: `git push -u origin <branch-name>`
- Branch names must start with `claude/` and match session ID

**Recent Commit Pattern**:
- Focus on dependency updates (Gradio 5.49+ compatibility)
- Infrastructure cleanup (gitignore, README metadata)
- Bug fixes for schema errors

---

## Coding Conventions & Patterns

### Python Style

- **Style Guide**: Generally follows PEP 8
- **Line Length**: ~100-120 characters (not strict)
- **Docstrings**: Present for classes and key methods
- **Type Hints**: Not used (NumPy arrays, dicts used extensively)

### Naming Conventions

- **Classes**: PascalCase (e.g., `GridWorldGame`)
- **Functions**: snake_case (e.g., `train_and_visualize`)
- **Constants**: UPPERCASE (not many in this codebase)
- **Private Methods**: Prefix with `_` (e.g., `_build_transitions`)

### Key Patterns

#### Error Handling Pattern
```python
try:
    # Operation
    result = some_function()
except Exception as e:
    error_text = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
    print(error_text)
    return default_values
```

#### Input Validation Pattern
```python
if not (3 <= grid_size <= 10):
    raise ValueError("Grid size must be between 3 and 10")
if not 0 < gamma < 1:
    raise ValueError("Gamma must be between 0 and 1")
```

#### Figure Conversion Pattern
```python
# Always clean up matplotlib figures
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Critical: prevent memory leaks
    return img
```

### Algorithm Implementation Pattern

**Value Iteration Structure**:
```python
V = {s: 0.0 for s in self.states}  # Initialize
for iteration in range(max_iter):
    max_change = 0.0
    new_V = V.copy()

    for state in self.states:
        if self.is_terminal(state):
            continue

        # Compute Q-values for all actions
        q_values = []
        for action in self.actions:
            q = self.get_reward(state)
            for next_state, prob in self.transitions[state][action].items():
                q += self.gamma * prob * V[next_state]
            q_values.append(q)

        new_V[state] = max(q_values)
        max_change = max(max_change, abs(new_V[state] - V[state]))

    V = new_V
    if max_change < threshold:
        break  # Converged
```

---

## Common Tasks & How-Tos

### Task: Add a New RL Algorithm (e.g., Q-Learning)

1. **Add method to GridWorldGame class** (game.py):
   ```python
   def q_learning(self, alpha=0.1, epsilon=0.1, episodes=1000):
       """Q-Learning implementation"""
       Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
       # Implementation...
       return Q, num_episodes
   ```

2. **Add extraction method if needed**:
   ```python
   def extract_policy_from_q(self, Q):
       """Extract greedy policy from Q-function"""
       policy = {}
       for state in self.states:
           if self.is_terminal(state):
               continue
           policy[state] = max(self.actions, key=lambda a: Q[state][a])
       return policy
   ```

3. **Update app.py UI** to include new algorithm in dropdown:
   ```python
   algorithm_choice = gr.Radio(
       choices=["Value Iteration", "Policy Iteration", "Q-Learning"],
       value="Value Iteration",
       label="Algorithm"
   )
   ```

4. **Add case in train_and_visualize()**:
   ```python
   elif algorithm == "Q-Learning":
       Q, episodes = game.q_learning()
       policy = game.extract_policy_from_q(Q)
       values = {s: max(Q[s].values()) for s in self.states}
   ```

### Task: Modify Grid Environment

**To change terminal positions** (game.py:32-35):
```python
self.terminals = {
    (0, self.cols-1): 100.0,    # Current: top-right
    (self.rows-1, 0): -50.0,    # Current: bottom-left
    # Add more terminals:
    # (self.rows//2, self.cols//2): 200.0
}
```

**To add obstacles**:
1. Add `self.obstacles = {(2, 2), (3, 3)}` in `__init__`
2. Modify `_build_transitions()` to treat obstacles like walls
3. Update `visualize_grid()` to show obstacles with special marker

**To change transition probabilities** (game.py:89-100):
```python
# Current: 70/15/15 split
probs = [0.7, 0.15, 0.15]
# Change to deterministic:
probs = [1.0, 0.0, 0.0]
```

### Task: Customize Visualization

**Color schemes** (game.py - in visualize_grid method):
- Modify `colors` dict for cell coloring
- Change `plt.cm.RdYlGn` colormap
- Adjust font sizes based on grid_size

**Add new visual elements**:
```python
# In visualize_grid(), after grid setup:
if custom_markers:
    for state in custom_markers:
        row, col = state
        ax.text(col + 0.5, self.rows - row - 0.5, '★',
                ha='center', va='center', fontsize=20)
```

### Task: Deploy to Hugging Face Spaces

1. **Verify README metadata** (lines 1-10):
   ```yaml
   ---
   title: Grid World RL Game
   emoji: 🎮
   sdk: gradio
   sdk_version: "5.49.0"
   app_file: app.py
   ---
   ```

2. **Push to HF**:
   - Create Space on Hugging Face
   - Choose Gradio SDK
   - Upload: app.py, game.py, requirements.txt, README.md
   - Auto-deploys (checks app_file: app.py)

3. **Monitor logs** for Gradio version compatibility

---

## Important Gotchas & Considerations

### 1. Matplotlib Memory Leaks
**Problem**: Matplotlib figures accumulate in memory if not closed
**Solution**: Always call `plt.close(fig)` after converting to PIL
**Location**: app.py:10-22 in `fig_to_pil()`

### 2. Gradio Version Compatibility
**Issue**: Gradio 5.49+ introduced schema changes
**Fix**: Recent commits updated dependencies
**When modifying**: Check Gradio changelog for breaking changes
**Reference**: Commit 09e90b3

### 3. Grid Size Limitations
**Hard Limits**: 3 ≤ grid_size ≤ 10
**Reason**: Larger grids cause:
- Visualization readability issues (arrows too small)
- Longer convergence times
- Memory concerns

**If increasing**: Update validation in game.py:18 and app.py:47

### 4. Stochastic vs Deterministic Transitions
**Key Setting**: `stochastic` parameter in `run_episode()`
- `True`: Uses 70/15/15 transition probabilities (realistic)
- `False`: Uses 100% intended direction (deterministic optimal path)

**UI Default**: Simulation tab uses checkbox for user control

### 5. Terminal State Handling
**Critical**: Terminals must have self-loops in transitions
**Location**: game.py:83-86
**Why**: Prevents value propagation errors in algorithms

### 6. NumPy vs Python Types
**Pattern**: Convert inputs to float for NumPy compatibility
```python
self.gamma = float(gamma)  # Not just gamma
self.r_value = float(r_value)
```

### 7. Policy Dictionary Format
**Structure**: `{state: action_index}` NOT `{state: action_name}`
**Action Indices**: 0=Up, 1=Down, 2=Left, 3=Right
**Conversion**: Use `self.action_names[action_index]` for display

---

## Parameter Ranges & Defaults

### GridWorldGame Parameters

| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `r_value` | float | -50 to +50 | 100 | Center cell reward |
| `gamma` | float | 0 < γ < 1 | 0.99 | Discount factor |
| `grid_size` | int | 3 to 10 | 5 | Grid dimensions (NxN) |

### Algorithm Parameters

**Value Iteration**:
- `threshold`: 1e-6 (convergence criterion)
- `max_iter`: 2000 (safety limit)

**Policy Iteration**:
- `max_iter`: 100 (policy update limit)

**Episode Simulation**:
- `max_steps`: 100 (prevent infinite loops)
- `stochastic`: bool (transition randomness)

### UI Parameter Sliders (app.py)

```python
grid_size_slider = gr.Slider(3, 10, value=5, step=1)
reward_slider = gr.Slider(-50, 50, value=3, step=1)
gamma_slider = gr.Slider(0.80, 0.99, value=0.95, step=0.01)
```

---

## Dependencies & Versioning

### requirements.txt

```
gradio>=5.49.0        # Web UI framework (CRITICAL: >= 5.49 for schema fixes)
numpy>=1.24.0         # Numerical computing, arrays
matplotlib>=3.7.0     # Plotting and visualization
pillow>=10.0.0        # Image processing (PIL)
```

### Version Compatibility Notes

- **Gradio 5.49+**: Required for schema type fixes (see commit 09e90b3)
- **NumPy 1.24+**: Modern array operations
- **Matplotlib 3.7+**: Enhanced visualization features
- **Pillow 10.0+**: Security updates

### Python Version
- **Recommended**: Python 3.8+
- **Tested**: 3.10, 3.11
- **Not specified**: requirements.txt lacks python_requires

---

## Visualization System

### Grid Visualization Color Coding

**Cell Background Colors** (game.py visualize_grid):
```python
colors = {
    'terminal_positive': '#4CAF50',  # Green
    'terminal_negative': '#F44336',  # Red
    'reward_cell': '#FFD700',        # Gold
    'agent_path': '#B3E5FC',         # Light blue
    'regular': value-based gradient  # RdYlGn colormap
}
```

**Symbol Meanings**:
- `★` = Terminal state (large reward/penalty)
- `◆` = Reward cell (intermediate reward)
- `🤖` = Current agent position
- `↑↓←→` = Policy action arrows

### Matplotlib Configuration

```python
fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
fontsize = max(8, 120 / grid_size)  # Scale with grid size
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
```

### Image Pipeline

1. Create matplotlib figure
2. Render grid with colors, values, arrows
3. Save to BytesIO buffer (PNG format)
4. Load with PIL.Image
5. Close matplotlib figure (prevent leaks)
6. Return PIL Image to Gradio

---

## Deployment Checklist

### Before Pushing Code

- [ ] Test locally with `python app.py`
- [ ] Verify all parameter combinations work
- [ ] Check error handling shows user-friendly messages
- [ ] Ensure matplotlib figures are closed (no memory leaks)
- [ ] Validate Gradio version compatibility
- [ ] Update README if features changed

### For Hugging Face Spaces

- [ ] README.md has correct YAML metadata (lines 1-10)
- [ ] `app_file: app.py` points to correct file
- [ ] SDK version matches requirements.txt
- [ ] All files committed (app.py, game.py, requirements.txt, README.md)
- [ ] No .gitignored files needed for deployment
- [ ] Server config: `server_name="0.0.0.0"`

### Git Operations

- [ ] Commit messages are descriptive
- [ ] Branch follows `claude/*` pattern
- [ ] Push with `-u origin <branch-name>`
- [ ] Retry up to 4 times on network errors (exponential backoff)

---

## Educational Context

### Learning Objectives

This codebase teaches:
1. **Markov Decision Processes**: States, actions, transitions, rewards
2. **Value Functions**: V(s) represents expected cumulative reward
3. **Policies**: Mapping from states to actions
4. **Bellman Equations**: Recursive value relationships
5. **Dynamic Programming**: Value/Policy Iteration algorithms
6. **Stochastic Environments**: Probabilistic transitions (70/15/15)
7. **Discount Factors**: γ balances immediate vs future rewards

### Suggested Experiments (from README)

1. **Risk Profiles**: Compare γ=0.99 (patient) vs γ=0.85 (greedy)
2. **Reward Seeking**: Positive center (+30) vs negative center (-30)
3. **Scalability**: 3x3 simple grid vs 10x10 complex grid
4. **Algorithm Comparison**: VI vs PI convergence speed

### Code as Pedagogy

**Design choices for learning**:
- Visual feedback at every step
- Configurable parameters with immediate effects
- Side-by-side algorithm comparison
- Convergence plots show learning progress
- Stochastic toggle demonstrates uncertainty

---

## API Reference

### GridWorldGame Class

```python
class GridWorldGame:
    def __init__(self, r_value=100, gamma=0.99, grid_size=5):
        """
        Initialize grid world environment

        Args:
            r_value (float): Center cell reward (-50 to +50)
            gamma (float): Discount factor (0 to 1)
            grid_size (int): Grid dimension (3 to 10)

        Raises:
            ValueError: If parameters out of valid range
        """

    def value_iteration(self, threshold=1e-6, max_iter=2000):
        """
        Run Value Iteration algorithm

        Returns:
            tuple: (values_dict, num_iterations)
        """

    def policy_iteration(self, max_iter=100):
        """
        Run Policy Iteration algorithm

        Returns:
            tuple: (policy_dict, values_dict, num_iterations)
        """

    def extract_policy(self, V):
        """
        Extract greedy policy from value function

        Args:
            V (dict): Value function {state: value}

        Returns:
            dict: Policy {state: action_index}
        """

    def run_episode(self, policy, max_steps=100, stochastic=True):
        """
        Simulate agent following policy

        Args:
            policy (dict): {state: action_index}
            max_steps (int): Maximum steps before termination
            stochastic (bool): Use probabilistic transitions

        Returns:
            tuple: (trajectory_list, total_reward, reward_list)
        """

    def visualize_grid(self, values=None, policy=None,
                       show_agent=False, title="", highlight_path=None):
        """
        Create matplotlib visualization of grid world

        Args:
            values (dict): Value function for coloring
            policy (dict): Policy for arrow overlay
            show_agent (bool): Show agent position
            title (str): Figure title
            highlight_path (set): States to highlight

        Returns:
            matplotlib.figure.Figure
        """

    def get_stats(self):
        """
        Get environment statistics

        Returns:
            dict: Statistics including grid_size, num_states, terminals, etc.
        """

    def compare_algorithms(self):
        """
        Run and compare VI and PI

        Returns:
            dict: Comparison metrics (iterations, convergence, values)
        """
```

### app.py Main Functions

```python
def train_and_visualize(grid_size, reward_value, discount_factor, algorithm):
    """
    Main training endpoint

    Returns:
        tuple: (policy_img, path_img, status_text, stats_text, convergence_img)
    """

def simulate_episode(grid_size, reward_value, discount_factor, use_stochastic):
    """
    Run live simulation

    Returns:
        tuple: (episode_img, episode_info_text)
    """

def compare_algorithms(grid_size, reward_value, discount_factor):
    """
    Compare VI vs PI

    Returns:
        tuple: (vi_img, pi_img, comparison_plot, comparison_text)
    """
```

---

## Troubleshooting Guide

### Issue: Gradio Interface Won't Launch

**Symptoms**: Import errors, schema errors
**Solution**:
```bash
pip install --upgrade gradio>=5.49.0
```
**Reference**: Commit 09e90b3

### Issue: Visualization Not Showing

**Possible Causes**:
1. Matplotlib figure not closed → memory leak
2. PIL conversion failed
3. Gradio Image component issue

**Debug**:
```python
print(f"Figure created: {fig is not None}")
img = fig_to_pil(fig)
print(f"PIL image: {img.size if img else 'None'}")
```

### Issue: Algorithm Not Converging

**Check**:
1. Grid too large (>10x10)? Increase max_iter
2. Discount factor γ too high (>0.999)? Convergence slows
3. Threshold too strict (<1e-10)? Relax to 1e-6

**Debug**:
```python
# In value_iteration, add:
if iteration % 100 == 0:
    print(f"Iteration {iteration}: max_change={max_change}")
```

### Issue: Agent Gets Stuck in Simulation

**Causes**:
- Policy has loops (shouldn't happen with optimal policy)
- max_steps too low (increase to 200)

**Solution**: Check policy extraction logic

### Issue: Colors Look Wrong

**Check**:
- Value ranges: Large values saturate colormap
- Normalize values before coloring
- Adjust vmin/vmax in imshow

---

## Future Enhancement Ideas

### Potential Features
1. **More Algorithms**: Q-Learning, SARSA, Monte Carlo
2. **Custom Environments**: User-defined terminals, obstacles
3. **Animation**: Step-by-step value propagation
4. **3D Visualization**: For value functions over time
5. **Policy Comparison**: Overlay multiple policies
6. **Export Results**: Download figures, save sessions
7. **Mobile Optimization**: Responsive Gradio layout

### Code Quality Improvements
1. Add unit tests (pytest)
2. Type hints (mypy compliance)
3. Docstring standardization (Google/NumPy style)
4. CI/CD pipeline (GitHub Actions)
5. Code coverage reporting

### Performance Optimizations
1. Vectorize value iteration with NumPy
2. Cache transition probabilities
3. Parallel algorithm execution
4. Progressive rendering for large grids

---

## Quick Reference

### File Line Counts
- app.py: 580 lines
- game.py: 538 lines
- flame.py: 380 lines (legacy)

### Key Constants
- Default grid: 5x5
- Default gamma: 0.99
- Default r_value: 100
- Terminal rewards: +100 (top-right), -50 (bottom-left)
- Step cost: -1
- Transition probs: 70/15/15

### Run Commands
```bash
python app.py              # Start web interface
python flame.py            # Run legacy experiment
pip install -r requirements.txt  # Install deps
```

### Important URLs
- Local: http://localhost:7860
- Repo: https://github.com/Ab-Romia/Reinforcement_Learning
- Docs: README.md (comprehensive user guide)

---

## Conclusion

This codebase is well-structured for educational purposes with:
- Clear separation between RL logic (game.py) and UI (app.py)
- Comprehensive error handling and input validation
- Beautiful visualizations with matplotlib
- Ready for deployment on Hugging Face Spaces

When working with this code:
1. Maintain the separation of concerns
2. Always validate user inputs
3. Close matplotlib figures to prevent leaks
4. Test via the web interface
5. Keep visualizations readable (consider grid size)

For questions or issues, refer to:
- README.md for user-facing documentation
- This CLAUDE.md for technical guidance
- Recent commits for context on changes

**Last Updated**: 2025-11-14
**Codebase Version**: Gradio 5.49+ compatible

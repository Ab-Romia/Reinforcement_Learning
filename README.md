---
title: Grid World RL Game
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.49.0"
app_file: app.py
pinned: false
---

# 🎮 Grid World RL Game

An interactive reinforcement learning game where you can watch an AI agent learn to navigate a grid world in real-time! Built with Gradio and deployable on Hugging Face Spaces.

## 🌟 Features

- **Interactive Web Interface**: No coding required - just adjust parameters and watch the agent learn
- **Real-time Visualization**: See the agent's policy, value function, and decision-making process
- **Multiple Algorithms**: Compare Value Iteration and Policy Iteration
- **Customizable Environment**: Adjust grid size, rewards, and discount factors
- **Live Simulations**: Watch trained agents navigate the environment step-by-step
- **Educational**: Perfect for learning RL concepts visually

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Ab-Romia/Reinforcement_Learning.git
cd Reinforcement_Learning

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will open in your browser at `http://localhost:7860`

### Hugging Face Spaces

This app is ready to deploy on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the SDK
3. Upload all files from this repository
4. The app will automatically deploy!

## 🎯 How It Works

### The Environment

The agent navigates a grid world with:
- **Terminal States** (★): Episodes end here with big rewards or penalties
  - Top-right corner: +100 reward
  - Bottom-left corner: -50 penalty
- **Reward Cells** (◆): Intermediate rewards the agent can collect
- **Regular Cells**: Small negative reward (-1) to encourage efficiency
- **Stochastic Movement**: 70% chance of intended direction, 15% each perpendicular

### The Learning Process

The agent uses **Reinforcement Learning** algorithms to find the optimal policy:

1. **Value Iteration**:
   - Iteratively computes the value of each state
   - Converges to optimal policy
   - Uses the Bellman optimality equation

2. **Policy Iteration**:
   - Alternates between policy evaluation and improvement
   - Often converges faster than value iteration
   - Guaranteed to find optimal policy

### Key Concepts

- **Policy**: The agent's strategy - which action to take in each state
- **Value Function**: Expected total reward from each state
- **Discount Factor (γ)**: How much the agent values future vs immediate rewards
- **Stochastic Transitions**: Movement isn't always predictable, like in real life!

## 🎮 Using the Interface

### Training Tab

1. **Configure Parameters**:
   - Grid Size: 3x3 to 8x8
   - Center Reward: -50 to +50
   - Discount Factor: 0.8 to 0.99
   - Algorithm: Value or Policy Iteration

2. **Click "Train Agent"** to run the learning algorithm

3. **View Results**:
   - Learned policy with arrows showing optimal actions
   - Value function as background colors
   - Convergence statistics

### Simulation Tab

Watch a trained agent navigate the environment:
- Stochastic movement (more realistic)
- Step-by-step visualization
- Total reward tracking

## 🔬 Experiment Ideas

Try these configurations to see different behaviors:

1. **Risk Taker vs Risk Avoider**:
   - High γ (0.99): Values future rewards, takes longer paths
   - Low γ (0.85): Prefers quick rewards, takes shortcuts

2. **Reward Attraction**:
   - Positive center reward (+30): Agent seeks it out
   - Negative center reward (-30): Agent avoids it

3. **Grid Complexity**:
   - Small grid (3x3): Simple, fast learning
   - Large grid (8x8): Complex, more interesting policies

4. **Penalty Sensitivity**:
   - Adjust bottom-left penalty and see how the agent's path changes

## 📁 Project Structure

```
Reinforcement_Learning/
├── app.py              # Gradio web interface
├── game.py             # Grid World game logic
├── flame.py            # Original MDP implementation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧠 Technical Details

### Algorithms

**Value Iteration**:
```
V(s) ← max_a Σ P(s'|s,a)[R(s) + γV(s')]
```

**Policy Iteration**:
```
1. Policy Evaluation: Compute V^π(s)
2. Policy Improvement: π'(s) ← argmax_a Q(s,a)
3. Repeat until convergence
```

### Transition Model

For action `a` from state `s`:
- 70% probability: intended direction
- 15% probability: perpendicular left
- 15% probability: perpendicular right
- If move would exit grid: stay in current state

### Rewards

- Terminal positive: +100
- Terminal negative: -50
- Center cell: configurable
- Step cost: -1 (encourages efficiency)

## 🎨 Visualization Features

- **Color-coded values**: Warmer colors = higher values
- **Policy arrows**: Clear direction indicators
- **Special state markers**:
  - ★ Terminal states
  - ◆ Reward cells
  - 🤖 Agent position
- **Real-time updates**: See learning progress

## 🔧 Advanced Usage

### Programmatic Access

```python
from game import GridWorldGame

# Create custom environment
game = GridWorldGame(r_value=20, gamma=0.95, grid_size=6)

# Train agent
values, iterations = game.value_iteration()
policy = game.extract_policy(values)

# Run episode
trajectory, reward = game.run_episode(policy)
print(f"Total reward: {reward}")

# Visualize
fig = game.visualize_grid(values, policy)
```

### Custom Environments

Modify `game.py` to create your own environments:
- Change terminal state positions
- Add more reward cells
- Adjust transition probabilities
- Implement obstacles

## 📚 Learning Resources

Great for understanding:
- Markov Decision Processes (MDPs)
- Dynamic Programming in RL
- Value and Policy Iteration
- Stochastic environments
- Discount factors and their effects

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new RL algorithms (Q-learning, SARSA, etc.)
- Improve visualizations
- Add new environment features
- Create tutorials

## 📄 License

MIT License - feel free to use for learning and teaching!

## 🙏 Acknowledgments

Built with:
- [Gradio](https://gradio.app/) - Easy ML web interfaces
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization

Inspired by classic RL textbooks:
- Sutton & Barto - "Reinforcement Learning: An Introduction"
- Russell & Norvig - "Artificial Intelligence: A Modern Approach"

## 📧 Contact

Questions or suggestions? Open an issue or reach out!

---

**Have fun learning Reinforcement Learning!** 🚀

Try different configurations, observe how the agent learns, and build intuition for RL concepts. The best way to learn is by experimenting!

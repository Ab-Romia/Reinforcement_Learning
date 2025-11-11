import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from game import GridWorldGame
import io
from PIL import Image


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def train_and_visualize(grid_size, reward_value, discount_factor, algorithm):
    """Train the agent and return visualizations"""

    # Create game
    game = GridWorldGame(r_value=reward_value, gamma=discount_factor, grid_size=grid_size)

    # Train based on algorithm
    if algorithm == "Value Iteration":
        values, iterations = game.value_iteration()
        policy = game.extract_policy(values)
        info_text = f"✅ Converged in {iterations} iterations"
    else:
        # For simplicity, we'll use value iteration but could add policy iteration
        values, iterations = game.value_iteration()
        policy = game.extract_policy(values)
        info_text = f"✅ Converged in {iterations} iterations"

    # Create visualizations
    game.reset_agent()

    # Grid with values and policy
    fig1 = game.visualize_grid(values=values, policy=policy, show_agent=False,
                                title=f"Learned Policy & Values ({algorithm})")
    img1 = fig_to_pil(fig1)

    # Run an episode
    trajectory, total_reward = game.run_episode(policy, stochastic=False)

    # Visualize trajectory
    game.agent_pos = trajectory[-1] if trajectory else game.agent_pos
    fig2 = game.visualize_grid(values=None, policy=policy, show_agent=True,
                                title=f"Agent at Final Position")
    img2 = fig_to_pil(fig2)

    # Get stats
    stats = game.get_stats()
    stats_text = f"""
### Game Statistics:
- **Grid Size:** {stats['grid_size']}
- **Total States:** {stats['num_states']}
- **Terminal States:** {stats['terminals']}
- **Discount Factor (γ):** {stats['gamma']:.2f}
- **Episodes Completed:** 1
- **Total Reward:** {total_reward:.2f}
- **Steps Taken:** {len(trajectory)-1}

### Legend:
- 🤖 Agent
- ★ Terminal State (Green=Reward, Red=Penalty)
- ◆ Reward Cell
- Arrows show optimal policy direction
"""

    return img1, img2, info_text, stats_text


def simulate_episode(grid_size, reward_value, discount_factor):
    """Simulate a single episode with the trained agent"""

    game = GridWorldGame(r_value=reward_value, gamma=discount_factor, grid_size=grid_size)
    values, _ = game.value_iteration()
    policy = game.extract_policy(values)

    # Run episode
    trajectory, total_reward = game.run_episode(policy, stochastic=True)

    # Create frames for each step
    frames = []
    game.reset_agent()

    for i, pos in enumerate(trajectory):
        game.agent_pos = pos
        fig = game.visualize_grid(values=values, policy=policy, show_agent=True,
                                  title=f"Step {i}/{len(trajectory)-1} | Reward: {total_reward:.2f}")
        frames.append(fig_to_pil(fig))

    episode_info = f"""
### Episode Summary:
- **Steps:** {len(trajectory)-1}
- **Total Reward:** {total_reward:.2f}
- **Path:** {' → '.join([f'({p[0]},{p[1]})' for p in trajectory[:5]])}{'...' if len(trajectory) > 5 else ''}
    """

    return frames[0] if frames else None, episode_info


# Custom CSS
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🎮 Grid World RL Game</h1>
        <p>Interactive Reinforcement Learning Environment</p>
        <p>Watch an AI agent learn to navigate a grid world using Value Iteration!</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")

            grid_size = gr.Slider(
                minimum=3, maximum=8, value=5, step=1,
                label="Grid Size",
                info="Size of the grid world (NxN)"
            )

            reward_value = gr.Slider(
                minimum=-50, maximum=50, value=10, step=5,
                label="Center Reward",
                info="Reward value at the center cell"
            )

            discount_factor = gr.Slider(
                minimum=0.8, maximum=0.99, value=0.95, step=0.01,
                label="Discount Factor (γ)",
                info="How much the agent values future rewards"
            )

            algorithm = gr.Radio(
                choices=["Value Iteration", "Policy Iteration"],
                value="Value Iteration",
                label="Learning Algorithm"
            )

            train_btn = gr.Button("🚀 Train Agent", variant="primary", size="lg")
            simulate_btn = gr.Button("▶️ Simulate Episode", variant="secondary", size="lg")

            gr.Markdown("""
            ### 📚 How it works:

            The agent learns to navigate a grid world to maximize rewards:

            1. **Terminal States** (★): Episode ends here
               - Top-right: +100 reward
               - Bottom-left: -50 penalty

            2. **Reward Cell** (◆): Collects reward without ending

            3. **Movement**: Stochastic (70% intended direction, 15% each side)

            4. **Goal**: Learn optimal policy to maximize total reward
            """)

        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")

            with gr.Tab("Training Results"):
                training_info = gr.Textbox(label="Training Info", lines=2)
                learned_policy = gr.Image(label="Learned Policy & Values")
                final_state = gr.Image(label="Final Agent Position")
                stats_display = gr.Markdown()

            with gr.Tab("Live Simulation"):
                sim_info = gr.Markdown()
                sim_image = gr.Image(label="Agent Simulation")

    # Connect buttons
    train_btn.click(
        fn=train_and_visualize,
        inputs=[grid_size, reward_value, discount_factor, algorithm],
        outputs=[learned_policy, final_state, training_info, stats_display]
    )

    simulate_btn.click(
        fn=simulate_episode,
        inputs=[grid_size, reward_value, discount_factor],
        outputs=[sim_image, sim_info]
    )

    gr.Markdown("""
    ---
    ### 🎯 Tips:
    - **Higher discount factor** (γ): Agent values long-term rewards more
    - **Lower discount factor**: Agent prefers immediate rewards
    - **Positive center reward**: Agent will try to collect it before reaching terminal
    - **Negative center reward**: Agent will avoid the center

    ### 🔬 Experiment Ideas:
    1. Set center reward to -30 and watch the agent avoid it
    2. Try γ=0.9 vs γ=0.99 and see how policies differ
    3. Increase grid size to make the problem harder

    Made with ❤️ using Reinforcement Learning
    """)

# Launch
if __name__ == "__main__":
    demo.launch()

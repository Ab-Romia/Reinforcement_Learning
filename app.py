import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from game import GridWorldGame
import io
from PIL import Image
import traceback


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    except Exception as e:
        print(f"Error converting figure: {e}")
        plt.close(fig)
        return None


def safe_execute(func, error_message="An error occurred"):
    """Decorator for safe execution with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_text = f"❌ {error_message}: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            print(error_text)
            # Return appropriate defaults based on expected outputs
            return (None,) * 5  # Adjust based on function outputs
    return wrapper


@safe_execute
def train_and_visualize(grid_size, reward_value, discount_factor, algorithm):
    """Train the agent and return comprehensive visualizations"""
    try:
        # Validate inputs
        grid_size = int(grid_size)
        reward_value = float(reward_value)
        discount_factor = float(discount_factor)

        if not (3 <= grid_size <= 10):
            return None, None, "❌ Grid size must be between 3 and 10", "", None

        if not (0 < discount_factor < 1):
            return None, None, "❌ Discount factor must be between 0 and 1", "", None

        # Create game
        game = GridWorldGame(r_value=reward_value, gamma=discount_factor, grid_size=grid_size)

        # Train based on algorithm
        if algorithm == "Value Iteration":
            values, iterations = game.value_iteration()
            policy = game.extract_policy(values)
            info_text = f"✅ **{algorithm}** converged in **{iterations}** iterations"
            training_time = "N/A"
        elif algorithm == "Policy Iteration":
            policy, values, iterations = game.policy_iteration()
            info_text = f"✅ **{algorithm}** converged in **{iterations}** iterations"
            training_time = "N/A"
        else:
            return None, None, "❌ Invalid algorithm selected", "", None

        # Create visualizations
        game.reset_agent()

        # Grid with values and policy
        fig1 = game.visualize_grid(values=values, policy=policy, show_agent=False,
                                    title=f"Optimal Policy & Value Function ({algorithm})")
        img1 = fig_to_pil(fig1)

        # Run an episode to show agent's path
        trajectory, total_reward, rewards = game.run_episode(policy, stochastic=False)

        # Visualize with agent at final position
        game.agent_pos = trajectory[-1] if trajectory else game.agent_pos
        fig2 = game.visualize_grid(values=None, policy=policy, show_agent=True,
                                    title=f"Agent's Optimal Path", highlight_path=set(trajectory))
        img2 = fig_to_pil(fig2)

        # Create convergence plot if available
        convergence_fig = None
        if hasattr(game, 'convergence_data') and game.convergence_data:
            convergence_fig = plot_convergence(game.convergence_data)
            convergence_img = fig_to_pil(convergence_fig)
        else:
            convergence_img = None

        # Get stats
        stats = game.get_stats()
        stats_text = f"""
### 📊 Training Statistics

**Environment:**
- Grid Size: {stats['grid_size']}
- Total States: {stats['num_states']}
- Terminal States: {stats['terminals']}
- Discount Factor (γ): {stats['gamma']:.3f}

**Learning Results:**
- Algorithm: {algorithm}
- Iterations to Converge: {iterations}
- Average State Value: {np.mean(list(values.values())):.2f}
- Max State Value: {max(values.values()):.2f}
- Min State Value: {min(values.values()):.2f}

**Test Episode:**
- Steps Taken: {len(trajectory)-1}
- Total Reward: {total_reward:.2f}
- Average Reward per Step: {(total_reward / max(len(trajectory)-1, 1)):.2f}
- Path: {' → '.join([f'({p[0]},{p[1]})' for p in trajectory[:8]])}{'...' if len(trajectory) > 8 else ''}

### 📖 Legend:
- 🤖 **Agent** - Your RL agent
- ★ **Terminal States** (Green=Reward, Red=Penalty)
- ◆ **Reward Cell** (Collectable without ending episode)
- **Arrows** show the optimal action in each state
- **Numbers** show the value of each state
- **Colors** indicate relative state values
"""

        return img1, img2, info_text, stats_text, convergence_img

    except ValueError as ve:
        return None, None, f"❌ Input Error: {str(ve)}", "", None
    except Exception as e:
        error_msg = f"❌ Training Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg, "", None


def plot_convergence(convergence_data):
    """Plot convergence progress"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = [d['iteration'] for d in convergence_data]
        max_changes = [d['max_change'] for d in convergence_data]
        avg_values = [d['avg_value'] for d in convergence_data]

        # Max change over iterations
        ax1.plot(iterations, max_changes, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Iteration', fontsize=12, weight='bold')
        ax1.set_ylabel('Max Value Change', fontsize=12, weight='bold')
        ax1.set_title('Convergence Progress', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Average value over iterations
        ax2.plot(iterations, avg_values, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Iteration', fontsize=12, weight='bold')
        ax2.set_ylabel('Average State Value', fontsize=12, weight='bold')
        ax2.set_title('Value Function Evolution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error plotting convergence: {e}")
        return None


@safe_execute
def simulate_episode(grid_size, reward_value, discount_factor, use_stochastic):
    """Simulate a single episode with the trained agent"""
    try:
        grid_size = int(grid_size)
        reward_value = float(reward_value)
        discount_factor = float(discount_factor)

        game = GridWorldGame(r_value=reward_value, gamma=discount_factor, grid_size=grid_size)
        values, _ = game.value_iteration()
        policy = game.extract_policy(values)

        # Run episode
        trajectory, total_reward, rewards = game.run_episode(policy, stochastic=use_stochastic)

        # Create visualization at final position
        game.agent_pos = trajectory[-1] if trajectory else game.agent_pos
        fig = game.visualize_grid(values=values, policy=policy, show_agent=True,
                                  title=f"Episode Result (Stochastic: {use_stochastic})",
                                  highlight_path=set(trajectory))
        img = fig_to_pil(fig)

        # Detailed episode info
        episode_info = f"""
### 🎮 Episode Summary

**Configuration:**
- Movement Type: {'Stochastic (70% intended)' if use_stochastic else 'Deterministic (100% intended)'}
- Grid Size: {grid_size}x{grid_size}
- Center Reward: {reward_value}

**Results:**
- Total Steps: {len(trajectory)-1}
- Total Reward: {total_reward:.2f}
- Average Reward: {(total_reward / max(len(trajectory)-1, 1)):.2f}

**Trajectory:**
{' → '.join([f'({p[0]},{p[1]})' for p in trajectory])}

**Rewards Collected:**
{', '.join([f'{r:.1f}' for r in rewards[:20]])}{'...' if len(rewards) > 20 else ''}

**Performance:**
- {'✅ Successfully reached terminal state!' if game.is_terminal(trajectory[-1]) else '⚠️ Did not reach terminal (max steps)'}
- {'💰 Collected center reward!' if any(r == reward_value for r in rewards) else '❌ Missed center reward'}
"""

        return img, episode_info

    except Exception as e:
        error_msg = f"❌ Simulation Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


@safe_execute
def compare_algorithms(grid_size, reward_value, discount_factor):
    """Compare Value Iteration vs Policy Iteration"""
    try:
        grid_size = int(grid_size)
        reward_value = float(reward_value)
        discount_factor = float(discount_factor)

        game = GridWorldGame(r_value=reward_value, gamma=discount_factor, grid_size=grid_size)
        comparison = game.compare_algorithms()

        if comparison is None:
            return None, None, None, "❌ Comparison failed"

        # Visualize both policies
        fig1 = game.visualize_grid(values=comparison['vi_values'],
                                    policy=comparison['vi_policy'],
                                    show_agent=False,
                                    title="Value Iteration Policy")
        img1 = fig_to_pil(fig1)

        fig2 = game.visualize_grid(values=comparison['pi_values'],
                                    policy=comparison['pi_policy'],
                                    show_agent=False,
                                    title="Policy Iteration Policy")
        img2 = fig_to_pil(fig2)

        # Create comparison plot
        fig_comp = plot_algorithm_comparison(comparison)
        img_comp = fig_to_pil(fig_comp)

        # Comparison text
        match_emoji = "✅" if comparison['policies_match'] else "❌"
        comparison_text = f"""
### 🔬 Algorithm Comparison Results

**Value Iteration:**
- Iterations: {comparison['vi_iterations']}
- Time: {comparison['vi_time']:.4f} seconds
- Avg State Value: {np.mean(list(comparison['vi_values'].values())):.2f}

**Policy Iteration:**
- Iterations: {comparison['pi_iterations']}
- Time: {comparison['pi_time']:.4f} seconds
- Avg State Value: {np.mean(list(comparison['pi_values'].values())):.2f}

**Comparison:**
- Policies Match: {match_emoji} {comparison['policies_match']}
- Speed Winner: {'Value Iteration' if comparison['vi_time'] < comparison['pi_time'] else 'Policy Iteration'}
- Iteration Winner: {'Value Iteration' if comparison['vi_iterations'] < comparison['pi_iterations'] else 'Policy Iteration'}

**Analysis:**
{get_comparison_analysis(comparison)}
"""

        return img1, img2, img_comp, comparison_text

    except Exception as e:
        error_msg = f"❌ Comparison Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


def plot_algorithm_comparison(comparison):
    """Plot comparison between algorithms"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Iterations comparison
        algorithms = ['Value\nIteration', 'Policy\nIteration']
        iterations = [comparison['vi_iterations'], comparison['pi_iterations']]
        times = [comparison['vi_time'], comparison['pi_time']]

        axes[0].bar(algorithms, iterations, color=['#4CAF50', '#2196F3'], alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Iterations to Converge', fontsize=12, weight='bold')
        axes[0].set_title('Convergence Speed (Iterations)', fontsize=14, weight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(iterations):
            axes[0].text(i, v + max(iterations)*0.02, str(v), ha='center', va='bottom', fontsize=12, weight='bold')

        # Time comparison
        axes[1].bar(algorithms, times, color=['#FF9800', '#9C27B0'], alpha=0.8, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Time (seconds)', fontsize=12, weight='bold')
        axes[1].set_title('Computation Time', fontsize=14, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(times):
            axes[1].text(i, v + max(times)*0.02, f'{v:.3f}s', ha='center', va='bottom', fontsize=12, weight='bold')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error plotting comparison: {e}")
        return None


def get_comparison_analysis(comparison):
    """Generate analysis text for algorithm comparison"""
    try:
        analysis = []

        iter_ratio = comparison['vi_iterations'] / max(comparison['pi_iterations'], 1)
        if iter_ratio > 2:
            analysis.append("- Policy Iteration converged much faster (fewer iterations)")
        elif iter_ratio < 0.5:
            analysis.append("- Value Iteration converged much faster (fewer iterations)")
        else:
            analysis.append("- Both algorithms converged in similar number of iterations")

        time_ratio = comparison['vi_time'] / max(comparison['pi_time'], 0.001)
        if time_ratio > 1.5:
            analysis.append("- Policy Iteration was faster in wall-clock time")
        elif time_ratio < 0.67:
            analysis.append("- Value Iteration was faster in wall-clock time")
        else:
            analysis.append("- Both algorithms took similar time")

        if comparison['policies_match']:
            analysis.append("- Both algorithms found the same optimal policy (as expected)")
        else:
            analysis.append("- ⚠️ Algorithms found different policies (unusual - check convergence)")

        return '\n'.join(analysis)

    except Exception as e:
        return f"Analysis unavailable: {str(e)}"


def create_demo_examples():
    """Create example configurations for users to try"""
    return [
        ["5", "20", "0.95", "Value Iteration"],
        ["7", "-30", "0.90", "Policy Iteration"],
        ["4", "50", "0.99", "Value Iteration"],
        ["6", "0", "0.85", "Policy Iteration"],
    ]


# Custom CSS for better styling
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

.header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: bold;
}

.header p {
    margin: 10px 0 0 0;
    font-size: 1.2em;
    opacity: 0.95;
}

.config-section {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: bold !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
    font-weight: bold !important;
}

.footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 2px solid #e0e0e0;
    color: #666;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Grid World RL Game") as demo:

    gr.HTML("""
    <div class="header">
        <h1>🎮 Grid World Reinforcement Learning</h1>
        <p>Interactive AI Training Platform - Watch agents learn to navigate complex environments!</p>
        <p style="font-size: 0.9em; margin-top: 10px;">Built with Value Iteration & Policy Iteration Algorithms</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Environment Configuration")

            with gr.Group():
                grid_size = gr.Slider(
                    minimum=3, maximum=10, value=5, step=1,
                    label="🔲 Grid Size",
                    info="Size of the grid world (NxN). Larger grids are more complex!"
                )

                reward_value = gr.Slider(
                    minimum=-50, maximum=50, value=10, step=5,
                    label="💎 Center Reward",
                    info="Reward at center cell. Try negative values for avoidance behavior!"
                )

                discount_factor = gr.Slider(
                    minimum=0.80, maximum=0.99, value=0.95, step=0.01,
                    label="⏱️ Discount Factor (γ)",
                    info="How much agent values future rewards. Higher = more patient"
                )

                algorithm = gr.Radio(
                    choices=["Value Iteration", "Policy Iteration"],
                    value="Value Iteration",
                    label="🧮 Learning Algorithm",
                    info="Choose the RL algorithm to train the agent"
                )

            gr.Markdown("### 🎯 Quick Actions")

            train_btn = gr.Button("🚀 Train Agent", variant="primary", size="lg")
            simulate_btn = gr.Button("▶️ Run Simulation", variant="secondary", size="lg")
            compare_btn = gr.Button("⚖️ Compare Algorithms", variant="secondary", size="lg")

            gr.Markdown("""
            ### 📚 How It Works

            **The Environment:**
            - **★ Terminal States**: Episode ends (Green +100, Red -50)
            - **◆ Center Reward**: Configurable reward cell
            - **Step Cost**: -1 per move (encourages efficiency)
            - **Stochastic Movement**: 70% intended, 15% each side

            **The Algorithms:**
            1. **Value Iteration**: Computes optimal values for all states
            2. **Policy Iteration**: Alternates evaluation and improvement

            **Tips:**
            - High γ (0.99): Patient, plans ahead
            - Low γ (0.85): Impatient, wants quick rewards
            - Negative center reward: Agent avoids it
            - Positive center reward: Agent seeks it out
            """)

        with gr.Column(scale=2):
            gr.Markdown("## 📊 Training & Results")

            with gr.Tabs():
                with gr.Tab("🎓 Training Results"):
                    training_status = gr.Textbox(
                        label="Training Status",
                        lines=2,
                        interactive=False
                    )

                    with gr.Row():
                        policy_viz = gr.Image(
                            label="Learned Policy & Value Function",
                            type="pil"
                        )
                        path_viz = gr.Image(
                            label="Optimal Path Visualization",
                            type="pil"
                        )

                    convergence_plot = gr.Image(
                        label="Convergence Analysis",
                        type="pil"
                    )

                    stats_display = gr.Markdown(label="Statistics")

                with gr.Tab("🎮 Live Simulation"):
                    use_stochastic = gr.Checkbox(
                        label="Use Stochastic Movement",
                        value=True,
                        info="Enable random movement (more realistic)"
                    )

                    sim_image = gr.Image(label="Simulation Result", type="pil")
                    sim_info = gr.Markdown()

                with gr.Tab("⚖️ Algorithm Comparison"):
                    gr.Markdown("Compare Value Iteration vs Policy Iteration side-by-side")

                    with gr.Row():
                        vi_viz = gr.Image(label="Value Iteration", type="pil")
                        pi_viz = gr.Image(label="Policy Iteration", type="pil")

                    comparison_plot = gr.Image(label="Performance Comparison", type="pil")
                    comparison_text = gr.Markdown()

    # Connect buttons to functions
    train_btn.click(
        fn=train_and_visualize,
        inputs=[grid_size, reward_value, discount_factor, algorithm],
        outputs=[policy_viz, path_viz, training_status, stats_display, convergence_plot]
    )

    simulate_btn.click(
        fn=simulate_episode,
        inputs=[grid_size, reward_value, discount_factor, use_stochastic],
        outputs=[sim_image, sim_info]
    )

    compare_btn.click(
        fn=compare_algorithms,
        inputs=[grid_size, reward_value, discount_factor],
        outputs=[vi_viz, pi_viz, comparison_plot, comparison_text]
    )

    # Examples
    gr.Examples(
        examples=[
            [5, 20, 0.95, "Value Iteration"],
            [7, -30, 0.90, "Policy Iteration"],
            [4, 50, 0.99, "Value Iteration"],
            [6, 0, 0.85, "Policy Iteration"],
        ],
        inputs=[grid_size, reward_value, discount_factor, algorithm],
        label="💡 Try These Configurations"
    )

    gr.HTML("""
    <div class="footer">
        <h3>🔬 Experiment Ideas</h3>
        <p><strong>Risk vs Reward:</strong> Try γ=0.99 vs γ=0.85 to see patience vs greed</p>
        <p><strong>Avoidance:</strong> Set center reward to -30 and watch the agent avoid it</p>
        <p><strong>Complexity:</strong> Increase grid size to 8x8 for challenging navigation</p>
        <p><strong>Comparison:</strong> See which algorithm converges faster for different configurations</p>
        <hr>
        <p>Built with ❤️ using Reinforcement Learning | Gradio | Python</p>
        <p style="font-size: 0.9em; margin-top: 10px;">
            Deploy on Hugging Face Spaces • Perfect for learning RL concepts • Interactive & Educational
        </p>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

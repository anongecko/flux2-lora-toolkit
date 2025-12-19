"""
Optimization tab for the Gradio interface.

Provides hyperparameter optimization capabilities using Optuna.
"""

import os
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp

from .help_utils import help_system


def opt_generate_recommendations(results: Dict[str, Any], n_trials: int) -> str:
    best_score = results.get("best_score", 0)

    if best_score >= 0.8:
        return "🎉 **Excellent results!** Your optimization found very high-quality parameters. Use the best_config.yaml for production training."
    elif best_score >= 0.6:
        return "👍 **Good results!** The optimized parameters should provide noticeable improvements. Download and use best_config.yaml."
    elif best_score >= 0.4:
        return "🤔 **Fair results.** Consider improving your dataset quality before production training."
    else:
        return "⚠️ **Poor results.** Check dataset quality and consider different optimization settings."


def create_optimization_tab(app: "LoRATrainingApp"):
    """
    Create the optimization tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Optimization Help & Tips", open=False):
        gr.Markdown(
            "**Hyperparameter Optimization** automatically finds the best LoRA training settings for your dataset using Bayesian optimization."
        )

    with gr.Row():
        with gr.Column(scale=1):
            # Left column: Optimization setup
            gr.Markdown("## ⚙️ Optimization Setup")

            with gr.Group():
                gr.Markdown("### Dataset")

                # Dataset selection
                opt_dataset_source = gr.Radio(
                    choices=["Upload ZIP", "Local Directory"],
                    value="Upload ZIP",
                    label="Dataset Source",
                )

                opt_dataset_upload = gr.File(
                    label="Upload Dataset (ZIP)",
                    file_types=[".zip"],
                    visible=True,
                )

                opt_dataset_path = gr.Textbox(
                    label="Dataset Directory Path",
                    placeholder="/path/to/dataset",
                    visible=False,
                )

                opt_dataset_status = gr.Textbox(
                    label="Dataset Status",
                    value="No dataset selected",
                    interactive=False,
                )

                gr.Markdown("### Optimization Settings")

                # Optimization parameters
                opt_trials = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Number of Trials",
                )

                opt_max_steps = gr.Slider(
                    minimum=200,
                    maximum=1000,
                    value=500,
                    step=50,
                    label="Steps per Trial",
                )

                opt_timeout = gr.Number(
                    label="Timeout (hours)",
                    value=None,
                    minimum=0.1,
                )

                opt_study_name = gr.Textbox(
                    label="Study Name",
                    value="flux2_lora_optimization",
                )

                # Output settings
                opt_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./optimization_results",
                )

                # Preset configurations
                with gr.Accordion("🎯 Quick Presets", open=False):
                    gr.Markdown(
                        "**Quick (20 trials)**: Fast optimization for testing. **Standard (50 trials)**: Recommended for production. **Maximum (100 trials)**: Best results with longest time."
                    )

                    # Preset buttons
                    with gr.Row():
                        quick_preset = gr.Button("⚡ Quick (20 trials)", size="sm")
                        standard_preset = gr.Button("🎯 Standard (50 trials)", size="sm")
                        max_preset = gr.Button("🏆 Maximum (100 trials)", size="sm")

                gr.Markdown("### Optimization Controls")

                # Control buttons
                opt_start_btn = gr.Button(
                    "🚀 Start Optimization",
                    variant="primary",
                    size="lg",
                )

                opt_stop_btn = gr.Button(
                    "⏹️ Stop Optimization",
                    variant="stop",
                    interactive=False,
                )

                # Status
                opt_status = gr.Textbox(
                    label="Optimization Status",
                    value="Ready to optimize",
                    interactive=False,
                    lines=3,
                )

        with gr.Column(scale=2):
            # Right column: Results and monitoring
            gr.Markdown("## 📊 Optimization Progress")

            with gr.Group():
                # Progress visualization
                opt_progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Optimization Progress (%)",
                    interactive=False,
                )

                opt_current_trial = gr.Textbox(
                    label="Current Trial",
                    value="0 / 0",
                    interactive=False,
                )

                opt_best_score = gr.Textbox(
                    label="Best Score Found",
                    value="No results yet",
                    interactive=False,
                )

                # Optimization history plot
                opt_history_plot = gr.LinePlot(
                    label="Optimization History",
                    x="Trial",
                    y="Score",
                    title="Quality Score vs Trial Number",
                )

            # Results section
            gr.Markdown("### 📋 Optimization Results")

            with gr.Tabs():
                with gr.TabItem("🏆 Best Parameters"):
                    opt_best_params = gr.JSON(
                        label="Best Hyperparameters",
                        value={},
                    )

                    # Download button for best config
                    download_best_config = gr.File(
                        label="Download Best Config",
                        file_count="single",
                        visible=False,
                    )

                    download_btn = gr.Button(
                        "📥 Download Best Config",
                        variant="secondary",
                    )

                with gr.TabItem("📊 Trial History"):
                    opt_trial_history = gr.Dataframe(
                        headers=["Trial", "Score", "Rank", "Alpha", "LR", "Batch Size", "Grad Acc"],
                        value=[],
                        datatype=[
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                        ],
                        label="Trial Results",
                    )

                    # Trial analysis
                    with gr.Accordion("📈 Trial Analysis", open=False):
                        gr.Markdown(
                            "**Score**: Quality metric (0-1). **Rank/Alpha**: Model capacity. **LR**: Learning rate. **Batch Size**: Images per step. Focus on highest-scoring trials."
                        )

                with gr.TabItem("📈 Parameter Importance"):
                    opt_param_importance = gr.Plot(
                        label="Parameter Importance",
                    )

                    importance_explanation = gr.Markdown(
                        "This chart shows which parameters had the most impact on results. Focus tuning efforts on the highest bars."
                    )

                with gr.TabItem("💡 Recommendations"):
                    opt_recommendations = gr.Markdown(
                        value="No optimization completed yet. Run optimization to see recommendations.",
                    )

                with gr.TabItem("📊 Trial History"):
                    opt_trial_history = gr.Dataframe(
                        headers=["Trial", "Score", "Rank", "Alpha", "LR", "Batch Size", "Grad Acc"],
                        value=[],
                        datatype=[
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                            "number",
                        ],
                        label="Trial Results",
                    )

                with gr.TabItem("📈 Parameter Importance"):
                    opt_param_importance = gr.Plot(
                        label="Parameter Importance",
                    )

    # State variables
    opt_active = gr.State(False)
    opt_results = gr.State({})
    opt_dataset_path = gr.State(None)
    opt_config_file = gr.State(None)

    # Preset handlers
    def apply_quick_preset():
        return {
            opt_trials: 20,
            opt_max_steps: 300,
            opt_timeout: 8.0,
        }

    def apply_standard_preset():
        return {
            opt_trials: 50,
            opt_max_steps: 500,
            opt_timeout: None,
        }

    def apply_max_preset():
        return {
            opt_trials: 100,
            opt_max_steps: 500,
            opt_timeout: None,
        }

    quick_preset.click(
        fn=apply_quick_preset,
        outputs=[opt_trials, opt_max_steps, opt_timeout],
    )

    standard_preset.click(
        fn=apply_standard_preset,
        outputs=[opt_trials, opt_max_steps, opt_timeout],
    )

    max_preset.click(
        fn=apply_max_preset,
        outputs=[opt_trials, opt_max_steps, opt_timeout],
    )

    # Event handlers
    def opt_update_dataset_visibility(source):
        if source == "Upload ZIP":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    opt_dataset_source.change(
        fn=opt_update_dataset_visibility,
        inputs=[opt_dataset_source],
        outputs=[opt_dataset_upload, opt_dataset_path],
    )

    # Dataset upload handler
    def opt_handle_dataset_upload(file_obj):
        if file_obj:
            from .training_tab import handle_dataset_upload

            status, path = handle_dataset_upload(app, file_obj)
            return status, path
        return "No file uploaded", None

    opt_dataset_upload.change(
        fn=opt_handle_dataset_upload,
        inputs=[opt_dataset_upload],
        outputs=[opt_dataset_status, opt_dataset_path],
    )

    # Dataset path handler
    def opt_handle_dataset_path(path):
        if path and path.strip():
            from .training_tab import handle_dataset_path

            status, dataset_path = handle_dataset_path(app, path.strip())
            return status
        return "No dataset path provided"

    opt_dataset_path.change(
        fn=opt_handle_dataset_path,
        inputs=[opt_dataset_path],
        outputs=[opt_dataset_status],
    )

    # Start optimization handler
    def opt_start_optimization_handler(
        trials,
        max_steps,
        timeout,
        study_name,
        output_dir,
        opt_active,
        opt_results,
        opt_dataset_path,
    ):
        if opt_active:
            return "Optimization is already running", opt_active, opt_results

        if not opt_dataset_path:
            return (
                "❌ No dataset selected. Please upload or specify a dataset path.",
                opt_active,
                opt_results,
            )

        try:
            # Check if Optuna is available
            import optuna
        except ImportError:
            return (
                "❌ Optuna is required for optimization. Install with: pip install optuna",
                opt_active,
                opt_results,
            )

        # Validate output directory
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"❌ Cannot create output directory: {e}", opt_active, opt_results

        # Start optimization in background
        def optimization_thread():
            try:
                from flux2_lora.optimization import create_optimizer

                app.update_training_state(
                    "opt_status", f"🚀 Starting optimization with {trials} trials..."
                )

                # Create optimizer
                optimizer = create_optimizer(
                    n_trials=int(trials),
                    dataset_path=opt_dataset_path,
                    output_dir=output_dir,
                    timeout_hours=timeout if timeout and timeout > 0 else None,
                    max_steps=int(max_steps),
                )

                # Progress callback
                def progress_callback(trial_number, total_trials, score=None, params=None):
                    progress = (trial_number / total_trials) * 100
                    app.update_training_state("opt_progress", progress)
                    app.update_training_state(
                        "opt_current_trial", f"{trial_number} / {total_trials}"
                    )

                    if score is not None:
                        app.update_training_state("opt_best_score", ".4f")

                    # Update trial history (simplified for UI)
                    if params:
                        app.update_training_state("opt_latest_params", params)

                    # Update status with progress
                    app.update_training_state(
                        "opt_status",
                        f"🔄 Running trial {trial_number}/{total_trials} | "
                        f"Progress: {progress:.1f}% | "
                        f"Best score: {app.get_training_state('opt_best_score', 'N/A')}",
                    )

                # Run optimization (this will be implemented with progress callbacks)
                results = optimizer.optimize(dataset_path=opt_dataset_path, study_name=study_name)

                # Generate recommendations
                recommendations = opt_generate_recommendations(results, trials)

                app.update_training_state("opt_results", results)
                app.update_training_state("opt_recommendations", recommendations)
                app.update_training_state(
                    "opt_status",
                    f"✅ Optimization completed! Best score: {results['best_score']:.4f}",
                )

                # Set config file path for download
                config_path = Path(output_dir) / "best_config.yaml"
                if config_path.exists():
                    app.update_training_state("opt_config_file", str(config_path))

            except Exception as e:
                app.update_training_state("opt_status", f"❌ Optimization failed: {str(e)}")
            finally:
                app.update_training_state("opt_active", False)

        # Start thread
        thread = threading.Thread(target=optimization_thread, daemon=True)
        thread.start()

        return (
            f"🚀 Optimization started with {trials} trials! This may take several hours.",
            True,
            {},
        )

    opt_start_btn.click(
        fn=opt_start_optimization_handler,
        inputs=[
            opt_trials,
            opt_max_steps,
            opt_timeout,
            opt_study_name,
            opt_output_dir,
            opt_active,
            opt_results,
            opt_dataset_path,
        ],
        outputs=[opt_status, opt_active, opt_results],
    )

    # Stop optimization handler
    def opt_stop_optimization_handler(opt_active):
        if not opt_active:
            return "No optimization is currently running", opt_active

        # Set stop flag (implement in optimizer)
        app.update_training_state("opt_should_stop", True)
        return "⏹️ Stopping optimization...", opt_active

    opt_stop_btn.click(
        fn=opt_stop_optimization_handler,
        inputs=[opt_active],
        outputs=[opt_status, opt_active],
    )

    # Download handler
    def opt_download_config(opt_config_file):
        if opt_config_file and Path(opt_config_file).exists():
            return opt_config_file
        return None

    download_btn.click(
        fn=opt_download_config,
        inputs=[opt_config_file],
        outputs=[download_best_config],
    )

    # Update UI periodically
    def opt_update_ui():
        opt_active = app.get_training_state("opt_active", False)
        progress = app.get_training_state("opt_progress", 0.0)
        current_trial = app.get_training_state("opt_current_trial", "0 / 0")
        best_score = app.get_training_state("opt_best_score", "No results yet")
        status = app.get_training_state("opt_status", "Ready to optimize")
        results = app.get_training_state("opt_results", {})
        recommendations = app.get_training_state(
            "opt_recommendations", "No optimization completed yet."
        )
        config_file = app.get_training_state("opt_config_file")

        # Prepare history data for plot
        import pandas as pd

        history_data = []
        if results and "trials_data" in results:
            for trial in results["trials_data"][:20]:
                if trial.get("state") == "COMPLETE":
                    history_data.append(
                        {
                            "Trial": trial.get("number", 0),
                            "Score": trial.get("value", 0),
                        }
                    )

        # Convert to DataFrame for LinePlot
        if history_data:
            history_df = pd.DataFrame(history_data)
        else:
            history_df = pd.DataFrame(columns=["Trial", "Score"])

        return (
            progress,
            current_trial,
            best_score,
            status,
            results.get("best_params", {}),
            history_df,
            recommendations,
            config_file if config_file and Path(config_file).exists() else None,
            gr.update(interactive=not opt_active),
            gr.update(interactive=opt_active),
            gr.update(visible=bool(config_file and Path(config_file).exists())),
        )

    # Set up periodic UI updates
    opt_timer = gr.Timer(2.0)
    opt_timer.tick(
        fn=opt_update_ui,
        outputs=[
            opt_progress_bar,
            opt_current_trial,
            opt_best_score,
            opt_status,
            opt_best_params,
            opt_trial_history,
            opt_recommendations,
            opt_config_file,
            opt_start_btn,
            opt_stop_btn,
            download_best_config,
        ],
    )

"""
Batch Operations tab for the Gradio interface.

Provides productivity features including batch training, bulk evaluation,
templates, and experiment management.
"""

import os
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List, Optional

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp

from .help_utils import help_system


def create_batch_tab(app: "LoRATrainingApp"):
    """
    Create the batch operations tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Batch Operations Help & Tips", open=False):
        gr.Markdown("""
        ## 🚀 Batch Operations

        **Batch processing** allows you to run multiple training jobs, evaluations, and experiments
        simultaneously, dramatically improving your productivity when working with multiple datasets
        or testing different configurations.

        ### Key Features
        - **Batch Training**: Queue multiple LoRA trainings to run sequentially
        - **Bulk Evaluation**: Test multiple checkpoints with the same prompts
        - **Experiment Tracking**: Compare results across different training runs
        - **Templates**: Save and reuse successful configurations
        - **Export Results**: Generate reports and export data for analysis

        ### When to Use Batch Operations
        - **Multiple datasets**: Training LoRAs for different subjects/styles
        - **Hyperparameter testing**: Comparing different training settings
        - **Quality validation**: Testing checkpoints across various scenarios
        - **Experimentation**: Systematic testing of different approaches
        """)

    with gr.Tabs():
        # Batch Training Tab
        with gr.TabItem("🏭 Batch Training"):
            gr.Markdown("### Queue Multiple Training Jobs")

            with gr.Row():
                with gr.Column(scale=1):
                    # Job creation
                    gr.Markdown("#### Create Training Job")

                    job_name = gr.Textbox(
                        label="Job Name",
                        placeholder="e.g., character_lora_v1",
                    )

                    # Template selection
                    available_templates = gr.Dropdown(
                        choices=list(app.templates.keys())
                        if app.templates
                        else ["No templates available"],
                        label="Use Template (Optional)",
                        value="No templates available" if not app.templates else None,
                    )

                    # Quick job setup
                    with gr.Accordion("⚡ Quick Setup", open=True):
                        dataset_path_batch = gr.Textbox(
                            label="Dataset Path",
                            placeholder="/path/to/dataset",
                        )

                        preset_batch = gr.Dropdown(
                            choices=["Character", "Style", "Concept"],
                            label="Preset",
                            value="Character",
                        )

                        rank_batch = gr.Slider(
                            minimum=4,
                            maximum=128,
                            value=16,
                            step=4,
                            label="LoRA Rank",
                        )

                        max_steps_batch = gr.Number(
                            value=1000,
                            label="Training Steps",
                            minimum=100,
                            maximum=10000,
                        )

                    create_job_btn = gr.Button("➕ Create Job", variant="secondary")

                    job_status = gr.HTML(value="", elem_id="batch-job-status")

                with gr.Column(scale=1):
                    # Job queue management
                    gr.Markdown("#### Job Queue")

                    job_queue_display = gr.Dataframe(
                        headers=["Job Name", "Type", "Status", "Progress", "Est. Time"],
                        value=[],
                        label="Queued Jobs",
                        elem_id="job-queue-table",
                    )

                    with gr.Row():
                        start_batch_btn = gr.Button(
                            "🚀 Start Batch",
                            variant="primary",
                            size="lg",
                        )

                        clear_queue_btn = gr.Button("🗑️ Clear Queue", variant="secondary")

                    batch_progress = gr.HTML(
                        value="<div style='text-align: center; padding: 20px; color: #666;'>No batch running</div>",
                        elem_id="batch-progress-display",
                    )

        # Templates Tab
        with gr.TabItem("📋 Templates"):
            gr.Markdown("### Configuration Templates")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Save Current Config as Template")

                    template_name = gr.Textbox(
                        label="Template Name",
                        placeholder="e.g., character_training_v1",
                    )

                    template_description = gr.Textbox(
                        label="Description",
                        lines=3,
                        placeholder="Describe what this template is good for...",
                    )

                    save_template_btn = gr.Button(
                        "💾 Save Template",
                        variant="secondary",
                    )

                    template_status = gr.HTML(value="")

                with gr.Column(scale=1):
                    gr.Markdown("#### Available Templates")

                    templates_list = gr.Dropdown(
                        choices=list(app.templates.keys()) if app.templates else ["No templates"],
                        label="Select Template",
                    )

                    load_template_btn = gr.Button(
                        "📂 Load Template",
                        variant="secondary",
                    )

                    delete_template_btn = gr.Button(
                        "🗑️ Delete Template",
                        variant="secondary",
                    )

                    template_details = gr.JSON(value={}, label="Template Details")

        # Experiments Tab
        with gr.TabItem("🧪 Experiments"):
            gr.Markdown("### Experiment Management")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Create Experiment")

                    exp_name = gr.Textbox(
                        label="Experiment Name",
                        placeholder="e.g., rank_comparison_test",
                    )

                    exp_description = gr.Textbox(
                        label="Description",
                        lines=3,
                        placeholder="Compare different LoRA ranks on character training...",
                    )

                    # Select jobs for experiment
                    available_jobs = gr.CheckboxGroup(
                        choices=[],  # Will be populated dynamically
                        label="Include Jobs",
                    )

                    create_exp_btn = gr.Button(
                        "🧪 Create Experiment",
                        variant="secondary",
                    )

                    exp_status = gr.HTML(value="")

                with gr.Column(scale=1):
                    gr.Markdown("#### Experiment Results")

                    experiments_list = gr.Dropdown(
                        choices=list(app.experiments.keys())
                        if app.experiments
                        else ["No experiments"],
                        label="Select Experiment",
                    )

                    exp_results = gr.JSON(value={}, label="Experiment Comparison")

                    export_exp_btn = gr.Button(
                        "📊 Export Results",
                        variant="secondary",
                    )

        # Export Tab
        with gr.TabItem("📤 Export"):
            gr.Markdown("### Export Results & Data")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Export Options")

                    export_jobs = gr.CheckboxGroup(
                        choices=[],  # Will be populated with completed jobs
                        label="Select Jobs to Export",
                    )

                    export_format = gr.Radio(
                        choices=["json", "csv", "html"],
                        value="json",
                        label="Export Format",
                    )

                    export_btn = gr.Button(
                        "📤 Export Selected",
                        variant="primary",
                    )

                    export_status = gr.HTML(value="")

                with gr.Column(scale=1):
                    gr.Markdown("#### Export History")

                    export_history = gr.Dataframe(
                        headers=["Timestamp", "Jobs Exported", "Format", "File Path"],
                        value=[],
                        label="Previous Exports",
                    )

                    # Summary stats
                    export_summary = gr.JSON(value={}, label="Export Summary")

    # Event handlers
    def create_batch_job_handler(app, name, template, dataset_path, preset, rank, max_steps):
        """Handle batch job creation."""
        if not name or not dataset_path:
            return "<div style='color: #f44336;'>❌ Please provide job name and dataset path</div>"

        # Load from template if selected
        config = {}
        if template and template != "No templates available":
            config = app.load_template(template)

        # Override with current settings
        config.update(
            {
                "dataset_path": dataset_path,
                "preset": preset.lower(),
                "rank": int(rank),
                "alpha": int(rank),  # Usually same as rank
                "learning_rate": 1e-4,
                "max_steps": int(max_steps),
                "batch_size": 4,
            }
        )

        job_id = app.create_batch_job(name, config, "training")
        return f"<div style='color: #4caf50;'>✅ Job '{name}' created with ID: {job_id}</div>"

    def start_batch_handler(app):
        """Handle batch execution start."""
        if not app.job_queue:
            return "<div style='color: #ff9800;'>⚠️ No jobs in queue</div>"

        app.execute_batch_jobs()
        return f"<div style='color: #2196f3;'>🚀 Started batch execution of {len(app.job_queue)} jobs</div>"

    def update_job_queue_display(app):
        """Update the job queue display."""
        queue_data = []
        for job in app.batch_jobs:
            queue_data.append(
                [
                    job["name"],
                    job["type"],
                    job["status"].title(),
                    f"{job['progress']:.1f}%" if job["progress"] > 0 else "0%",
                    f"{job['estimated_time']:.0f}s" if job.get("estimated_time") else "Unknown",
                ]
            )
        return queue_data

    def update_batch_progress(app):
        """Update batch progress display."""
        if not app.job_queue:
            return "<div style='text-align: center; padding: 20px; color: #666;'>No batch running</div>"

        running_jobs = [j for j in app.job_queue if j["status"] == "running"]
        if not running_jobs:
            return (
                "<div style='text-align: center; padding: 20px; color: #666;'>Batch completed</div>"
            )

        # Show progress for running jobs
        progress_html = "<div style='padding: 10px;'>"
        for job in running_jobs[:3]:  # Show first 3 running jobs
            progress_html += f"""
            <div style='margin: 8px 0;'>
                <strong>{job["name"]}</strong>: {job["progress"]:.1f}% complete
                <div style='width: 100%; height: 6px; background: #e0e0e0; border-radius: 3px; margin: 4px 0;'>
                    <div style='width: {job["progress"]}%; height: 100%; background: #2196f3; border-radius: 3px;'></div>
                </div>
            </div>
            """
        progress_html += "</div>"

        return progress_html

    def save_template_handler(app, name, description):
        """Handle template saving."""
        if not name:
            return "<div style='color: #f44336;'>❌ Please provide a template name</div>"

        # Get current training config (simplified)
        config = app.get_smart_defaults("training")
        app.save_template(name, config, description)

        return f"<div style='color: #4caf50;'>✅ Template '{name}' saved successfully</div>"

    save_template_btn.click(
        fn=lambda name, desc: save_template_handler(app, name, desc),
        inputs=[template_name, template_description],
        outputs=[template_status],
    )

    def load_template_handler(app, template_name):
        """Handle template loading."""
        if not template_name or template_name == "No templates":
            return {}

        template_data = app.templates.get(template_name, {})
        return template_data

    load_template_btn.click(
        fn=lambda name: load_template_handler(app, name),
        inputs=[templates_list],
        outputs=[template_details],
    )

    def create_experiment_handler(app, name, description, selected_jobs):
        """Handle experiment creation."""
        if not name or not selected_jobs:
            return "<div style='color: #f44336;'>❌ Please provide experiment name and select jobs</div>"

        # Find job IDs from names
        job_ids = []
        for job_name in selected_jobs:
            job = next((j for j in app.job_history if j["name"] == job_name), None)
            if job:
                job_ids.append(job["id"])

        if not job_ids:
            return "<div style='color: #f44336;'>❌ No valid jobs found</div>"

        exp_id = app.create_experiment(name, job_ids, description)
        return f"<div style='color: #4caf50;'>✅ Experiment '{name}' created with {len(job_ids)} jobs</div>"

    create_exp_btn.click(
        fn=lambda name, desc, jobs: create_experiment_handler(app, name, desc, jobs),
        inputs=[exp_name, exp_description, available_jobs],
        outputs=[exp_status],
    )

    def load_experiment_results(app, exp_name):
        """Load experiment comparison results."""
        if not exp_name or exp_name == "No experiments":
            return {}

        # Get experiment ID from name
        exp_id = None
        for eid, exp_data in app.experiments.items():
            if exp_data.get("name") == exp_name:
                exp_id = eid
                break

        if not exp_id:
            return {"error": f"Experiment '{exp_name}' not found"}

        return app.get_experiment_comparison_data(exp_id)

    def export_jobs_handler(app, selected_jobs, export_format):
        """Handle job export."""
        if not selected_jobs:
            return "<div style='color: #f44336;'>❌ Please select jobs to export</div>"

        # Find job IDs from names
        job_ids = []
        for job_name in selected_jobs:
            job = next((j for j in app.job_history if j["name"] == job_name), None)
            if job:
                job_ids.append(job["id"])

        if not job_ids:
            return "<div style='color: #f44336;'>❌ No valid jobs found</div>"

        export_path = app.export_results(job_ids, export_format)
        return f"<div style='color: #4caf50;'>✅ Results exported to: {export_path}</div>"

    experiments_list.change(
        fn=lambda exp: load_experiment_results(app, exp),
        inputs=[experiments_list],
        outputs=[exp_results],
    )

    export_btn.click(
        fn=lambda jobs, fmt: export_jobs_handler(app, jobs, fmt),
        inputs=[export_jobs, export_format],
        outputs=[export_status],
    )

    def update_available_jobs(app):
        """Update available jobs for experiments."""
        completed_jobs = [j["name"] for j in app.job_history if j["status"] == "completed"]
        return completed_jobs, completed_jobs

    # Update available jobs for experiments
    available_jobs_update = gr.Timer(5.0)  # Update every 5 seconds
    available_jobs_update.tick(
        fn=lambda: update_available_jobs(app), outputs=[available_jobs, export_jobs]
    )

    # Update job queue display periodically
    queue_update_timer = gr.Timer(2.0)  # Update every 2 seconds
    queue_update_timer.tick(
        fn=lambda: (update_job_queue_display(app), update_batch_progress(app)),
        outputs=[job_queue_display, batch_progress],
    )

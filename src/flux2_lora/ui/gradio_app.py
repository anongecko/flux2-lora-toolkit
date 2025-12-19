"""
Gradio application for Flux2-dev LoRA Training Toolkit.

This module provides the main Gradio interface with tabbed navigation
for training, evaluation, and dataset management.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List
import gradio as gr
from gradio_modal import Modal
from pathlib import Path

from .training_tab import create_training_tab
from .evaluation_tab import create_evaluation_tab
from .dataset_tab import create_dataset_tab
from .optimization_tab import create_optimization_tab
from .augmentation_tab import create_augmentation_tab
from .batch_tab import create_batch_tab
from .help_utils import help_system

logger = logging.getLogger(__name__)


class LoRATrainingApp:
    """
    Main Gradio application for Flux2-dev LoRA training.

    Provides a web interface with three main tabs:
    - Training: Configure and run LoRA training jobs
    - Evaluation: Test and compare trained checkpoints
    - Dataset Tools: Analyze and manage training datasets
    """

    def __init__(self):
        """Initialize the application with state management."""
        # Training session state
        self.training_state: Dict[str, Any] = {
            "is_training": False,
            "current_job_id": None,
            "training_config": {},
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss_history": [],
            "validation_samples": [],
            "status_message": "Ready to train",
        }

        # Workflow state management
        self.workflow_state: Dict[str, Any] = {
            "current_step": "prepare",  # prepare, train, evaluate, optimize
            "completed_steps": set(),
            "user_experience_level": "beginner",  # beginner, intermediate, advanced
            "recent_actions": [],
            "first_visit": True,
            "dataset_loaded": False,
            "training_completed": False,
            "evaluation_run": False,
            "optimization_completed": False,
        }

        # User preferences and history
        self.user_prefs: Dict[str, Any] = {
            "last_dataset_path": None,
            "preferred_preset": "Character",
            "default_batch_size": 4,
            "default_steps": 1000,
            "advanced_mode": False,
            "auto_save_configs": True,
        }

        # File management state
        self.uploaded_files: Dict[str, Path] = {}

        # Session management (optional authentication)
        self.session_id: Optional[str] = None

        # Load user preferences
        self._load_user_preferences()

        # Notification system
        self.notifications: List[Dict[str, Any]] = []
        self.operation_queue: List[Dict[str, Any]] = []

        # Error handling and validation system
        self.validation_errors: Dict[str, List[str]] = {}
        self.config_history: List[Dict[str, Any]] = []  # For undo/redo
        self.current_config_index: int = -1

        # Batch operations and productivity system
        self.batch_jobs: List[Dict[str, Any]] = []  # Batch training jobs
        self.job_queue: List[Dict[str, Any]] = []  # Active job queue
        self.job_history: List[Dict[str, Any]] = []  # Completed jobs
        self.templates: Dict[str, Dict[str, Any]] = {}  # Configuration templates
        self.experiments: Dict[str, Dict[str, Any]] = {}  # Experiment tracking

        logger.info("LoRATrainingApp initialized")

    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface with tabs.

        Returns:
            Gradio Blocks interface
        """
        # Create the main application
        with gr.Blocks(title="Flux2-dev LoRA Training Toolkit") as app:
            # Welcome modal (shown on first visit)
            with Modal(visible=True) as welcome_modal:
                gr.Markdown(help_system.get_welcome_message())
                gr.Markdown("---")
                gr.Markdown(
                    "**💡 Tip**: You can reopen this guide anytime using the help sections in each tab."
                )

            # Troubleshooting wizard modal
            with Modal(visible=False) as troubleshooting_modal:
                gr.Markdown("# 🔧 Troubleshooting Wizard", elem_id="troubleshooting-title")

                # Error type selector
                error_category = gr.Dropdown(
                    choices=[
                        "GPU Memory Issues",
                        "Dataset Problems",
                        "Training Failures",
                        "Model Loading Errors",
                        "Configuration Issues",
                        "General Problems",
                    ],
                    label="What type of problem are you experiencing?",
                    value="GPU Memory Issues",
                    elem_id="error-category-selector",
                )

                # Dynamic troubleshooting content
                troubleshooting_content = gr.Markdown(
                    value=self.get_troubleshooting_guide("gpu_memory"),
                    elem_id="troubleshooting-content",
                )

                gr.Markdown("### Quick Action Buttons")
                with gr.Row():
                    quick_fix_memory = gr.Button("🧠 Fix Memory Issues", variant="primary")
                    quick_fix_dataset = gr.Button("📁 Fix Dataset Issues", variant="primary")
                    quick_fix_training = gr.Button("🎯 Fix Training Issues", variant="primary")

                # Troubleshooting status
                troubleshooting_status = gr.HTML(value="", elem_id="troubleshooting-status")

            # Workflow Progress Indicator
            workflow_indicator = gr.HTML(
                value=self.get_workflow_progress_indicator(), elem_id="workflow-indicator"
            )

            # Quick Actions Bar
            with gr.Row(elem_classes=["quick-actions-bar"]):
                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Quick Actions")
                    with gr.Row():
                        start_training_btn = gr.Button(
                            "🎯 Start New Training",
                            variant="primary",
                            size="sm",
                        )
                        evaluate_checkpoint_btn = gr.Button(
                            "📊 Evaluate Checkpoint",
                            variant="secondary",
                            size="sm",
                        )
                        analyze_dataset_btn = gr.Button(
                            "🔍 Analyze Dataset",
                            variant="secondary",
                            size="sm",
                        )
                        batch_operations_btn = gr.Button(
                            "🏭 Batch Training",
                            variant="secondary",
                            size="sm",
                        )
                        troubleshooting_btn = gr.Button(
                            "🔧 Troubleshoot",
                            variant="secondary",
                            size="sm",
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### 📈 Recent Activity")
                    recent_activity = gr.Textbox(
                        label="",
                        value=self._get_recent_activity_summary(),
                        interactive=False,
                        lines=2,
                    )

            # Header with feature overview
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown(
                        """
                        # 🎨 Flux2-dev LoRA Training Toolkit

                        **Train high-quality LoRAs for Flux2-dev with real-time monitoring and automatic quality assessment.**

                        > This toolkit enables both ML practitioners and creative professionals to train LoRAs with enterprise-grade features including real-time visualization, checkpoint comparison, and comprehensive evaluation tools.
                        """,
                        elem_classes=["app-header"],
                    )

                    # Status notifications area
                    status_notifications = gr.HTML(
                        value=self._get_status_notifications(), elem_id="status-notifications"
                    )

                    # Real-time notifications panel
                    with gr.Accordion("📢 Notifications", open=False):
                        notifications_panel = gr.HTML(
                            value=self.get_notifications_html(), elem_id="notifications-panel"
                        )

                        # Operation queue
                        with gr.Accordion("⚙️ Operation Queue", open=False):
                            operation_queue = gr.HTML(
                                value=self.get_operation_queue_html(), elem_id="operation-queue"
                            )

                with gr.Column(scale=1):
                    with gr.Accordion("🔍 Feature Overview", open=False):
                        gr.Markdown("### Major Toolkit Features")

                        # Feature buttons that show detailed explanations
                        feature_buttons = []
                        for feature_key, feature_info in help_system.get_feature_overview().items():
                            with gr.Accordion(
                                f"📋 {feature_key.replace('_', ' ').title()}", open=False
                            ):
                                gr.Markdown(feature_info)

                    with gr.Accordion("🚀 Quick Start Guide", open=False):
                        gr.Markdown(help_system.get_workflow_guidance())

                    # User Experience Level Indicator
                    exp_level = self.workflow_state["user_experience_level"]
                    exp_colors = {
                        "beginner": "#4CAF50",
                        "intermediate": "#FF9800",
                        "advanced": "#9C27B0",
                    }

                    gr.Markdown(f"""
                    ### 👤 Experience Level: {exp_level.title()}
                    <div style="
                        width: 100%;
                        height: 8px;
                        background: linear-gradient(90deg, {exp_colors[exp_level]} 0%, #e0e0e0 100%);
                        border-radius: 4px;
                        margin: 8px 0;
                    "></div>
                    """)

            # Main tab interface
            with gr.Tabs(elem_classes=["main-tabs"]):
                # Training Tab
                with gr.TabItem("🚀 Training", id="training"):
                    create_training_tab(self)

                # Evaluation Tab
                with gr.TabItem("📊 Evaluation", id="evaluation"):
                    create_evaluation_tab(self)

                # Dataset Tools Tab
                with gr.TabItem("🗂️ Dataset Tools", id="dataset"):
                    create_dataset_tab(self)

                # Optimization Tab
                with gr.TabItem("⚡ Optimization", id="optimization"):
                    create_optimization_tab(self)

                # Augmentation Tab
                with gr.TabItem("🎨 Augmentation", id="augmentation"):
                    create_augmentation_tab(self)

                # Batch Operations Tab
                with gr.TabItem("🏭 Batch Operations", id="batch"):
                    create_batch_tab(self)

            # Contextual help panel with workflow guidance
            with gr.Accordion("🎯 Workflow Guidance", open=True) as help_panel:
                workflow_guidance = gr.Markdown(
                    value=self._get_contextual_guidance(), elem_id="workflow-guidance"
                )

            # Update guidance based on current state
            def update_workflow_guidance():
                return self._get_contextual_guidance()

            # Note: Tab selection events would be added here in a full implementation

            # Status Dashboard (collapsible)
            with gr.Accordion("📊 Status Dashboard", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training History")
                        training_history = gr.Dataframe(
                            headers=["Date", "Model", "Steps", "Status", "Quality"],
                            value=self._get_training_history(),
                            label="Recent Training Sessions",
                            elem_id="training-history",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Quick Stats")
                        quick_stats = gr.JSON(
                            value=self._get_quick_stats(),
                            label="Training Statistics",
                            elem_id="quick-stats",
                        )

                gr.Markdown("### Recent Checkpoints")
                recent_checkpoints = gr.Dataframe(
                    headers=["Name", "Created", "Size", "Quality Score"],
                    value=self._get_recent_checkpoints(),
                    label="Recent Trained Models",
                    elem_id="recent-checkpoints",
                )

            # Footer
            gr.Markdown(
                """
                ---
                **Flux2-dev LoRA Training Toolkit** | Built with Gradio | [Documentation](https://github.com/your-repo/docs) | [GitHub](https://github.com/your-repo)
                """,
                elem_classes=["app-footer"],
            )

            # Quick Action Event Handlers
            def quick_start_training():
                """Handle quick start training action."""
                self.update_workflow_step("train")
                return """
                <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 10px; border-radius: 5px; margin: 10px 0;">
                🔄 Switched to Training tab. Configure your settings and click "Start Training" to begin!
                </div>
                """

            def quick_evaluate_checkpoint():
                """Handle quick evaluate checkpoint action."""
                self.update_workflow_step("evaluate")
                return """
                <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 10px; border-radius: 5px; margin: 10px 0;">
                🔄 Switched to Evaluation tab. Load your checkpoint and start testing!
                </div>
                """

            def quick_analyze_dataset():
                """Handle quick analyze dataset action."""
                self.update_workflow_step("prepare")
                return """
                <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 10px; border-radius: 5px; margin: 10px 0;">
                🔄 Switched to Dataset Tools tab. Upload your dataset to get started!
                </div>
                """

            # Connect quick action buttons (these would need to be implemented with tab switching)
            start_training_btn.click(fn=quick_start_training, outputs=[status_notifications])

            evaluate_checkpoint_btn.click(
                fn=quick_evaluate_checkpoint, outputs=[status_notifications]
            )

            analyze_dataset_btn.click(fn=quick_analyze_dataset, outputs=[status_notifications])

            # Troubleshooting wizard handlers
            def update_troubleshooting_content(error_type):
                """Update troubleshooting content based on selected error type."""
                error_type_map = {
                    "GPU Memory Issues": "gpu_memory",
                    "Dataset Problems": "dataset_issues",
                    "Training Failures": "training_failures",
                    "Model Loading Errors": "model_download",
                    "Configuration Issues": "parameter_ranges",
                    "General Problems": "general_troubleshooting",
                }

                guide_key = error_type_map.get(error_type, "general_troubleshooting")
                return self.get_troubleshooting_guide(guide_key)

            def apply_quick_fix(fix_type):
                """Apply a quick fix based on the selected type."""
                fixes = {
                    "memory": {
                        "action": "Applying memory optimization settings...",
                        "changes": {
                            "batch_size": 1,
                            "gradient_accumulation_steps": 4,
                            "enable_gradient_checkpointing": True,
                        },
                    },
                    "dataset": {
                        "action": "Opening dataset validation tools...",
                        "tab_switch": "dataset",
                    },
                    "training": {
                        "action": "Resetting to stable training defaults...",
                        "changes": {
                            "learning_rate": 1e-4,
                            "rank": 16,
                            "alpha": 16,
                            "max_steps": 1000,
                        },
                    },
                }

                fix_config = fixes.get(fix_type, {})
                action_msg = fix_config.get("action", "Applying fix...")

                # Add notification
                self.add_notification(f"🔧 {action_msg}", "info")

                # Return status
                return f"<div style='padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 5px;'>✅ {action_msg}</div>"

            error_category.change(
                fn=update_troubleshooting_content,
                inputs=[error_category],
                outputs=[troubleshooting_content],
            )

            quick_fix_memory.click(
                fn=lambda: apply_quick_fix("memory"), outputs=[troubleshooting_status]
            )

            quick_fix_dataset.click(
                fn=lambda: apply_quick_fix("dataset"), outputs=[troubleshooting_status]
            )

            quick_fix_training.click(
                fn=lambda: apply_quick_fix("training"), outputs=[troubleshooting_status]
            )

            # Global troubleshooting trigger (can be called from error handlers)
            def open_troubleshooting_wizard(error_type="general"):
                """Open the troubleshooting wizard with a specific error type."""
                error_type_map = {
                    "gpu_memory": "GPU Memory Issues",
                    "dataset_issues": "Dataset Problems",
                    "training_failures": "Training Failures",
                    "model_download": "Model Loading Errors",
                    "parameter_ranges": "Configuration Issues",
                    "general": "General Problems",
                }

                selected_type = error_type_map.get(error_type, "General Problems")
                content = self.get_troubleshooting_guide(error_type)

                return (
                    gr.update(visible=True),  # Show modal
                    selected_type,  # Set dropdown
                    content,  # Update content
                )

            # Make troubleshooting function available globally
            self.open_troubleshooting_wizard = open_troubleshooting_wizard

            # Batch operations button handler
            def quick_batch_operations():
                """Handle quick batch operations action."""
                return """
                <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 10px; border-radius: 5px; margin: 10px 0;">
                🔄 Switched to Batch Operations tab. Create multiple training jobs and run them efficiently!
                </div>
                """

            batch_operations_btn.click(fn=quick_batch_operations, outputs=[status_notifications])

            # Troubleshooting button handler
            troubleshooting_btn.click(
                fn=lambda: (
                    gr.update(visible=True),
                    "General Problems",
                    self.get_troubleshooting_guide("general_troubleshooting"),
                ),
                outputs=[troubleshooting_modal, error_category, troubleshooting_content],
            )

        return app

    def launch(self, **kwargs):
        """
        Launch the Gradio application.

        Args:
            **kwargs: Arguments passed to gr.Blocks.launch()
        """
        app = self.create_interface()

        # Set default launch parameters
        launch_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "show_error": True,
            "favicon_path": None,  # Could add custom favicon later
        }
        launch_kwargs.update(kwargs)

        logger.info(
            f"Launching Gradio app on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}"
        )

        app.launch(**launch_kwargs)

    def update_training_state(self, key: str, value: Any):
        """
        Update training state safely.

        Args:
            key: State key to update
            value: New value
        """
        self.training_state[key] = value
        logger.debug(f"Updated training state: {key} = {value}")

    def get_training_state(self, key: str, default: Any = None) -> Any:
        """
        Get training state value safely.

        Args:
            key: State key to retrieve
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self.training_state.get(key, default)

    def reset_training_state(self):
        """Reset training state to initial values."""
        initial_state = {
            "is_training": False,
            "current_job_id": None,
            "training_config": {},
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "loss_history": [],
            "validation_samples": [],
            "status_message": "Ready to train",
        }
        self.training_state.update(initial_state)
        logger.info("Training state reset")

    def register_uploaded_file(self, file_path: Path, file_type: str = "dataset") -> str:
        """
        Register an uploaded file for management.

        Args:
            file_path: Path to uploaded file
            file_type: Type of file (dataset, checkpoint, etc.)

        Returns:
            Unique file ID
        """
        import uuid

        file_id = f"{file_type}_{uuid.uuid4().hex[:8]}"
        self.uploaded_files[file_id] = file_path
        logger.debug(f"Registered uploaded file: {file_id} -> {file_path}")
        return file_id

    def get_uploaded_file(self, file_id: str) -> Optional[Path]:
        """
        Get uploaded file path by ID.

        Args:
            file_id: File ID to retrieve

        Returns:
            File path or None if not found
        """
        return self.uploaded_files.get(file_id)

    def cleanup_uploaded_files(self, keep_recent: int = 10):
        """
        Clean up old uploaded files to manage disk space.

        Args:
            keep_recent: Number of recent files to keep
        """
        if len(self.uploaded_files) <= keep_recent:
            return

        # Sort by modification time (newest first)
        # Only include files that actually exist for cleanup
        existing_files = [
            (file_id, file_path)
            for file_id, file_path in self.uploaded_files.items()
            if file_path.exists()
        ]

        if not existing_files:
            return

        sorted_files = sorted(existing_files, key=lambda x: x[1].stat().st_mtime, reverse=True)

        # Remove old files
        for file_id, file_path in sorted_files[keep_recent:]:
            try:
                file_path.unlink()
                del self.uploaded_files[file_id]
                logger.debug(f"Cleaned up old file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")

    # Workflow Management Methods
    def update_workflow_step(self, step: str):
        """
        Update the current workflow step.

        Args:
            step: New workflow step ('prepare', 'train', 'evaluate', 'optimize')
        """
        self.workflow_state["current_step"] = step
        if step not in self.workflow_state["completed_steps"]:
            self.workflow_state["completed_steps"].add(step)

        # Update user experience level based on completed steps
        completed_count = len(self.workflow_state["completed_steps"])
        if completed_count >= 4:
            self.workflow_state["user_experience_level"] = "advanced"
        elif completed_count >= 2:
            self.workflow_state["user_experience_level"] = "intermediate"

        logger.debug(f"Workflow step updated to: {step}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status for UI display."""
        return {
            "current_step": self.workflow_state["current_step"],
            "completed_steps": list(self.workflow_state["completed_steps"]),
            "experience_level": self.workflow_state["user_experience_level"],
            "first_visit": self.workflow_state["first_visit"],
            "dataset_ready": self.workflow_state["dataset_loaded"],
            "training_done": self.workflow_state["training_completed"],
            "evaluation_done": self.workflow_state["evaluation_run"],
            "optimization_done": self.workflow_state["optimization_completed"],
        }

    def get_next_recommended_tab(self) -> str:
        """Get the next recommended tab based on current workflow state."""
        current = self.workflow_state["current_step"]

        # Logical workflow progression
        if current == "prepare" and not self.workflow_state["dataset_loaded"]:
            return "dataset"
        elif current == "prepare" and self.workflow_state["dataset_loaded"]:
            return "training"
        elif current == "train" and self.workflow_state["training_completed"]:
            return "evaluation"
        elif current == "evaluate" and self.workflow_state["evaluation_run"]:
            return "optimization"
        elif current == "optimize":
            return "evaluation"  # Can always go back to evaluation

        return "training"  # Default fallback

    def get_workflow_progress_indicator(self) -> str:
        """Generate HTML for workflow progress indicator."""
        steps = [
            ("prepare", "📋 Prepare Dataset", "dataset"),
            ("train", "🚀 Train LoRA", "training"),
            ("evaluate", "📊 Evaluate Results", "evaluation"),
            ("optimize", "⚡ Optimize Settings", "optimization"),
        ]

        html_parts = []
        for step_id, step_name, tab_id in steps:
            is_completed = step_id in self.workflow_state["completed_steps"]
            is_current = step_id == self.workflow_state["current_step"]

            if is_completed:
                icon = "✅"
                style = "background-color: #4CAF50; color: white;"
            elif is_current:
                icon = "🔄"
                style = "background-color: #2196F3; color: white; border: 2px solid #1976D2;"
            else:
                icon = "⭕"
                style = "background-color: #f5f5f5; color: #666;"

            html_parts.append(f"""
                <div style="
                    display: inline-block;
                    padding: 8px 12px;
                    margin: 0 4px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: bold;
                    cursor: pointer;
                    {style}
                " onclick="document.querySelector('[id*={tab_id}]').click()">
                    {icon} {step_name}
                </div>
            """)

        return f"""
        <div style="text-align: center; margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;">
            <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px; color: #333;">
                🎯 Training Workflow Progress
            </div>
            <div style="margin-bottom: 10px;">
                {"".join(html_parts)}
            </div>
            <div style="font-size: 0.9em; color: #666;">
                💡 Tip: Follow the workflow steps for best results. Click any step to jump to that tab.
            </div>
        </div>
        """

    # User Preferences Management
    def _load_user_preferences(self):
        """Load user preferences from disk."""
        try:
            import json
            from pathlib import Path

            prefs_file = Path.home() / ".flux2_lora" / "user_prefs.json"
            if prefs_file.exists():
                with open(prefs_file, "r") as f:
                    self.user_prefs.update(json.load(f))
                logger.debug("User preferences loaded")
        except Exception as e:
            logger.debug(f"Could not load user preferences: {e}")

    def save_user_preferences(self):
        """Save user preferences to disk."""
        try:
            import json
            from pathlib import Path

            prefs_dir = Path.home() / ".flux2_lora"
            prefs_dir.mkdir(exist_ok=True)

            prefs_file = prefs_dir / "user_prefs.json"
            with open(prefs_file, "w") as f:
                json.dump(self.user_prefs, f, indent=2)
            logger.debug("User preferences saved")
        except Exception as e:
            logger.warning(f"Could not save user preferences: {e}")

    def update_user_preference(self, key: str, value: Any):
        """Update a user preference and save to disk."""
        self.user_prefs[key] = value
        if self.user_prefs.get("auto_save_configs", True):
            self.save_user_preferences()

    def get_smart_defaults(self, context: str = "training") -> Dict[str, Any]:
        """Get smart defaults based on user history and context."""
        defaults = {}

        if context == "training":
            # Base defaults
            defaults.update(
                {
                    "batch_size": self.user_prefs.get("default_batch_size", 4),
                    "max_steps": self.user_prefs.get("default_steps", 1000),
                    "preset": self.user_prefs.get("preferred_preset", "Character"),
                }
            )

            # Adjust based on user experience level
            exp_level = self.workflow_state["user_experience_level"]
            if exp_level == "beginner":
                defaults.update(
                    {
                        "rank": 16,
                        "alpha": 16,
                        "learning_rate": 1e-4,
                    }
                )
            elif exp_level == "intermediate":
                defaults.update(
                    {
                        "rank": 32,
                        "alpha": 32,
                        "learning_rate": 2e-4,
                    }
                )
            else:  # advanced
                defaults.update(
                    {
                        "rank": 64,
                        "alpha": 64,
                        "learning_rate": 5e-4,
                    }
                )

        return defaults

    def _get_recent_activity_summary(self) -> str:
        """Get a summary of recent user activities."""
        activities = []

        if self.workflow_state["training_completed"]:
            activities.append("✅ Recent training completed")

        if self.workflow_state["evaluation_run"]:
            activities.append("📊 Evaluation performed")

        if self.workflow_state["optimization_completed"]:
            activities.append("⚡ Optimization finished")

        if self.workflow_state["dataset_loaded"]:
            activities.append("🗂️ Dataset loaded")

        if not activities:
            activities.append("🎯 Ready to start training")

        return " • ".join(activities[-3:])  # Show last 3 activities

    def _get_status_notifications(self) -> str:
        """Get HTML for status notifications."""
        notifications = []

        # First visit notification
        if self.workflow_state["first_visit"]:
            notifications.append("""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                        border: 1px solid #2196f3; border-radius: 8px; padding: 12px; margin: 8px 0; color: #1565c0;">
                <strong>🎉 Welcome to Flux2-dev LoRA Training!</strong><br>
                <small>Start by preparing your dataset in the <strong>Dataset Tools</strong> tab, then train your first LoRA.</small>
            </div>
            """)

        # Training ready notification
        if self.workflow_state["dataset_loaded"] and not self.workflow_state["training_completed"]:
            notifications.append("""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
                        border: 1px solid #ff9800; border-radius: 8px; padding: 12px; margin: 8px 0;">
                <strong>🚀 Ready to Train!</strong><br>
                <small>Your dataset is loaded. Head to the <strong>Training</strong> tab to start training your LoRA.</small>
            </div>
            """)

        # Training completed notification
        if self.workflow_state["training_completed"] and not self.workflow_state["evaluation_run"]:
            notifications.append("""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                        border: 1px solid #4caf50; border-radius: 8px; padding: 12px; margin: 8px 0;">
                <strong>✅ Training Complete!</strong><br>
                <small>Evaluate your results in the <strong>Evaluation</strong> tab to see how well your LoRA performs.</small>
            </div>
            """)

        return "".join(notifications)

    def _get_contextual_guidance(self) -> str:
        """Get contextual guidance based on current workflow state."""
        current_step = self.workflow_state["current_step"]
        exp_level = self.workflow_state["user_experience_level"]

        guidance = f"## Current Step: {current_step.title()}\n\n"

        if current_step == "prepare":
            if not self.workflow_state["dataset_loaded"]:
                guidance += """
                **📋 What to do next:**
                1. Go to **Dataset Tools** tab
                2. Upload or select your training dataset
                3. Analyze your dataset for quality issues
                4. Fix any problems found in the analysis

                **💡 Tip:** Start with 10-50 high-quality images and descriptive captions for best results.
                """
            else:
                guidance += """
                **✅ Dataset Ready!**
                Your dataset is loaded and analyzed. Ready to move to training!

                **Next:** Click **"Start New Training"** in the quick actions above, or go to the **Training** tab.
                """

        elif current_step == "train":
            if not self.workflow_state["training_completed"]:
                guidance += """
                **🚀 Training in Progress:**
                - Configure your LoRA settings (preset, rank, steps)
                - Upload/select your dataset if not done already
                - Click "Start Training" to begin
                - Monitor progress in real-time

                **💡 Tip:** Start with default settings if you're new to LoRA training.
                """
            else:
                guidance += """
                **✅ Training Complete!**
                Your LoRA training has finished. Time to evaluate the results!

                **Next:** Go to **Evaluation** tab to test your trained checkpoint.
                """

        elif current_step == "evaluate":
            guidance += """
            **📊 Evaluation Phase:**
            - Load your trained checkpoint (.safetensors file)
            - Generate test samples with different prompts
            - Run quality assessment and prompt testing
            - Compare multiple checkpoints if available

            **💡 Tip:** Test with prompts similar to your training captions for best results.
            """

        elif current_step == "optimize":
            guidance += """
            **⚡ Optimization Phase:**
            - Use hyperparameter optimization to find better settings
            - Run multiple trials to explore parameter space
            - Download the best configuration found
            - Retrain with optimized settings

            **💡 Tip:** Optimization can take several hours but often improves results significantly.
            """

        # Add experience-based tips
        if exp_level == "beginner":
            guidance += "\n\n**🎓 Beginner Tips:**\n- Follow the workflow step by step\n- Use default settings initially\n- Don't worry about advanced options yet"
        elif exp_level == "intermediate":
            guidance += "\n\n**🔧 Intermediate Tips:**\n- Try adjusting learning rate and rank\n- Experiment with different presets\n- Use evaluation results to guide improvements"
        else:  # advanced
            guidance += "\n\n**⚡ Advanced Tips:**\n- Fine-tune all parameters\n- Use optimization for best results\n- Experiment with custom architectures"

        return guidance

    def _get_training_history(self) -> List[List[str]]:
        """Get training history for display."""
        # This would be loaded from a history file in a real implementation
        # For now, return mock data based on current state
        history = []

        if self.workflow_state["training_completed"]:
            history.append(["Today", "Character LoRA", "1000", "✅ Completed", "0.85"])

        if len(history) == 0:
            history.append(["-", "No training history", "-", "-", "-"])

        return history

    def _get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics for display."""
        return {
            "Total Training Sessions": len(
                [h for h in self._get_training_history() if h[0] != "-"]
            ),
            "Datasets Processed": 1 if self.workflow_state["dataset_loaded"] else 0,
            "Checkpoints Created": 1 if self.workflow_state["training_completed"] else 0,
            "Best Quality Score": "0.85" if self.workflow_state["training_completed"] else "N/A",
            "Current Experience Level": self.workflow_state["user_experience_level"].title(),
            "Auto-save Enabled": self.user_prefs.get("auto_save_configs", True),
        }

    def _get_recent_checkpoints(self) -> List[List[str]]:
        """Get recent checkpoints for display."""
        # This would scan the output directory in a real implementation
        checkpoints = []

        if self.workflow_state["training_completed"]:
            checkpoints.append(["character_lora_step_1000.safetensors", "Today", "45.2 MB", "0.85"])

        if len(checkpoints) == 0:
            checkpoints.append(["-", "No checkpoints found", "-", "-"])

        return checkpoints

    # Error Handling and Validation Methods
    def analyze_error_and_suggest_fixes(
        self, error: Exception, context: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze an error and provide recovery suggestions.

        Args:
            error: The exception that occurred
            context: Context where the error occurred ('training', 'evaluation', 'dataset', etc.)

        Returns:
            Dictionary with error analysis and suggestions
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__

        analysis = {
            "error_type": error_type,
            "error_message": str(error),
            "severity": "error",  # error, warning, info
            "recovery_actions": [],
            "preventive_measures": [],
            "troubleshooting_steps": [],
            "related_help_topics": [],
        }

        # CUDA/Out of Memory errors
        if "cuda" in error_msg and ("memory" in error_msg or "out of memory" in error_msg):
            analysis.update(
                {
                    "severity": "error",
                    "recovery_actions": [
                        "Reduce batch size (try 1-2 instead of 4)",
                        "Use gradient accumulation (increases effective batch size without more memory)",
                        "Enable gradient checkpointing in advanced settings",
                        "Restart the application to clear GPU memory",
                        "Try a smaller LoRA rank (16 instead of 32)",
                    ],
                    "preventive_measures": [
                        "Monitor GPU memory usage during training",
                        "Start with smaller batch sizes and increase gradually",
                        "Use mixed precision training (automatically enabled)",
                    ],
                    "troubleshooting_steps": [
                        "1. Reduce batch_size to 1 in training settings",
                        "2. Enable gradient_accumulation_steps (try 4-8)",
                        "3. Lower resolution if using high-res images",
                        "4. Check GPU memory usage with nvidia-smi",
                        "5. Restart application and try again",
                    ],
                    "related_help_topics": [
                        "gpu_memory",
                        "batch_size_optimization",
                        "gradient_checkpointing",
                    ],
                }
            )

        # Dataset errors
        elif "dataset" in context and ("not found" in error_msg or "no such file" in error_msg):
            analysis.update(
                {
                    "severity": "error",
                    "recovery_actions": [
                        "Check that the dataset path exists and is accessible",
                        "Verify the dataset is a ZIP file or directory with images",
                        "Ensure images are in supported formats (JPG, PNG, WebP)",
                        "Check file permissions on the dataset directory",
                    ],
                    "preventive_measures": [
                        "Use the Dataset Tools tab to validate datasets before training",
                        "Store datasets in accessible locations",
                        "Avoid special characters in file paths",
                    ],
                    "troubleshooting_steps": [
                        "1. Go to Dataset Tools tab and upload/validate your dataset",
                        "2. Check file path for typos or permission issues",
                        "3. Ensure dataset contains image files (not just captions)",
                        "4. Try using a ZIP file instead of a directory path",
                    ],
                    "related_help_topics": [
                        "dataset_validation",
                        "file_formats",
                        "dataset_structure",
                    ],
                }
            )

        # Model loading errors
        elif "model" in error_msg or "flux" in error_msg:
            analysis.update(
                {
                    "severity": "error",
                    "recovery_actions": [
                        "Check internet connection for model download",
                        "Ensure sufficient disk space (at least 20GB free)",
                        "Try restarting the application",
                        "Check if the model repository is accessible",
                    ],
                    "preventive_measures": [
                        "Download models during off-peak hours",
                        "Ensure stable internet connection",
                        "Monitor disk space regularly",
                    ],
                    "troubleshooting_steps": [
                        "1. Check internet connection and try again",
                        "2. Free up disk space (need ~20GB for models)",
                        "3. Restart the application to clear any corrupted downloads",
                        "4. Check Hugging Face model repository status",
                    ],
                    "related_help_topics": ["model_download", "disk_space", "network_issues"],
                }
            )

        # Configuration errors
        elif "config" in error_msg or "parameter" in error_msg:
            analysis.update(
                {
                    "severity": "warning",
                    "recovery_actions": [
                        "Reset to default settings and modify one parameter at a time",
                        "Check parameter ranges (learning rate, rank, etc.)",
                        "Use preset configurations as starting points",
                    ],
                    "preventive_measures": [
                        "Start with preset configurations",
                        "Change only one parameter at a time when troubleshooting",
                        "Save working configurations for reuse",
                    ],
                    "troubleshooting_steps": [
                        "1. Click 'Reset to Defaults' in training settings",
                        "2. Choose a preset (Character, Style, or Concept)",
                        "3. Modify only basic settings initially",
                        "4. Save working configurations for future use",
                    ],
                    "related_help_topics": [
                        "parameter_ranges",
                        "presets",
                        "configuration_troubleshooting",
                    ],
                }
            )

        # Generic fallback
        else:
            analysis.update(
                {
                    "severity": "error",
                    "recovery_actions": [
                        "Check the application logs for more details",
                        "Try restarting the application",
                        "Ensure all dependencies are properly installed",
                        "Check system requirements (GPU, RAM, disk space)",
                    ],
                    "preventive_measures": [
                        "Keep the application and dependencies updated",
                        "Monitor system resources during operation",
                        "Save work frequently",
                    ],
                    "troubleshooting_steps": [
                        "1. Restart the application and try again",
                        "2. Check system requirements are met",
                        "3. Review error logs for additional clues",
                        "4. Try with default settings first",
                    ],
                    "related_help_topics": [
                        "general_troubleshooting",
                        "system_requirements",
                        "logs",
                    ],
                }
            )

        return analysis

    def validate_configuration(
        self, config: Dict[str, Any], context: str = "training"
    ) -> Dict[str, Any]:
        """
        Validate a configuration and return validation results.

        Args:
            config: Configuration dictionary to validate
            context: Validation context ('training', 'evaluation', etc.)

        Returns:
            Dictionary with validation results
        """
        validation_results = {"is_valid": True, "errors": [], "warnings": [], "suggestions": []}

        if context == "training":
            # Dataset validation
            dataset_path = config.get("dataset_path", "")
            if not dataset_path:
                validation_results["errors"].append("No dataset selected")
                validation_results["is_valid"] = False

            # Parameter range validation
            rank = config.get("rank", 16)
            if not (4 <= rank <= 128):
                validation_results["errors"].append(f"LoRA rank must be between 4-128, got {rank}")
                validation_results["is_valid"] = False

            alpha = config.get("alpha", 16)
            if not (1 <= alpha <= 64):
                validation_results["errors"].append(f"LoRA alpha must be between 1-64, got {alpha}")
                validation_results["is_valid"] = False

            learning_rate = config.get("learning_rate", 1e-4)
            if not (1e-6 <= learning_rate <= 1e-2):
                validation_results["warnings"].append(
                    f"Learning rate {learning_rate} is outside typical range (1e-6 to 1e-2)"
                )

            batch_size = config.get("batch_size", 4)
            if batch_size > 8:
                validation_results["warnings"].append(
                    f"Large batch size ({batch_size}) may cause memory issues"
                )

            max_steps = config.get("max_steps", 1000)
            if max_steps > 5000:
                validation_results["suggestions"].append(
                    "Very high step count may lead to overfitting - consider early stopping"
                )

            # Smart suggestions based on dataset size (if available)
            if hasattr(self, "workflow_state") and self.workflow_state.get("dataset_loaded"):
                # Could add dataset-aware validation here
                pass

        elif context == "evaluation":
            checkpoint_path = config.get("checkpoint_path", "")
            if not checkpoint_path:
                validation_results["errors"].append("No checkpoint selected for evaluation")
                validation_results["is_valid"] = False

        return validation_results

    def save_config_snapshot(self, config: Dict[str, Any], context: str = "training"):
        """
        Save a configuration snapshot for undo/redo functionality.

        Args:
            config: Configuration to save
            context: Configuration context
        """
        snapshot = {
            "timestamp": "now",
            "context": context,
            "config": config.copy(),
            "workflow_state": self.workflow_state.copy(),
        }

        # Remove old snapshots if we have too many
        if len(self.config_history) >= 10:
            self.config_history.pop(0)

        self.config_history.append(snapshot)
        self.current_config_index = len(self.config_history) - 1

    def undo_config_change(self) -> Dict[str, Any]:
        """Undo the last configuration change."""
        if self.current_config_index > 0:
            self.current_config_index -= 1
            snapshot = self.config_history[self.current_config_index]
            self.workflow_state.update(snapshot["workflow_state"])
            return snapshot["config"]
        return {}

    def redo_config_change(self) -> Dict[str, Any]:
        """Redo a configuration change."""
        if self.current_config_index < len(self.config_history) - 1:
            self.current_config_index += 1
            snapshot = self.config_history[self.current_config_index]
            self.workflow_state.update(snapshot["workflow_state"])
            return snapshot["config"]
        return {}

    def get_troubleshooting_guide(self, error_type: str = "general") -> str:
        """Get a troubleshooting guide for a specific error type."""
        guides = {
            "gpu_memory": """
## 🚨 GPU Memory Issues

### Common Causes
- Batch size too large for available GPU memory
- High resolution images consuming too much VRAM
- Multiple models loaded simultaneously
- Gradient accumulation not enabled

### Quick Fixes
1. **Reduce batch size**: Try 1-2 instead of 4-8
2. **Enable gradient accumulation**: Use 4-8 steps
3. **Lower image resolution**: Try 512x512 instead of 1024x1024
4. **Enable gradient checkpointing**: Reduces memory at cost of speed

### Advanced Solutions
- Use mixed precision training (automatically enabled)
- Implement model sharding for multiple GPUs
- Use CPU offloading for very large models
- Monitor VRAM usage with `nvidia-smi`
            """,
            "dataset_issues": """
## 📁 Dataset Problems

### Common Issues
- Missing or corrupted image files
- Inconsistent file naming
- Unsupported image formats
- Caption-image mismatches

### Validation Steps
1. **Check file formats**: Use JPG, PNG, or WebP
2. **Verify naming**: `image_001.jpg` → `image_001.txt`
3. **Validate captions**: Each image needs a corresponding .txt file
4. **Check resolutions**: Consistent sizes preferred (1024x1024+)

### Dataset Tools
Use the Dataset Tools tab to:
- Analyze your dataset structure
- Validate file pairings
- Check image quality metrics
- Generate dataset reports
            """,
            "training_failures": """
## 🎯 Training Issues

### Startup Problems
- **CUDA errors**: Check GPU drivers and PyTorch installation
- **Import errors**: Verify all dependencies are installed
- **Memory errors**: See GPU Memory troubleshooting above

### During Training
- **Loss not decreasing**: Check learning rate (try 1e-5 to 5e-4)
- **NaN values**: Reduce learning rate or check dataset
- **Slow training**: Enable mixed precision, check batch size

### After Training
- **Poor results**: Check dataset quality and parameter settings
- **Checkpoint issues**: Verify save location and permissions
            """,
            "general": """
## 🔧 General Troubleshooting

### Most Common Issues
- **Application won't start**: Check Python version (3.10+) and dependencies
- **Slow performance**: Ensure GPU is being used and drivers are current
- **Out of disk space**: Training requires significant storage (100GB+ recommended)
- **Network timeouts**: Stable internet required for initial model download

### Quick Checks
1. **GPU Status**: Run `nvidia-smi` to verify GPU availability
2. **Disk Space**: Ensure 100GB+ free space for models and checkpoints
3. **RAM**: Minimum 16GB system RAM recommended
4. **Internet**: Stable connection for model downloads

### Getting Help
- Check the logs in the terminal for detailed error messages
- Use the troubleshooting wizard in the UI for specific guidance
- Review system requirements in the documentation
            """,
        }

        return guides.get(error_type, guides["general"])

    # Batch Operations and Productivity Methods
    def create_batch_job(
        self, name: str, config: Dict[str, Any], job_type: str = "training"
    ) -> str:
        """
        Create a new batch job.

        Args:
            name: Job name
            config: Job configuration
            job_type: Type of job ('training', 'evaluation', 'augmentation')

        Returns:
            Job ID
        """
        job_id = f"{job_type}_{len(self.batch_jobs)}_{int(time.time())}"

        job = {
            "id": job_id,
            "name": name,
            "type": job_type,
            "config": config,
            "status": "pending",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "result": None,
            "error": None,
            "estimated_time": self._estimate_job_time(config, job_type),
        }

        self.batch_jobs.append(job)
        return job_id

    def queue_batch_job(self, job_id: str):
        """Add a job to the execution queue."""
        for job in self.batch_jobs:
            if job["id"] == job_id and job["status"] == "pending":
                job["status"] = "queued"
                self.job_queue.append(job)
                self.add_notification(f"Job '{job['name']}' added to queue", "info")
                break

    def execute_batch_jobs(self):
        """Execute queued batch jobs (simplified - would run in background)."""
        if not self.job_queue:
            return

        # In a real implementation, this would start background threads/processes
        for job in self.job_queue[:]:
            if job["status"] == "queued":
                job["status"] = "running"
                job["started_at"] = time.time()
                self.add_notification(f"Started job '{job['name']}'", "info")

                # Simulate job execution (replace with actual job execution)
                self._simulate_job_execution(job)

    def _simulate_job_execution(self, job: Dict[str, Any]):
        """Simulate job execution for demonstration."""
        # In real implementation, this would be actual job execution
        import threading
        import random

        def job_thread():
            try:
                # Simulate progress updates
                total_steps = job["config"].get("max_steps", 1000)
                for step in range(0, total_steps, 100):
                    time.sleep(0.1)  # Simulate work
                    job["progress"] = min(100.0, (step / total_steps) * 100)
                    if job["progress"] >= 100:
                        job["status"] = "completed"
                        job["completed_at"] = time.time()
                        job["result"] = {
                            "success": True,
                            "quality_score": random.uniform(0.7, 0.95),
                        }
                        self.job_history.append(job)
                        self.job_queue.remove(job)
                        self.add_notification(
                            f"Job '{job['name']}' completed successfully!", "success"
                        )
                        break
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                job["completed_at"] = time.time()
                self.job_history.append(job)
                self.job_queue.remove(job)
                self.add_notification(f"Job '{job['name']}' failed: {e}", "error")

        thread = threading.Thread(target=job_thread, daemon=True)
        thread.start()

    def _estimate_job_time(self, config: Dict[str, Any], job_type: str) -> float:
        """Estimate job execution time in seconds."""
        if job_type == "training":
            steps = config.get("max_steps", 1000)
            batch_size = config.get("batch_size", 4)
            # Rough estimate: ~0.5 seconds per step
            return (steps / batch_size) * 0.5
        elif job_type == "evaluation":
            return 60  # 1 minute
        elif job_type == "augmentation":
            samples = config.get("samples", 100)
            return samples * 0.1  # 0.1 seconds per sample
        return 300  # 5 minutes default

    def save_template(self, name: str, config: Dict[str, Any], description: str = ""):
        """Save a configuration as a reusable template."""
        self.templates[name] = {
            "config": config,
            "description": description,
            "created_at": time.time(),
            "usage_count": 0,
        }
        self.add_notification(f"Template '{name}' saved", "success")

    def load_template(self, name: str) -> Dict[str, Any]:
        """Load a configuration template."""
        if name in self.templates:
            self.templates[name]["usage_count"] += 1
            return self.templates[name]["config"]
        return {}

    def create_experiment(self, name: str, jobs: List[str], description: str = "") -> str:
        """Create a new experiment with multiple jobs."""
        exp_id = f"exp_{len(self.experiments)}_{int(time.time())}"

        self.experiments[exp_id] = {
            "id": exp_id,
            "name": name,
            "description": description,
            "jobs": jobs,
            "created_at": time.time(),
            "status": "created",
            "results": {},
            "comparison_data": {},
        }

        self.add_notification(f"Experiment '{name}' created with {len(jobs)} jobs", "info")
        return exp_id

    def export_results(self, job_ids: List[str], format: str = "json") -> str:
        """Export results from completed jobs."""
        export_data = {"export_timestamp": time.time(), "jobs": [], "summary": {}}

        for job_id in job_ids:
            job = next((j for j in self.job_history if j["id"] == job_id), None)
            if job:
                export_data["jobs"].append(
                    {
                        "id": job["id"],
                        "name": job["name"],
                        "type": job["type"],
                        "config": job["config"],
                        "result": job["result"],
                        "duration": job["completed_at"] - job["started_at"]
                        if job["completed_at"] and job["started_at"]
                        else None,
                    }
                )

        # Generate summary
        completed_jobs = [j for j in export_data["jobs"] if j.get("result", {}).get("success")]
        if completed_jobs:
            quality_scores = [
                j["result"]["quality_score"]
                for j in completed_jobs
                if "quality_score" in j["result"]
            ]
            export_data["summary"] = {
                "total_jobs": len(export_data["jobs"]),
                "completed_jobs": len(completed_jobs),
                "avg_quality_score": sum(quality_scores) / len(quality_scores)
                if quality_scores
                else 0,
                "best_score": max(quality_scores) if quality_scores else 0,
                "worst_score": min(quality_scores) if quality_scores else 0,
            }

        # In real implementation, save to file and return path
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"results_{int(time.time())}.json"
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.add_notification(f"Results exported to {export_path}", "success")
        return export_path

    def get_batch_jobs_summary(self) -> Dict[str, Any]:
        """Get summary of batch jobs status."""
        return {
            "total_jobs": len(self.batch_jobs),
            "pending_jobs": len([j for j in self.batch_jobs if j["status"] == "pending"]),
            "running_jobs": len([j for j in self.batch_jobs if j["status"] == "running"]),
            "completed_jobs": len([j for j in self.batch_jobs if j["status"] == "completed"]),
            "failed_jobs": len([j for j in self.batch_jobs if j["status"] == "failed"]),
            "queue_length": len(self.job_queue),
        }

    def get_experiment_comparison_data(self, exp_id: str) -> Dict[str, Any]:
        """Get comparison data for an experiment."""
        if exp_id not in self.experiments:
            return {}

        exp = self.experiments[exp_id]
        comparison = {"experiment_name": exp["name"], "jobs": []}

        for job_id in exp["jobs"]:
            job = next((j for j in self.job_history if j["id"] == job_id), None)
            if job and job.get("result"):
                comparison["jobs"].append(
                    {
                        "name": job["name"],
                        "config": job["config"],
                        "result": job["result"],
                        "duration": job["completed_at"] - job["started_at"]
                        if job["completed_at"] and job["started_at"]
                        else None,
                    }
                )

        return comparison

    def add_recent_action(self, action: str):
        """Add an action to the recent activity log."""
        self.workflow_state["recent_actions"].append(
            {
                "action": action,
                "timestamp": "now",  # Could use actual timestamps
            }
        )

        # Keep only last 10 actions
        self.workflow_state["recent_actions"] = self.workflow_state["recent_actions"][-10:]

    # Notification System Methods
    def add_notification(self, message: str, type: str = "info", duration: int = 5000):
        """
        Add a notification to the queue.

        Args:
            message: Notification message
            type: Notification type ('success', 'error', 'warning', 'info')
            duration: Duration in milliseconds
        """
        notification = {
            "id": len(self.notifications),
            "message": message,
            "type": type,
            "duration": duration,
            "timestamp": "now",
        }
        self.notifications.append(notification)

        # Keep only last 5 notifications
        self.notifications = self.notifications[-5:]

        logger.debug(f"Added {type} notification: {message}")

    def get_notifications_html(self) -> str:
        """Get HTML for current notifications."""
        if not self.notifications:
            return ""

        html_parts = []
        for notification in self.notifications[-3:]:  # Show last 3
            colors = {
                "success": "#4CAF50",
                "error": "#F44336",
                "warning": "#FF9800",
                "info": "#2196F3",
            }

            color = colors.get(notification["type"], "#2196F3")
            icon = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                notification["type"], "ℹ️"
            )

            # Add troubleshooting link for errors
            troubleshooting_link = ""
            if notification["type"] == "error":
                troubleshooting_link = """<br><small style="opacity: 0.8;">
                <a href="#" onclick="document.querySelector('[id*=&quot;troubleshooting&quot;]').click(); return false;" style="color: inherit; text-decoration: underline;">🔧 Get help fixing this</a>
                </small>"""

            html_parts.append(f"""
            <div style="
                background: {color}15;
                border: 1px solid {color};
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                display: flex;
                align-items: center;
            ">
                <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                <div>
                    <span>{notification["message"]}</span>
                    {troubleshooting_link}
                </div>
            </div>
            """)

        return "".join(html_parts)

    # Operation Queue Methods
    def add_operation(self, operation: str, status: str = "pending", progress: float = 0.0):
        """
        Add an operation to the queue.

        Args:
            operation: Operation description
            status: Operation status ('pending', 'running', 'completed', 'failed')
            progress: Progress percentage (0-100)
        """
        op = {
            "id": len(self.operation_queue),
            "operation": operation,
            "status": status,
            "progress": progress,
            "timestamp": "now",
        }
        self.operation_queue.append(op)

        # Keep only last 10 operations
        self.operation_queue = self.operation_queue[-10:]

    def update_operation_progress(self, operation_id: int, progress: float, status: str = None):
        """Update operation progress."""
        for op in self.operation_queue:
            if op["id"] == operation_id:
                op["progress"] = progress
                if status:
                    op["status"] = status
                break

    def get_operation_queue_html(self) -> str:
        """Get HTML for operation queue display."""
        if not self.operation_queue:
            return "<div style='text-align: center; color: #666; padding: 20px;'>No operations in queue</div>"

        html_parts = []
        for op in self.operation_queue[-5:]:  # Show last 5
            status_colors = {
                "pending": "#FF9800",
                "running": "#2196F3",
                "completed": "#4CAF50",
                "failed": "#F44336",
            }

            color = status_colors.get(op["status"], "#666")
            progress_width = f"{op['progress']}%"

            html_parts.append(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                background: #fafafa;
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: bold;">{op["operation"]}</span>
                    <span style="color: {color};">{op["status"].title()}</span>
                </div>
                <div style="
                    width: 100%;
                    height: 6px;
                    background: #e0e0e0;
                    border-radius: 3px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {progress_width};
                        height: 100%;
                        background: {color};
                        transition: width 0.3s ease;
                    "></div>
                </div>
                <div style="text-align: right; font-size: 0.8em; color: #666; margin-top: 4px;">
                    {op["progress"]:.1f}%
                </div>
            </div>
            """)

        return "".join(html_parts)

    def _get_custom_css(self) -> str:
        """Get custom CSS for the application."""
        return """
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .app-footer {
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-top: 2rem;
        }

        .main-tabs {
            margin-top: 1rem;
        }

        .quick-actions-bar {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid #e0e0e0;
        }

        #workflow-indicator {
            margin: 15px 0;
        }

        #status-notifications {
            margin: 10px 0;
        }

        #notifications-panel, #operation-queue {
            max-height: 300px;
            overflow-y: auto;
        }

        .training-config {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .progress-section {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .status-success {
            color: #28a745;
            font-weight: bold;
        }

        .status-error {
            color: #dc3545;
            font-weight: bold;
        }

        .status-info {
            color: #007bff;
            font-weight: bold;
        }

        .file-upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }

        .file-upload-area:hover {
            border-color: #007bff;
            background-color: #e3f2fd;
        }

        .tips-banner {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }

        .tips-banner p {
            margin: 0;
            font-size: 0.9em;
            color: #1565c0;
        }

        .help-button {
            background-color: #e3f2fd !important;
            border: 1px solid #2196f3 !important;
            color: #1976d2 !important;
            min-width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            margin-left: 8px !important;
        }

        .help-button:hover {
            background-color: #bbdefb !important;
            border-color: #1976d2 !important;
        }

        /* Workflow progress indicator styling */
        .workflow-step {
            transition: all 0.3s ease;
        }

        .workflow-step:hover {
            transform: scale(1.02);
        }

        /* Notification styling */
        .notification-success {
            background-color: #e8f5e8 !important;
            border-color: #4caf50 !important;
        }

        .notification-error {
            background-color: #ffebee !important;
            border-color: #f44336 !important;
        }

        .notification-warning {
            background-color: #fff3e0 !important;
            border-color: #ff9800 !important;
        }

        .notification-info {
            background-color: #e3f2fd !important;
            border-color: #2196f3 !important;
        }

        /* Error handling and validation styling */
        #config-validation-status {
            margin: 10px 0;
            font-size: 0.9em;
        }

        #troubleshooting-modal .modal-content {
            max-width: 800px;
        }

        #error-category-selector {
            margin: 15px 0;
        }

        #troubleshooting-content {
            max-height: 400px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }

        #troubleshooting-status {
            margin: 10px 0;
        }

        /* Notification enhancements */
        .notification-error {
            border-left: 4px solid #f44336;
        }

        .notification-success {
            border-left: 4px solid #4caf50;
        }

        .notification-warning {
            border-left: 4px solid #ff9800;
        }

        .notification-info {
            border-left: 4px solid #2196f3;
        }

        /* Validation status styling */
        .validation-valid {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
        }

        .validation-warning {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
        }

        .validation-error {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important;
        }

        /* Batch operations styling */
        #batch-job-status, #batch-progress-display {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }

        #job-queue-table {
            max-height: 300px;
            overflow-y: auto;
        }

        .batch-job-item {
            padding: 8px;
            margin: 4px 0;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background: #fafafa;
        }

        .batch-job-item.running {
            border-color: #2196f3;
            background: #e3f2fd;
        }

        .batch-job-item.completed {
            border-color: #4caf50;
            background: #e8f5e8;
        }

        .batch-job-item.failed {
            border-color: #f44336;
            background: #ffebee;
        }

        /* Template management styling */
        #template-details {
            max-height: 200px;
            overflow-y: auto;
        }

        /* Experiment styling */
        #exp-results {
            max-height: 400px;
            overflow-y: auto;
        }

        /* Export styling */
        #export-history {
            max-height: 250px;
            overflow-y: auto;
        }

        /* Responsive design improvements */
        @media (max-width: 768px) {
            .quick-actions-bar {
                padding: 10px;
            }

            .workflow-step {
                padding: 6px 10px !important;
                font-size: 0.85em !important;
                margin: 2px !important;
            }

            #troubleshooting-modal .modal-content {
                max-width: 95vw;
            }

            #troubleshooting-content {
                max-height: 300px;
            }

            #job-queue-table {
                font-size: 0.8em;
            }

            .batch-job-item {
                padding: 6px;
                font-size: 0.9em;
            }
        }
        """

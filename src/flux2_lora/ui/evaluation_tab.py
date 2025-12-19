"""
Evaluation tab for the Gradio interface.

Provides tools for testing and comparing trained LoRA checkpoints.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp

from .help_utils import help_system


def load_checkpoint_for_evaluation(app: "LoRATrainingApp", checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint for evaluation and return metadata.

    Args:
        app: Main application instance
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    try:
        path = Path(checkpoint_path)

        if not path.exists():
            return {"error": f"Checkpoint file not found: {checkpoint_path}"}

        if not path.is_file():
            return {"error": f"Path is not a file: {checkpoint_path}"}

        # Basic file validation
        if not checkpoint_path.endswith(".safetensors"):
            return {"error": "Only .safetensors files are supported for security"}

        # Get file information
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Try to extract basic metadata from checkpoint
        metadata = {
            "path": str(path),
            "filename": path.name,
            "size_mb": round(file_size_mb, 2),
            "loaded": False,
            "status": "Ready for loading",
        }

        # Store checkpoint path in app state
        app.update_training_state("evaluation_checkpoint", str(path))

        return metadata

    except Exception as e:
        return {"error": f"Failed to load checkpoint: {str(e)}"}


def run_quality_assessment(
    app: "LoRATrainingApp", checkpoint_path: str, test_prompts: List[str]
) -> Dict[str, Any]:
    """
    Run quality assessment on a checkpoint.

    Args:
        app: Main application instance
        checkpoint_path: Path to checkpoint
        test_prompts: List of prompts to test

    Returns:
        Assessment results
    """
    try:
        from ..evaluation.quality_metrics import QualityAssessor

        assessor = QualityAssessor()

        # Use default prompts if none provided
        if not test_prompts or all(not p.strip() for p in test_prompts):
            test_prompts = [
                "A portrait of a person with distinctive features",
                "A landscape scene with mountains and water",
                "A detailed close-up of an object",
            ]

        # Run assessment with limited samples for speed
        results = assessor.assess_checkpoint_quality(
            checkpoint_path=checkpoint_path,
            test_prompts=test_prompts,
            num_samples_per_prompt=2,  # Reduced for UI speed
            generation_kwargs={
                "num_inference_steps": 20,  # Reduced for speed
                "guidance_scale": 7.5,
            },
        )

        return results

    except Exception as e:
        return {"error": f"Quality assessment failed: {str(e)}"}


def run_prompt_test(
    app: "LoRATrainingApp", checkpoint_path: str, concept: str, trigger_word: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run prompt testing suite on a checkpoint.

    Args:
        app: Main application instance
        checkpoint_path: Path to checkpoint
        concept: Concept to test
        trigger_word: Optional trigger word

    Returns:
        Test results
    """
    try:
        from ..evaluation.prompt_testing import PromptTester

        tester = PromptTester(
            checkpoint_path=checkpoint_path,
            trigger_word=trigger_word,
        )

        # Run a quick test with limited prompts
        test_suite = tester.create_test_suite(concept=concept)

        # Limit to basic tests for UI speed
        basic_tests = [test for test in test_suite if test.category == "basic"][:3]

        if not basic_tests:
            basic_tests = test_suite[:3]  # Fallback

        results = tester.run_test_suite(basic_tests)

        return results

    except Exception as e:
        return {"error": f"Prompt testing failed: {str(e)}"}


def run_checkpoint_comparison(
    app: "LoRATrainingApp", checkpoint_paths: List[str], test_prompt: str
) -> Dict[str, Any]:
    """
    Run checkpoint comparison.

    Args:
        app: Main application instance
        checkpoint_paths: List of checkpoint paths
        test_prompt: Test prompt to use

    Returns:
        Comparison results
    """
    try:
        from ..evaluation.checkpoint_compare import CheckpointComparator

        comparator = CheckpointComparator(
            checkpoint_paths=checkpoint_paths,
            test_prompts=[test_prompt] if test_prompt else ["A beautiful landscape"],
            num_inference_steps=20,  # Reduced for speed
            guidance_scale=7.5,
        )

        # Run comparison
        results = comparator.run_full_comparison()

        return results

    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}


def create_evaluation_tab(app: "LoRATrainingApp"):
    """
    Create the evaluation tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Evaluation Help & Tips", open=False):
        gr.Markdown(help_system.get_evaluation_help_text())

        # Additional feature explanations
        with gr.Accordion("📈 Quality Assessment Deep Dive", open=False):
            gr.Markdown(help_system.get_feature_overview()["quality_assessment"])
            gr.Markdown("""
            ### Understanding Quality Metrics

            #### CLIP Score (0-1 scale)
            - **What it measures**: Semantic similarity between your prompt and generated image
            - **Excellent**: >0.8 (image strongly matches prompt)
            - **Good**: 0.6-0.8 (image matches prompt reasonably well)
            - **Poor**: <0.6 (image doesn't match prompt well)
            - **Factors**: Prompt clarity, training quality, subject complexity

            #### Diversity Score (0-1 scale)
            - **What it measures**: Variety in generated images (prevents repetitive results)
            - **Excellent**: >0.7 (good variety, not repetitive)
            - **Good**: 0.5-0.7 (moderate variety)
            - **Poor**: <0.5 (very repetitive or mode-collapsed)
            - **Factors**: Dataset diversity, training stability, LoRA capacity

            #### Overfitting Risk (0-1 scale)
            - **What it measures**: How closely results match training images (too close = bad)
            - **Low Risk**: <0.7 (good generalization)
            - **Medium Risk**: 0.7-0.85 (some overfitting, monitor closely)
            - **High Risk**: >0.85 (significant overfitting, stop training)
            - **Factors**: Training duration, dataset size, learning rate

            #### Composite Quality Score
            - **Weighted combination**: CLIP (40%) + Diversity (30%) + Overfitting penalty (30%)
            - **Excellent**: >0.8 overall score
            - **Good**: 0.6-0.8 overall score
            - **Needs Improvement**: <0.6 overall score
            """)

        with gr.Accordion("🔄 Checkpoint Comparison Guide", open=False):
            gr.Markdown(help_system.get_feature_overview()["checkpoint_comparison"])
            gr.Markdown("""
            ### When to Compare Checkpoints

            #### During Training
            - Compare checkpoints every 100-200 steps
            - Look for quality improvements over time
            - Stop when quality peaks or starts declining

            #### After Training
            - Test multiple checkpoints from different stages
            - Find the sweet spot (good quality, low overfitting)
            - Compare different training configurations

            #### Best Practices
            - Use consistent prompts for fair comparison
            - Generate multiple samples per checkpoint (3-5 minimum)
            - Consider both automated metrics and visual inspection
            - Save top 2-3 checkpoints for final evaluation

            ### Reading Comparison Results

            #### Grid Layout
            - **Rows**: Different prompts/test cases
            - **Columns**: Different checkpoints
            - **Cells**: Generated images for that prompt+checkpoint combination

            #### Metrics Table
            - **CLIP Scores**: Which checkpoint best understands prompts
            - **Diversity Scores**: Which checkpoint produces varied results
            - **Quality Rankings**: Overall best performer across all metrics

            #### Making Decisions
            - Prioritize checkpoints with high CLIP scores
            - Avoid checkpoints with high overfitting risk
            - Balance quality and generalization
            - Consider your specific use case (quality vs. variety)
            """)

        with gr.Accordion("🧪 Prompt Testing Best Practices", open=False):
            gr.Markdown("""
            ### Systematic Prompt Testing

            #### Basic Tests
            - Simple prompts: "A [trigger] in the style of..."
            - Direct subject: "Photo of [trigger]"
            - Minimal context: "[trigger] portrait"
            - **Goal**: Verify basic functionality works

            #### Positioning Tests
            - Start of prompt: "[trigger] walking in a garden"
            - Middle of prompt: "A beautiful scene with [trigger] in it"
            - End of prompt: "Create an image of a forest, [trigger]"
            - **Goal**: Check trigger word reliability regardless of position

            #### Composition Tests
            - Complex scenes: "[trigger] fighting a dragon in a castle"
            - Multiple subjects: "[trigger] with friends at a party"
            - Environmental: "[trigger] underwater in an ocean scene"
            - **Goal**: Test how well your LoRA composes with other elements

            #### Style Transfer Tests
            - Art styles: "[trigger] in the style of Van Gogh"
            - Time periods: "Medieval [trigger] knight"
            - Genres: "Cyberpunk [trigger] in a futuristic city"
            - **Goal**: Verify LoRA works across different artistic contexts

            #### Negative Tests
            - Opposites: "Photo of someone who is not [trigger]"
            - Exclusions: "A group of people, none of whom are [trigger]"
            - **Goal**: Ensure your LoRA doesn't activate inappropriately

            ### Interpreting Test Results

            #### Success Ratings
            - **Excellent**: Consistent, high-quality results across all tests
            - **Good**: Reliable results for most prompts, minor issues
            - **Fair**: Works for basic prompts but struggles with complex ones
            - **Poor**: Unreliable or incorrect results

            #### Common Issues
            - **Inconsistent triggering**: Trigger word works sometimes but not always
            - **Style bleeding**: LoRA affects prompts where it shouldn't activate
            - **Limited composition**: Struggles when combined with other subjects/styles
            - **Overfitting**: Only works with training-style prompts

            #### Improvement Strategies
            - **More training data**: Add diverse examples of your subject
            - **Better captions**: Use trigger words consistently in training captions
            - **Extended training**: Train for more steps to improve generalization
            - **Refined dataset**: Remove outliers and improve consistency
            """)

        with gr.Accordion("📊 Choosing the Best Checkpoint", open=False):
            gr.Markdown("""
            ### Checkpoint Selection Strategy

            #### Multi-Criteria Decision Making
            When multiple checkpoints perform well, consider:
            - **Primary Use Case**: Quality vs. versatility vs. consistency
            - **Prompt Complexity**: Simple prompts vs. complex compositions
            - **Style Compatibility**: How well it works with different art styles
            - **Overfitting Risk**: Generalization vs. training data similarity

            #### Quality vs. Training Stage
            - **Early checkpoints** (100-500 steps): May lack detail but good generalization
            - **Mid checkpoints** (500-1000 steps): Balance of quality and generalization
            - **Late checkpoints** (1000+ steps): High quality but risk of overfitting

            #### Practical Testing
            1. **Narrow down**: Use metrics to identify top 3-5 candidates
            2. **Manual testing**: Generate 5-10 images per checkpoint
            3. **Edge cases**: Test unusual prompts and compositions
            4. **Consistency check**: Same prompt multiple times should give similar quality
            5. **Final selection**: Choose based on your specific needs

            #### Production Considerations
            - **Reliability**: Choose checkpoint with consistent results
            - **Versatility**: Select one that works across different prompt types
            - **Quality threshold**: Ensure minimum quality standards are met
            - **Future-proofing**: Consider how you'll use this LoRA long-term

            ### Common Selection Mistakes
            - **Latest = Best**: Later checkpoints aren't always better
            - **Metrics Only**: Human judgment still matters
            - **Single Prompt**: Test with multiple prompt types
            - **Ignoring Overfitting**: High training similarity can limit usefulness
            - **No Edge Testing**: Unusual prompts reveal limitations
            """)

    with gr.Row():
        with gr.Column(scale=1):
            # Left column: Controls
            gr.Markdown("## 🔍 Checkpoint Evaluation")

            with gr.Group():
                gr.Markdown("### Load Checkpoint")

                # Checkpoint selection
                checkpoint_source = gr.Radio(
                    choices=["Upload File", "Local Path"],
                    value="Upload File",
                    label="Checkpoint Source",
                )

                checkpoint_upload = gr.File(
                    label="Upload Checkpoint (.safetensors)",
                    file_types=[".safetensors"],
                    visible=True,
                )

                checkpoint_path = gr.Textbox(
                    label="Checkpoint Path",
                    placeholder="/path/to/checkpoint.safetensors",
                    visible=False,
                )

                # Checkpoint info
                checkpoint_info = gr.JSON(
                    label="Checkpoint Information",
                    value={},
                )

                gr.Markdown("### Test Generation")

                # Prompt input
                test_prompt = gr.Textbox(
                    label="Test Prompt",
                    placeholder="Enter a prompt to test the checkpoint...",
                    lines=3,
                )

                # Generation parameters
                with gr.Row():
                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=9,
                        value=3,
                        label="Number of Samples",
                    )

                    inference_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        label="Inference Steps",
                    )

                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    label="Guidance Scale",
                )

                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Samples",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### Quality Assessment")

                # Assessment buttons
                assess_btn = gr.Button(
                    "📊 Assess Quality",
                    variant="secondary",
                )

                test_prompts_btn = gr.Button(
                    "🧪 Run Prompt Tests",
                    variant="secondary",
                )

        with gr.Column(scale=2):
            # Right column: Results
            gr.Markdown("## 🎯 Results")

            # Generated images
            generated_gallery = gr.Gallery(
                label="Generated Images",
                columns=3,
                height=400,
                allow_preview=True,
            )

            # Quality metrics
            with gr.Accordion("📈 Quality Metrics", open=False):
                quality_metrics = gr.JSON(label="Quality Assessment", value={})

                metrics_plot = gr.Plot(label="Quality Visualization")

            # Prompt test results
            with gr.Accordion("🧪 Prompt Test Results", open=False):
                test_results = gr.JSON(label="Test Results", value={})

                test_summary = gr.Markdown(value="No tests run yet.")

            # Comparison section
            gr.Markdown("### Checkpoint Comparison")

            with gr.Group():
                # Multi-checkpoint selection for comparison
                comparison_checkpoints = gr.Files(
                    label="Select Checkpoints to Compare",
                    file_types=[".safetensors"],
                    file_count="multiple",
                )

                compare_btn = gr.Button(
                    "🔄 Compare Checkpoints",
                    variant="secondary",
                )

                # Comparison results
                comparison_results = gr.Gallery(label="Comparison Grid", columns=4, height=300)

                comparison_metrics = gr.Dataframe(
                    label="Comparison Metrics",
                    headers=["Checkpoint", "CLIP Score", "Diversity", "Quality"],
                    datatype=["str", "number", "number", "number"],
                    value=[],
                )

    # State variables for UI updates
    loaded_checkpoint = gr.State(None)
    comparison_state = gr.State({})

    # Event handlers
    def update_checkpoint_visibility(source):
        """Update checkpoint input visibility based on source selection."""
        if source == "Upload File":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    checkpoint_source.change(
        fn=update_checkpoint_visibility,
        inputs=[checkpoint_source],
        outputs=[checkpoint_upload, checkpoint_path],
    )

    # Checkpoint upload handler
    def handle_checkpoint_upload(file_obj):
        """Handle checkpoint file upload."""
        if file_obj:
            # For uploaded files, use the temporary path
            checkpoint_path = file_obj.name
            metadata = load_checkpoint_for_evaluation(app, checkpoint_path)
            return metadata
        return {}

    checkpoint_upload.change(
        fn=handle_checkpoint_upload,
        inputs=[checkpoint_upload],
        outputs=[checkpoint_info, loaded_checkpoint],
    )

    # Checkpoint path handler
    def handle_checkpoint_path(path):
        """Handle checkpoint path input."""
        if path and path.strip():
            metadata = load_checkpoint_for_evaluation(app, path.strip())
            return metadata
        return {}

    checkpoint_path.change(
        fn=handle_checkpoint_path,
        inputs=[checkpoint_path],
        outputs=[checkpoint_info, loaded_checkpoint],
    )

    # Generate samples handler
    def generate_samples_handler(
        loaded_checkpoint, test_prompt, num_samples, inference_steps, guidance_scale
    ):
        """Handle sample generation."""
        if not loaded_checkpoint or "error" in loaded_checkpoint:
            return [], {"error": "No valid checkpoint loaded"}

        if not test_prompt or not test_prompt.strip():
            return [], {"error": "Please enter a test prompt"}

        try:
            # Import required modules
            from ..core.model_loader import ModelLoader
            import torch

            # Load model and checkpoint
            model_loader = ModelLoader()
            model, _ = model_loader.load_flux2_dev()

            # Load LoRA weights
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, loaded_checkpoint["path"])

            # Generate samples
            with torch.no_grad():
                result = model(
                    prompt=test_prompt,
                    num_images_per_prompt=int(num_samples),
                    num_inference_steps=int(inference_steps),
                    guidance_scale=float(guidance_scale),
                    height=1024,
                    width=1024,
                )

            images = result.images if hasattr(result, "images") else []

            return images, {"success": True, "generated": len(images), "prompt": test_prompt}

        except Exception as e:
            return [], {"error": f"Generation failed: {str(e)}"}

    generate_btn.click(
        fn=generate_samples_handler,
        inputs=[loaded_checkpoint, test_prompt, num_samples, inference_steps, guidance_scale],
        outputs=[generated_gallery, quality_metrics],
    )

    # Quality assessment handler
    def assess_quality_handler(loaded_checkpoint, test_prompt):
        """Handle quality assessment."""
        if not loaded_checkpoint or "error" in loaded_checkpoint:
            return {"error": "No valid checkpoint loaded"}

        # Prepare prompts
        prompts = []
        if test_prompt and test_prompt.strip():
            prompts.append(test_prompt.strip())
        prompts.extend(["A portrait of a person", "A landscape scene", "An object in detail"])

        try:
            results = run_quality_assessment(app, loaded_checkpoint["path"], prompts)

            # Format results for display
            if "error" in results:
                return results

            display_results = {
                "checkpoint": loaded_checkpoint["filename"],
                "quality_score": results.get("quality_score", 0),
                "clip_score": results.get("clip_score", 0),
                "diversity_score": results.get("diversity_score", 0),
                "is_overfitting": results.get("is_overfitting", False),
                "prompts_tested": len(prompts),
            }

            return display_results

        except Exception as e:
            return {"error": f"Assessment failed: {str(e)}"}

    assess_btn.click(
        fn=assess_quality_handler,
        inputs=[loaded_checkpoint, test_prompt],
        outputs=[quality_metrics],
    )

    # Prompt testing handler
    def test_prompts_handler(loaded_checkpoint, test_prompt):
        """Handle prompt testing."""
        if not loaded_checkpoint or "error" in loaded_checkpoint:
            return {"error": "No valid checkpoint loaded"}

        # Extract concept from prompt or use default
        concept = "person"
        if test_prompt and test_prompt.strip():
            # Simple concept extraction
            prompt_lower = test_prompt.lower()
            if "portrait" in prompt_lower or "person" in prompt_lower:
                concept = "person"
            elif "landscape" in prompt_lower or "scene" in prompt_lower:
                concept = "landscape"
            elif "object" in prompt_lower:
                concept = "object"

        try:
            results = run_prompt_test(app, loaded_checkpoint["path"], concept)

            if "error" in results:
                return results

            # Format results for display
            summary = results.get("summary", {})
            display_results = {
                "concept": concept,
                "tests_run": summary.get("total_tests", 0),
                "success_rate": summary.get("success_rate", 0),
                "average_score": summary.get("average_score", 0),
                "best_category": "N/A",
                "recommendations": results.get("analysis", {}).get("recommendations", [])[:2],
            }

            return display_results

        except Exception as e:
            return {"error": f"Testing failed: {str(e)}"}

    test_prompts_btn.click(
        fn=test_prompts_handler, inputs=[loaded_checkpoint, test_prompt], outputs=[test_results]
    )

    # Comparison handler
    def compare_checkpoints_handler(checkpoint_files, test_prompt):
        """Handle checkpoint comparison."""
        if not checkpoint_files:
            return [], {"error": "No checkpoint files selected"}

        # Extract file paths
        checkpoint_paths = []
        for file_obj in checkpoint_files:
            if hasattr(file_obj, "name"):
                checkpoint_paths.append(file_obj.name)
            else:
                checkpoint_paths.append(str(file_obj))

        if len(checkpoint_paths) < 2:
            return [], {"error": "Please select at least 2 checkpoints to compare"}

        prompt = (
            test_prompt.strip() if test_prompt and test_prompt.strip() else "A beautiful landscape"
        )

        try:
            results = run_checkpoint_comparison(app, checkpoint_paths, prompt)

            if "error" in results:
                return [], results

            # Extract images from results
            images = []
            if "image_grid" in results:
                images.append(results["image_grid"])
            elif "generated_images" in results:
                images.extend(results["generated_images"])

            # Format metrics for display
            metrics_data = []
            if "metrics" in results:
                for ckpt_name, metrics in results["metrics"].items():
                    metrics_data.append(
                        [
                            ckpt_name,
                            round(metrics.get("clip_score", 0), 3),
                            round(metrics.get("diversity_score", 0), 3),
                            round(metrics.get("quality_score", 0), 3),
                        ]
                    )

            return images, metrics_data

        except Exception as e:
            return [], {"error": f"Comparison failed: {str(e)}"}

    compare_btn.click(
        fn=compare_checkpoints_handler,
        inputs=[comparison_checkpoints, test_prompt],
        outputs=[comparison_results, comparison_metrics],
    )

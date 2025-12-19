"""
Help and guidance utilities for the Gradio interface.

Provides tooltips, help text, and guidance for users.
"""

from typing import Dict, Any


class HelpSystem:
    """Centralized help system for the Gradio interface."""

    @staticmethod
    def get_welcome_message() -> str:
        """Get the welcome message for first-time users."""
        return """
        ## 🎨 Welcome to Flux2-dev LoRA Training Toolkit!

        ### What This Toolkit Does
        This toolkit helps you create custom AI models (LoRAs) that can generate images in specific styles, characters, or concepts.
        LoRA training adapts the powerful Flux2-dev model to understand your unique subject without retraining the entire model.

        ### Quick Start Workflow
        1. **Prepare Dataset**: 10-50 images + descriptive captions
        2. **Train LoRA**: Choose preset, upload data, start training
        3. **Evaluate**: Test your LoRA with different prompts
        4. **Iterate**: Refine based on results and train again if needed

        ### Four Main Areas
        - **🚀 Training**: Create your custom LoRA model
        - **📊 Evaluation**: Test and compare your trained models
        - **🗂️ Dataset Tools**: Analyze and improve your training data
        - **⚡ Optimization**: Automatically find the best training settings

        ### Pro Tips
        - Start with the **Training** tab for your first LoRA
        - Use **Dataset Tools** to validate and improve your data
        - Try **Optimization** to squeeze maximum quality from your training
        - Use **Evaluation** to test and compare different approaches

        ### Need Help?
        - Click the help sections (💡) at the top of each tab for detailed guidance
        - Hover over components for quick tips
        - Check the comprehensive documentation in the project README

        Happy training! 🎨✨
        """

    @staticmethod
    def get_feature_overview() -> Dict[str, str]:
        """Get detailed explanations of major toolkit features."""
        return {
            "lora_training": """
            ## 🎯 LoRA Training
            **What it does**: Trains small adapter layers on top of Flux2-dev to customize image generation for specific subjects.

            **When to use**: When you want the AI to generate images in a particular style, character, or concept that isn't well-represented in the base model.

            **How it works**: Instead of retraining millions of parameters, LoRA adds small trainable layers that adapt the model's behavior. This is efficient and preserves the base model's capabilities while adding your customization.

            **Expected results**: After training, you can use trigger words in prompts to activate your LoRA's learned style or subject.
            """,
            "validation_sampling": """
            ## 👁️ Validation Sampling
            **What it does**: Generates sample images during training to preview your LoRA's progress and catch issues early.

            **Why it matters**: Training can take hours, so you need to monitor progress. Validation samples show you exactly what your LoRA is learning.

            **How to interpret**: Look for images that increasingly match your subject/style. If samples look wrong, you may need to adjust your dataset or training settings.
            """,
            "quality_assessment": """
            ## 📈 Quality Assessment
            **What it does**: Measures how well your trained LoRA performs using automated metrics.

            **Key metrics**:
            - **CLIP Score**: How closely generated images match your prompts (0-1 scale)
            - **Diversity Score**: Variety in generated images (prevents repetitive results)
            - **Overfitting Detection**: Checks if your LoRA only memorizes training images

            **When to use**: After training completes, or to compare multiple checkpoints from different training stages.
            """,
            "checkpoint_comparison": """
            ## 🔄 Checkpoint Comparison
            **What it does**: Side-by-side comparison of multiple trained LoRA checkpoints to find the best one.

            **Why it's useful**: Training creates multiple checkpoints. This tool helps you identify which training stage produced the best results.

            **How it works**: Generates the same prompts with different checkpoints and displays results in a grid for easy comparison.
            """,
            "dataset_analysis": """
            ## 📊 Dataset Analysis
            **What it does**: Comprehensive analysis of your training dataset to identify issues and optimization opportunities.

            **Checks performed**:
            - Image quality and consistency
            - Caption completeness and quality
            - File pairing (image ↔ caption matching)
            - Resolution and format validation
            - Statistical overview and recommendations

            **Benefits**: Fixes dataset issues before training, leading to better results and faster training.
            """,
            "preset_system": """
            ## ⚙️ Training Presets
            **What they are**: Optimized configurations for different types of LoRA training.

            **Available presets**:
            - **Character**: For people, characters, creatures (higher learning rate, more steps)
            - **Style**: For artistic styles, painting techniques (balanced settings)
            - **Concept**: For objects, scenes, abstract ideas (conservative settings)

            **Why use presets**: They provide proven starting points. Fine-tune advanced settings as needed.
            """,
            "hyperparameter_optimization": """
            ## ⚡ Hyperparameter Optimization
            **What it does**: Automatically finds the best training settings for your specific dataset using Bayesian optimization.

            **When to use**: When you want to squeeze maximum quality out of your LoRA training, or when default settings aren't working well.

            **How it works**: Tests different combinations of LoRA rank, learning rate, batch size, etc., and uses machine learning to find the optimal settings for your dataset.

            **Benefits**: Can improve final LoRA quality by 10-30% and make training more efficient.
            """,
            "dataset_augmentation": """
            ## 🎨 Dataset Augmentation
            **What it does**: Generates additional training samples by applying various transformations to your existing dataset.

            **When to use**: When you have a small dataset, limited variety, or want to improve model robustness and generalization.

            **How it works**: Applies image transformations (flips, color changes) and text augmentations (synonym replacement) to create diverse training samples.

            **Benefits**: Can 2-3x your effective dataset size, reduce overfitting, and improve model generalization.
            """,
        }

    @staticmethod
    def get_augmentation_help_text() -> str:
        """Get detailed help text for the augmentation tab."""
        return """
        ## 🎨 Dataset Augmentation

        ### What is Dataset Augmentation?
        Dataset augmentation creates additional training samples by applying various transformations
        to your existing images and captions. This helps improve model robustness and generalization.

        ### Why Use Augmentation?
        **Expand Limited Datasets**: Turn 50 images into 150+ training samples
        **Improve Generalization**: Help models work with variations they haven't seen
        **Reduce Overfitting**: Prevent models from memorizing specific images
        **Increase Robustness**: Make models work better with different lighting, angles, etc.

        ### Types of Augmentations

        #### Image Augmentations
        - **Geometric**: Flips, rotations, scaling, cropping
        - **Color**: Brightness, contrast, saturation adjustments
        - **Noise**: Gaussian noise, salt-and-pepper noise
        - **Blur**: Gaussian blur, motion blur effects

        #### Text Augmentations
        - **Synonym Replacement**: Replace words with synonyms
        - **Random Deletion**: Remove random words
        - **Random Swap**: Rearrange word order
        - **Backtranslation**: Translate and re-translate (advanced)

        ### Best Practices

        #### Quality Preservation
        - Enable quality checks to avoid degraded samples
        - Review augmented samples before training
        - Balance augmentation intensity with realism

        #### Balanced Augmentation
        - Don't over-augment (2-3 augmentations per sample max)
        - Mix different augmentation types
        - Include original samples alongside augmented ones

        #### Domain Awareness
        - Choose augmentations appropriate for your subject
        - Avoid transformations that change semantic meaning
        - Test augmentation impact on your use case

        ### When to Augment

        #### Small Datasets (< 50 images)
        - Augmentation can 2-3x your effective dataset size
        - Helps prevent overfitting
        - Improves model generalization

        #### Limited Variety
        - Add different angles and poses
        - Generate lighting variations
        - Create stylistic diversity

        #### Domain Gap Issues
        - Bridge differences between training and target data
        - Generate more representative samples
        - Improve robustness to real-world variations

        ### Common Mistakes
        - **Over-augmentation**: Too many transformations degrade quality
        - **Inappropriate augmentations**: Flipping text, unnatural color changes
        - **Ignoring quality**: Using degraded samples hurts training
        - **No validation**: Not checking if augmentations help performance

        ### Measuring Impact
        - Compare model performance with/without augmentation
        - Check if augmented models generalize better
        - Validate that augmentations don't introduce artifacts
        - Monitor training stability and convergence
        """

    @staticmethod
    def get_optimization_help_text() -> str:
        """Get detailed help text for the optimization tab."""
        return """
        ## ⚡ Hyperparameter Optimization

        ### What is Hyperparameter Optimization?
        Automatically find the best training settings for your specific dataset by testing different combinations of parameters.
        This process uses Bayesian optimization to intelligently explore the parameter space and find optimal settings.

        ### Why Optimize Hyperparameters?
        **Better Results**: Optimized settings can improve final LoRA quality by 10-30%
        **Faster Training**: Better parameters often train faster and more reliably
        **Memory Efficiency**: Optimized batch sizes and accumulation settings use GPU memory better
        **Dataset-Specific**: Each dataset may have different optimal settings

        ### What Gets Optimized
        - **LoRA Rank** (4-128): Model capacity - higher values learn more complex patterns
        - **LoRA Alpha** (4-128): Strength scaling - usually set equal to rank
        - **Learning Rate** (1e-6 to 1e-2): Training speed - too high = unstable, too low = slow
        - **Batch Size** (1-16): Images processed together - affects memory and stability
        - **Gradient Accumulation** (1-8): Effective batch size when GPU memory is limited

        ### How the Optimization Works
        1. **Initial Trials**: Tests random parameter combinations to understand the space
        2. **Bayesian Learning**: Uses results to predict which parameters might work better
        3. **Focused Search**: Concentrates on promising parameter ranges
        4. **Early Stopping**: Stops poor trials early to save time
        5. **Quality Evaluation**: Measures each trial's performance automatically

        ### Choosing Number of Trials
        - **20 trials**: Quick optimization (4-8 hours) - good for testing
        - **50 trials**: Standard optimization (10-20 hours) - recommended
        - **100+ trials**: Maximum optimization (20-40 hours) - diminishing returns

        ### Understanding the Results
        - **Best Score**: Quality metric (0-1) - higher is better
        - **Parameter Importance**: Which settings had the most impact
        - **Optimization History**: How quality improved over trials
        - **Best Config**: Ready-to-use settings for production training

        ### Best Practices
        - **Start with a Good Dataset**: Optimization amplifies both good and bad data
        - **Use Representative Data**: Your test dataset should match your full dataset
        - **Monitor Progress**: Check that scores are improving over time
        - **Balance Time vs. Quality**: More trials = better results but longer time
        - **Save Results**: Keep the best_config.yaml for production training

        ### Common Optimization Issues
        - **All Trials Fail**: Check dataset quality and GPU memory
        - **Poor Final Scores**: Dataset may be too small or inconsistent
        - **No Improvement**: Try wider parameter ranges or more trials
        - **GPU Memory Errors**: Reduce batch sizes in optimization config

        ### After Optimization
        1. **Review Best Parameters**: Check the optimization_results.json
        2. **Use Best Config**: Train with the optimized settings
        3. **Fine-tune if Needed**: The optimized settings are a great starting point
        4. **Compare Results**: Test the optimized LoRA vs. non-optimized versions
        """

    @staticmethod
    def get_workflow_guidance() -> str:
        """Get step-by-step workflow guidance."""
        return """
        ## 🚀 Complete LoRA Training Workflow

        ### Phase 1: Preparation (10-30 minutes)
        1. **Collect Images**: Gather 10-50 high-quality images of your subject
        2. **Write Captions**: Create detailed .txt files describing each image
        3. **Organize Dataset**: Put images and captions in one directory
        4. **Validate**: Use Dataset Tools to check for issues

        ### Phase 2: Training (1-4 hours)
        1. **Choose Preset**: Character/Style/Concept based on your subject
        2. **Upload Dataset**: ZIP your images and captions
        3. **Start Training**: Monitor progress and validation samples
        4. **Watch Progress**: Loss should decrease steadily

        ### Phase 3: Evaluation (15-30 minutes)
        1. **Load Checkpoint**: Use a checkpoint from training (try multiple)
        2. **Test Prompts**: Try various prompts with your trigger word
        3. **Quality Assessment**: Run automated metrics
        4. **Compare Checkpoints**: Find the best training stage

        ### Phase 4: Iteration (if needed)
        1. **Analyze Results**: What works well? What needs improvement?
        2. **Refine Dataset**: Add more images, improve captions, fix issues
        3. **Adjust Settings**: Modify training parameters based on results
        4. **Train Again**: Use insights to improve your LoRA

        ### Phase 5: Optimization (Optional but Recommended)
        1. **Run Hyperparameter Optimization**: Find best settings for your dataset
        2. **Use Optimized Config**: Train final LoRA with optimized parameters
        3. **Compare Results**: Verify optimization improved quality
        4. **Fine-tune if Needed**: Make small adjustments based on results

        ### Phase 6: Production Use
        1. **Save Best Checkpoint**: Keep your highest-quality LoRA
        2. **Document Usage**: Note trigger words and optimal settings
        3. **Test Edge Cases**: Try unusual prompts to understand limitations
        4. **Share or Deploy**: Use your LoRA in your workflow

        ### Success Checklist
        - [ ] Dataset has 10+ high-quality, consistent images
        - [ ] Captions are detailed and descriptive (10-20 words each)
        - [ ] Training loss decreases steadily over time
        - [ ] Validation samples show clear learning progress
        - [ ] Quality metrics show good CLIP scores (>0.7)
        - [ ] Generated images match your subject/style reliably
        - [ ] Hyperparameter optimization completed (optional but recommended)
        """

    @staticmethod
    def get_training_tooltips() -> Dict[str, str]:
        """Get tooltips for training tab components."""
        return {
            "dataset_source": "Choose how to provide your training dataset - upload a ZIP file or specify a local directory path",
            "dataset_upload": "Upload a ZIP file containing your training images and captions. Images should be high-resolution (1024x1024+) and captions should describe your subject in detail",
            "dataset_dir": "Absolute path to your dataset directory on the local filesystem. Should contain images and corresponding caption files",
            "dataset_status": "Shows validation results for your selected dataset including image count and any issues found",
            "preset": "Choose a training preset optimized for your use case:\n• Character: Best for training specific characters or people\n• Style: Best for artistic styles or painting techniques\n• Concept: Best for objects, animals, or abstract concepts",
            "rank": "LoRA rank - higher values (64-128) give more capacity but train slower and use more memory. Start with 16-32 for most cases",
            "alpha": "LoRA alpha scaling factor. Usually set equal to rank, controls the strength of LoRA adaptation",
            "learning_rate": "How fast the model learns. Higher values (1e-4) learn faster but may be unstable. Lower values (1e-5) are more stable but slower",
            "max_steps": "Total number of training steps. More steps generally improve quality but take longer. 1000-2000 steps is typical",
            "batch_size": "Number of images processed simultaneously. Larger batches train faster but use more GPU memory",
            "start_training": "Begin training with current configuration. Training will run in the background and you can monitor progress",
            "stop_training": "Gracefully stop the current training session. Progress will be saved",
            "pause_resume": "Temporarily pause training or resume from pause. Useful for checking results mid-training",
            "progress_bar": "Visual progress indicator showing training completion percentage",
            "step_info": "Current training step and total steps (e.g., '500 / 1000')",
            "loss_plot": "Shows how training loss decreases over time. Steady downward trend indicates healthy training",
            "validation_gallery": "Sample images generated during training to preview your LoRA's capabilities",
            "training_logs": "Detailed training logs including loss values, learning rate, and system information",
        }

    @staticmethod
    def get_evaluation_tooltips() -> Dict[str, str]:
        """Get tooltips for evaluation tab components."""
        return {
            "checkpoint_upload": "Upload your trained LoRA checkpoint file (.safetensors format only for security)",
            "checkpoint_path": "Local path to your LoRA checkpoint file",
            "prompt_input": "Test prompt for your LoRA. Include trigger words and be descriptive",
            "concept_detection": "Automatically detect the concept/trigger word from your checkpoint filename",
            "trigger_word": "The special word that activates your LoRA (e.g., 'sks' or character name)",
            "sample_generation": "Generate sample images using your checkpoint with the test prompt",
            "quality_assessment": "Run comprehensive quality metrics including CLIP scores and overfitting detection",
            "checkpoint_comparison": "Compare multiple checkpoints side-by-side with the same prompts",
            "best_checkpoint_selection": "Automatically select the best checkpoint based on quality metrics",
        }

    @staticmethod
    def get_dataset_tooltips() -> Dict[str, str]:
        """Get tooltips for dataset tools tab components."""
        return {
            "dataset_upload": "Upload a ZIP file containing your dataset for analysis",
            "dataset_path": "Local path to dataset directory for analysis",
            "analyze_dataset": "Run comprehensive dataset analysis including statistics, quality checks, and recommendations",
            "validate_dataset": "Check for common dataset issues and get actionable fixes",
            "image_browser": "Browse through your dataset images with navigation and caption display",
            "statistics_view": "View detailed statistics about image resolutions, caption lengths, and quality metrics",
            "validation_report": "Detailed report of dataset issues with severity levels and fix recommendations",
        }

    @staticmethod
    def get_training_help_text() -> str:
        """Get detailed help text for the training tab."""
        return """
        ## 🎯 Training Your First LoRA

        ### What is LoRA Training?
        LoRA (Low-Rank Adaptation) is a technique that trains small adapter layers on top of a large pre-trained model (Flux2-dev).
        This allows you to customize the model's behavior for specific subjects, styles, or concepts without retraining the entire model.

        ### Step 1: Prepare Your Dataset
        - **Upload ZIP**: Create a ZIP file with your images and captions
        - **File Structure**: Each image should have a corresponding .txt file with the same name
        - **Image Quality**: Use high-resolution images (1024x1024 minimum)
        - **Caption Style**: Write detailed descriptions of your subject
        - **Quantity**: Start with 10-50 high-quality images

        ### Step 2: Choose Training Settings
        - **Character LoRA**: For training specific people, characters, or creatures
        - **Style LoRA**: For artistic styles, painting techniques, or visual aesthetics
        - **Concept LoRA**: For objects, animals, scenes, or abstract concepts

        ### Step 3: Advanced Settings (Optional)
        - **LoRA Rank**: Start with 16-32, increase for complex subjects
        - **Learning Rate**: Use preset defaults (1e-4), adjust if training is unstable
        - **Training Steps**: 1000-2000 steps typical, more for complex subjects
        - **Batch Size**: Reduce if you get GPU memory errors

        ### Step 4: Monitor Training
        - **Loss Plot**: Should steadily decrease over time
        - **Validation Samples**: Preview your LoRA's capabilities
        - **Progress**: Monitor completion percentage and estimated time

        ### Tips for Success
        - Train on a variety of poses/angles for characters
        - Use consistent lighting and style in your dataset
        - Include trigger words in your captions
        - Don't over-train (watch for overfitting in validation samples)
        - Save checkpoints regularly for testing

        ### Common Issues
        - **GPU Memory Error**: Reduce batch size or LoRA rank
        - **Loss Not Decreasing**: Check dataset quality or increase learning rate
        - **Poor Results**: Verify captions match your intended subject
        """

    @staticmethod
    def get_evaluation_help_text() -> str:
        """Get detailed help text for the evaluation tab."""
        return """
        ## 🔍 Evaluating Your LoRA

        ### What is LoRA Evaluation?
        After training, you need to test your LoRA to ensure it works correctly and produces the desired results.
        Evaluation helps you understand your LoRA's strengths, weaknesses, and optimal use cases.

        ### Testing Your Checkpoint
        1. **Load Checkpoint**: Upload or specify path to your LoRA checkpoint (.safetensors file)
        2. **Enter Prompt**: Write a descriptive prompt including your trigger word
        3. **Generate Samples**: Create multiple images to test consistency
        4. **Review Results**: Check if images match your expectations

        ### Quality Assessment
        - **CLIP Score**: How well images match your prompts (0-1, higher is better)
        - **Diversity Score**: Variety in generated images (higher is better)
        - **Overfitting Check**: Similarity to training images (lower risk is better)

        ### Checkpoint Comparison
        Compare multiple checkpoints to find the best one:
        1. Upload multiple checkpoints from different training stages
        2. Enter test prompts for consistent evaluation
        3. Compare results side-by-side in a grid layout
        4. Use quality metrics to choose the best version

        ### Prompt Testing Suite
        Run comprehensive tests to understand your LoRA's capabilities:
        - **Basic Tests**: Simple prompts to check fundamental functionality
        - **Positioning Tests**: How well trigger words work in different positions
        - **Composition Tests**: Complex scenes and interactions
        - **Negative Tests**: What your LoRA struggles with

        ### Best Practices
        - Test with various prompts and styles
        - Compare checkpoints from different training stages (every 100-200 steps)
        - Use quality metrics to guide your decisions
        - Save the best checkpoint for production use
        - Test edge cases and unusual prompts

        ### Interpreting Results
        - **High CLIP Score (>0.8)**: Excellent prompt adherence
        - **High Diversity (>0.7)**: Model isn't repetitive
        - **Low Overfitting Risk (<0.8)**: Model generalizes well
        - **Consistent Quality**: Reliable results across different prompts
        - **Good Prompt Adherence**: Trigger words work reliably in different contexts
        """

    @staticmethod
    def get_dataset_help_text() -> str:
        """Get detailed help text for the dataset tools tab."""
        return """
        ## 📊 Dataset Analysis & Validation

        ### Why Dataset Quality Matters
        Your dataset is the foundation of your LoRA training. Poor quality datasets lead to poor results.
        The dataset tools help you understand and improve your training data before starting training.

        ### Dataset Requirements
        - **Image Formats**: JPG, PNG, WebP supported (RGB, consistent resolution)
        - **Caption Files**: .txt files with same name as images (UTF-8 encoded)
        - **Resolution**: 1024x1024 minimum recommended (higher for detailed subjects)
        - **Consistency**: Maintain similar style, lighting, and quality across images
        - **Quantity**: Start with 10-50 images, more for complex subjects

        ### Analysis Features
        - **Statistics Dashboard**: Image resolutions, file sizes, aspect ratios, color spaces
        - **Caption Analysis**: Length distribution, vocabulary size, word frequency
        - **Quality Checks**: Missing files, format issues, corrupted images
        - **Validation Report**: Comprehensive assessment with severity levels
        - **Image Browser**: Visual inspection of your dataset with navigation

        ### Common Issues & Fixes
        - **Missing Captions**: Every image needs a .txt file with the same name
        - **Inconsistent Naming**: Use sequential naming (image_001.jpg, image_001.txt)
        - **Mixed Resolutions**: Resize images to consistent dimensions (1024x1024 recommended)
        - **Poor Captions**: Write detailed, descriptive captions (aim for 10-20 words)
        - **Low Quantity**: Add more diverse examples of your subject
        - **Inconsistent Style**: Remove images that don't match your desired aesthetic

        ### Optimization Tips
        - Use high-quality, high-resolution images from your subject
        - Write detailed captions describing visual features and context
        - Include various angles/poses for characters and objects
        - Maintain consistent art style, lighting, and composition
        - Remove outliers, duplicates, or low-quality images
        - Balance your dataset (avoid over-representing certain poses/styles)

        ### Validation Checks Performed
        - ✅ File pairing verification (image ↔ caption matching)
        - ✅ Image format and corruption detection
        - ✅ Caption length and quality assessment
        - ✅ Resolution consistency analysis
        - ✅ File size distribution review
        - ✅ Encoding and format validation
        - ✅ Vocabulary diversity measurement

        ### Using the Image Browser
        The image browser helps you visually inspect your dataset:
        - Navigate through images with previous/next buttons
        - Jump to specific images by index
        - Random sampling for quality checks
        - View captions alongside images
        - Identify outliers or problematic images
        """


# Global help system instance
help_system = HelpSystem()

"""
Prompt testing suite for comprehensive LoRA evaluation.

This module provides systematic prompt testing capabilities to evaluate LoRA
performance across different prompt complexities, trigger word positions,
and composition scenarios.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..core.model_loader import ModelLoader
from ..core.lora_config import FluxLoRAConfig
from .quality_metrics import QualityAssessor

logger = logging.getLogger(__name__)


@dataclass
class PromptTest:
    """Single prompt test case with metadata."""

    name: str
    prompt: str
    expected_features: List[str]
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "basic", "positioning", "composition", "complexity", "negative"
    trigger_position: Optional[str] = None  # "start", "middle", "end", "multiple"
    description: Optional[str] = None

    def __post_init__(self):
        """Validate test case data."""
        if self.difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid difficulty: {self.difficulty}")

        if self.category not in ["basic", "positioning", "composition", "complexity", "negative"]:
            raise ValueError(f"Invalid category: {self.category}")

        if self.trigger_position and self.trigger_position not in [
            "start",
            "middle",
            "end",
            "multiple",
        ]:
            raise ValueError(f"Invalid trigger position: {self.trigger_position}")


@dataclass
class PromptTestResult:
    """Result of a single prompt test."""

    test: PromptTest
    generated_images: List[Image.Image]
    clip_score: float
    diversity_score: float
    adherence_score: float
    success_rating: str  # "excellent", "good", "fair", "poor"
    notes: Optional[str] = None

    @property
    def score(self) -> float:
        """Overall score combining multiple metrics."""
        # Weight the different metrics
        weights = {
            "clip_score": 0.4,
            "diversity_score": 0.3,
            "adherence_score": 0.3,
        }

        return (
            weights["clip_score"] * self.clip_score
            + weights["diversity_score"] * self.diversity_score
            + weights["adherence_score"] * self.adherence_score
        )


class PromptTester:
    """
    Comprehensive prompt testing system for LoRA evaluation.

    Tests LoRA models across different prompt complexities, trigger word positioning,
    composition scenarios, and negative prompt handling to identify strengths
    and weaknesses.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        trigger_word: Optional[str] = None,
    ):
        """
        Initialize prompt tester.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            device: Device for inference ('auto', 'cpu', 'cuda')
            trigger_word: Optional trigger word to test positioning
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.trigger_word = trigger_word or self._extract_trigger_from_checkpoint()

        # Initialize components
        self.quality_assessor = QualityAssessor(device=device)
        self.model_loader = ModelLoader()
        self.model = None

        # Test configuration
        self.num_inference_steps = 25
        self.guidance_scale = 7.5
        self.num_samples_per_prompt = 3
        self.seed = 42

        logger.info(f"PromptTester initialized for checkpoint: {self.checkpoint_path.name}")
        logger.info(f"Trigger word: {self.trigger_word}")

    def _extract_trigger_from_checkpoint(self) -> str:
        """Extract trigger word from checkpoint name."""
        # Simple heuristic: use checkpoint name as trigger word
        return self.checkpoint_path.stem.lower().replace("_", " ").replace("-", " ")

    def load_checkpoint(self) -> None:
        """Load the LoRA checkpoint."""
        if self.model is not None:
            return  # Already loaded

        logger.info(f"Loading checkpoint: {self.checkpoint_path}")

        # Load base model
        model, _ = self.model_loader.load_flux2_dev(device=self.quality_assessor.device)

        # Load LoRA weights
        try:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(model, str(self.checkpoint_path))
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def create_test_suite(self, concept: Optional[str] = None) -> List[PromptTest]:
        """
        Generate comprehensive test prompt suite.

        Args:
            concept: Optional concept name (uses trigger word if None)

        Returns:
            List of PromptTest cases
        """
        concept = concept or self.trigger_word

        test_suite = []

        # Basic tests - simple prompts
        test_suite.extend(self._create_basic_tests(concept))

        # Positioning tests - trigger word at different positions
        test_suite.extend(self._create_positioning_tests(concept))

        # Composition tests - multiple concepts/styles
        test_suite.extend(self._create_composition_tests(concept))

        # Complexity tests - varying prompt lengths and details
        test_suite.extend(self._create_complexity_tests(concept))

        # Negative prompt tests
        test_suite.extend(self._create_negative_tests(concept))

        logger.info(f"Created test suite with {len(test_suite)} prompts")
        return test_suite

    def _create_basic_tests(self, concept: str) -> List[PromptTest]:
        """Create basic prompt tests."""
        return [
            PromptTest(
                name="basic_standalone",
                prompt=f"{concept}",
                expected_features=[f"Basic {concept} concept"],
                difficulty="easy",
                category="basic",
                trigger_position="start",
                description="Simple standalone concept",
            ),
            PromptTest(
                name="basic_with_article",
                prompt=f"A {concept}",
                expected_features=[f"Basic {concept} concept"],
                difficulty="easy",
                category="basic",
                trigger_position="end",
                description="Concept with article",
            ),
            PromptTest(
                name="basic_plural",
                prompt=f"{concept}s",
                expected_features=[f"Multiple {concept} concepts"],
                difficulty="easy",
                category="basic",
                trigger_position="start",
                description="Plural form of concept",
            ),
        ]

    def _create_positioning_tests(self, concept: str) -> List[PromptTest]:
        """Create trigger word positioning tests."""
        return [
            PromptTest(
                name="position_start",
                prompt=f"{concept} in a beautiful landscape",
                expected_features=[f"{concept} concept", "landscape setting"],
                difficulty="medium",
                category="positioning",
                trigger_position="start",
                description="Trigger word at beginning",
            ),
            PromptTest(
                name="position_middle",
                prompt=f"A beautiful {concept} in the style of impressionism",
                expected_features=[f"{concept} concept", "impressionist style"],
                difficulty="medium",
                category="positioning",
                trigger_position="middle",
                description="Trigger word in middle",
            ),
            PromptTest(
                name="position_end",
                prompt=f"Highly detailed artwork featuring {concept}",
                expected_features=[f"{concept} concept", "high detail", "artwork style"],
                difficulty="medium",
                category="positioning",
                trigger_position="end",
                description="Trigger word at end",
            ),
            PromptTest(
                name="position_multiple",
                prompt=f"{concept} and another {concept} together",
                expected_features=[f"Multiple {concept} concepts", "composition"],
                difficulty="hard",
                category="positioning",
                trigger_position="multiple",
                description="Multiple trigger words",
            ),
        ]

    def _create_composition_tests(self, concept: str) -> List[PromptTest]:
        """Create composition and multi-concept tests."""
        return [
            PromptTest(
                name="composition_with_animal",
                prompt=f"{concept} riding a horse in the desert",
                expected_features=[f"{concept} concept", "horse", "desert setting", "action"],
                difficulty="hard",
                category="composition",
                trigger_position="start",
                description="Concept in action with other elements",
            ),
            PromptTest(
                name="composition_with_objects",
                prompt=f"{concept} holding a sword and shield",
                expected_features=[f"{concept} concept", "sword", "shield", "action pose"],
                difficulty="hard",
                category="composition",
                trigger_position="start",
                description="Concept with multiple objects",
            ),
            PromptTest(
                name="composition_emotional",
                prompt=f"Happy {concept} celebrating victory",
                expected_features=[f"{concept} concept", "happy expression", "victory celebration"],
                difficulty="medium",
                category="composition",
                trigger_position="middle",
                description="Concept with emotional context",
            ),
        ]

    def _create_complexity_tests(self, concept: str) -> List[PromptTest]:
        """Create varying complexity tests."""
        return [
            PromptTest(
                name="complexity_detailed",
                prompt=f"Highly detailed {concept} with intricate patterns, professional digital art, sharp focus, illustration, complex background",
                expected_features=[
                    f"{concept} concept",
                    "high detail",
                    "intricate patterns",
                    "digital art",
                    "complex background",
                ],
                difficulty="hard",
                category="complexity",
                trigger_position="middle",
                description="Highly detailed complex prompt",
            ),
            PromptTest(
                name="complexity_minimal",
                prompt=f"{concept}, simple",
                expected_features=[f"{concept} concept", "simple style"],
                difficulty="easy",
                category="complexity",
                trigger_position="start",
                description="Minimal prompt with concept",
            ),
            PromptTest(
                name="complexity_stylized",
                prompt=f"{concept} in the style of Picasso, cubist art, geometric shapes, abstract representation",
                expected_features=[
                    f"{concept} concept",
                    "Picasso style",
                    "cubism",
                    "geometric",
                    "abstract",
                ],
                difficulty="hard",
                category="complexity",
                trigger_position="start",
                description="Artistic style transformation",
            ),
        ]

    def _create_negative_tests(self, concept: str) -> List[PromptTest]:
        """Create negative prompt tests."""
        return [
            PromptTest(
                name="negative_blurry",
                prompt=f"Sharp focus {concept}, no blur, clear details",
                expected_features=[f"{concept} concept", "sharp focus", "clear details"],
                difficulty="medium",
                category="negative",
                trigger_position="middle",
                description="Negative prompt for blur avoidance",
            ),
            PromptTest(
                name="negative_distorted",
                prompt=f"{concept} with perfect anatomy, no deformities",
                expected_features=[f"{concept} concept", "perfect anatomy"],
                difficulty="medium",
                category="negative",
                trigger_position="start",
                description="Negative prompt for anatomical correctness",
            ),
        ]

    def run_test_suite(
        self,
        test_suite: List[PromptTest],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run complete test suite and collect results.

        Args:
            test_suite: List of PromptTest cases to run
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with test results and analysis
        """
        logger.info(f"Running test suite with {len(test_suite)} prompts")

        self.load_checkpoint()
        results = []

        for i, test_case in enumerate(test_suite):
            logger.info(f"Running test {i + 1}/{len(test_suite)}: {test_case.name}")

            try:
                result = self._run_single_test(test_case)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(test_suite), result)

            except Exception as e:
                logger.error(f"Failed to run test {test_case.name}: {e}")
                # Create failed result
                failed_result = PromptTestResult(
                    test=test_case,
                    generated_images=[],
                    clip_score=0.0,
                    diversity_score=0.0,
                    adherence_score=0.0,
                    success_rating="poor",
                    notes=f"Test failed: {str(e)}",
                )
                results.append(failed_result)

        # Analyze results
        analysis = self._analyze_test_results(results)

        return {
            "results": results,
            "analysis": analysis,
            "summary": self._create_summary(results),
            "test_suite_info": {
                "total_tests": len(test_suite),
                "completed_tests": len(results),
                "trigger_word": self.trigger_word,
                "checkpoint": str(self.checkpoint_path),
            },
        }

    def _run_single_test(self, test_case: PromptTest) -> PromptTestResult:
        """Run a single prompt test."""
        # Generate images
        with torch.no_grad():
            images = self.model(
                prompt=test_case.prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_samples_per_prompt,
                height=1024,
                width=1024,
            ).images

        # Assess quality
        assessment = self.quality_assessor.assess_checkpoint_quality(
            checkpoint_path=self.checkpoint_path,
            test_prompts=[test_case.prompt],
            num_samples_per_prompt=self.num_samples_per_prompt,
            generation_kwargs={
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            },
        )

        # Extract metrics
        clip_score = assessment.get("clip_score", 0.0)
        diversity_score = assessment.get("diversity_score", 0.0)

        # Calculate adherence score (how well prompt features are captured)
        adherence_score = self._calculate_adherence_score(test_case, assessment)

        # Determine success rating
        success_rating = self._determine_success_rating(
            clip_score, diversity_score, adherence_score
        )

        return PromptTestResult(
            test=test_case,
            generated_images=images,
            clip_score=clip_score,
            diversity_score=diversity_score,
            adherence_score=adherence_score,
            success_rating=success_rating,
        )

    def _calculate_adherence_score(
        self, test_case: PromptTest, assessment: Dict[str, Any]
    ) -> float:
        """Calculate how well the generated images adhere to expected features."""
        # Simple heuristic: use CLIP score as proxy for adherence
        # In a more sophisticated implementation, this could analyze specific features
        base_score = assessment.get("clip_score", 0.0)

        # Adjust based on prompt complexity
        if test_case.difficulty == "easy":
            return min(base_score * 1.1, 1.0)  # Slight bonus for easy prompts
        elif test_case.difficulty == "hard":
            return base_score * 0.9  # Penalty for hard prompts
        else:
            return base_score

    def _determine_success_rating(
        self, clip_score: float, diversity_score: float, adherence_score: float
    ) -> str:
        """Determine success rating based on metrics."""
        avg_score = (clip_score + diversity_score + adherence_score) / 3

        if avg_score >= 0.8:
            return "excellent"
        elif avg_score >= 0.6:
            return "good"
        elif avg_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _analyze_test_results(self, results: List[PromptTestResult]) -> Dict[str, Any]:
        """Analyze test results to identify patterns and insights."""
        analysis = {
            "overall_performance": {},
            "category_performance": {},
            "difficulty_performance": {},
            "positioning_performance": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        }

        if not results:
            return analysis

        # Overall performance
        scores = [r.score for r in results]
        analysis["overall_performance"] = {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": self._create_score_distribution(scores),
        }

        # Category performance
        categories = {}
        for result in results:
            cat = result.test.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result.score)

        analysis["category_performance"] = {
            cat: {
                "average_score": sum(scores) / len(scores),
                "test_count": len(scores),
            }
            for cat, scores in categories.items()
        }

        # Difficulty performance
        difficulties = {}
        for result in results:
            diff = result.test.difficulty
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result.score)

        analysis["difficulty_performance"] = {
            diff: {
                "average_score": sum(scores) / len(scores),
                "test_count": len(scores),
            }
            for diff, scores in difficulties.items()
        }

        # Positioning performance
        positions = {}
        for result in results:
            pos = result.test.trigger_position
            if pos and pos not in positions:
                positions[pos] = []
            if pos:
                positions[pos].append(result.score)

        analysis["positioning_performance"] = {
            pos: {
                "average_score": sum(scores) / len(scores),
                "test_count": len(scores),
            }
            for pos, scores in positions.items()
        }

        # Identify strengths and weaknesses
        category_scores = analysis["category_performance"]
        analysis["strengths"] = [
            cat for cat, perf in category_scores.items() if perf["average_score"] >= 0.7
        ]
        analysis["weaknesses"] = [
            cat for cat, perf in category_scores.items() if perf["average_score"] < 0.5
        ]

        # Generate recommendations
        if analysis["weaknesses"]:
            analysis["recommendations"].append(
                f"Focus on improving performance in: {', '.join(analysis['weaknesses'])}"
            )

        diff_perf = analysis["difficulty_performance"]
        if diff_perf.get("hard", {}).get("average_score", 0) < 0.6:
            analysis["recommendations"].append(
                "Complex prompts perform poorly - consider fine-tuning with more diverse training data"
            )

        pos_perf = analysis["positioning_performance"]
        if pos_perf.get("end", {}).get("average_score", 0) < pos_perf.get("start", {}).get(
            "average_score", 1
        ):
            analysis["recommendations"].append(
                "Trigger word at end of prompt performs worse - consider emphasizing concept placement"
            )

        return analysis

    def _create_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Create score distribution for analysis."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

        for score in scores:
            if score >= 0.8:
                distribution["excellent"] += 1
            elif score >= 0.6:
                distribution["good"] += 1
            elif score >= 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1

        return distribution

    def _create_summary(self, results: List[PromptTestResult]) -> Dict[str, Any]:
        """Create summary statistics."""
        if not results:
            return {}

        total_tests = len(results)
        successful_tests = len([r for r in results if r.success_rating in ["excellent", "good"]])

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_score": sum(r.score for r in results) / total_tests,
            "best_test": max(results, key=lambda r: r.score).test.name,
            "worst_test": min(results, key=lambda r: r.score).test.name,
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        format: str = "markdown",
    ) -> str:
        """
        Generate comprehensive test report.

        Args:
            results: Results from run_test_suite
            output_path: Optional path to save report
            format: Report format ('markdown' or 'html')

        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
            output_path = f"./prompt_test_report_{timestamp}.{format}"

        output_path = Path(output_path)

        if format == "markdown":
            content = self._generate_markdown_report(results)
        elif format == "html":
            content = self._generate_html_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Report generated: {output_path}")
        return str(output_path)

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report."""
        test_results = results["results"]
        analysis = results["analysis"]
        summary = results["summary"]
        info = results["test_suite_info"]

        report = f"""# LoRA Prompt Testing Report

## Overview

- **Checkpoint**: {info["checkpoint"]}
- **Trigger Word**: {info["trigger_word"]}
- **Total Tests**: {info["total_tests"]}
- **Completed Tests**: {info["completed_tests"]}
- **Success Rate**: {summary.get("success_rate", 0):.1%}

## Summary Statistics

- **Average Score**: {summary.get("average_score", 0):.3f}
- **Best Performing Test**: {summary.get("best_test", "N/A")}
- **Worst Performing Test**: {summary.get("worst_test", "N/A")}

## Performance by Category

| Category | Average Score | Test Count |
|----------|---------------|------------|
"""

        for cat, perf in analysis.get("category_performance", {}).items():
            report += f"| {cat} | {perf['average_score']:.3f} | {perf['test_count']} |\n"

        report += "\n## Performance by Difficulty\n\n"
        report += "| Difficulty | Average Score | Test Count |\n"
        report += "|------------|---------------|------------|\n"

        for diff, perf in analysis.get("difficulty_performance", {}).items():
            report += f"| {diff} | {perf['average_score']:.3f} | {perf['test_count']} |\n"

        report += "\n## Performance by Trigger Position\n\n"
        report += "| Position | Average Score | Test Count |\n"
        report += "|----------|---------------|------------|\n"

        for pos, perf in analysis.get("positioning_performance", {}).items():
            report += f"| {pos} | {perf['average_score']:.3f} | {perf['test_count']} |\n"

        report += "\n## Detailed Test Results\n\n"
        report += "| Test Name | Category | Difficulty | Score | Rating | Notes |\n"
        report += "|-----------|----------|------------|-------|--------|-------|\n"

        for result in test_results:
            notes = result.notes or ""
            report += f"| {result.test.name} | {result.test.category} | {result.test.difficulty} | {result.score:.3f} | {result.success_rating} | {notes} |\n"

        report += "\n## Analysis Insights\n\n"

        if analysis.get("strengths"):
            report += (
                f"### Strengths\n- {chr(10).join('- ' + s for s in analysis['strengths'])}\n\n"
            )

        if analysis.get("weaknesses"):
            report += f"### Areas for Improvement\n- {chr(10).join('- ' + w for w in analysis['weaknesses'])}\n\n"

        if analysis.get("recommendations"):
            report += f"### Recommendations\n- {chr(10).join('- ' + r for r in analysis['recommendations'])}\n\n"

        return report

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report (simplified version)."""
        summary = results["summary"]
        info = results["test_suite_info"]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LoRA Prompt Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>LoRA Prompt Testing Report</h1>

            <h2>Overview</h2>
            <p><strong>Checkpoint:</strong> {info["checkpoint"]}</p>
            <p><strong>Trigger Word:</strong> {info["trigger_word"]}</p>
            <p><strong>Success Rate:</strong> {summary.get("success_rate", 0):.1%}</p>
            <p><strong>Average Score:</strong> {summary.get("average_score", 0):.3f}</p>

            <h2>Test Results Summary</h2>
            <p>Report generated with comprehensive prompt testing suite.</p>
        </body>
        </html>
        """

        return html

from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score, AggScore
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.utils import strip_thinking_blocks
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='ifeval-th',
        pretty_name='IFEval-Thai',
        description='IFEval Thai is a benchmark for evaluating instruction-following language models in Thai language, focusing on their ability to understand and respond to various prompts.',  # noqa: E501
        tags=[Tags.INSTRUCTION_FOLLOWING],
        dataset_id='scb10x/ifeval-th',
        subset_list=['default'],
        extra_params={'dataset_hub': 'huggingface'},
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='คุณคือผู้ช่วยอัจฉริยะที่ต้องปฏิบัติตามคำสั่งอย่างระมัดระวังและแม่นยำ\n\n{question}'
    )
)
class IFEvalThAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        prompt = record['prompt']

        # Apply prompt template if available
        if self.prompt_template:
            prompt = self.prompt_template.format(question=prompt)

        return Sample(
            input=prompt,  # Use string format instead of ChatMessage for better compatibility
            target=record.get('instruction_id_list', []),
            metadata={
                'instruction_id_list': record.get('instruction_id_list', []),
                'kwargs': record.get('kwargs', []),
                'prompt': record.get('prompt', ''),
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the answer from the prediction."""
        # Strip Qwen3 thinking blocks before evaluation
        return strip_thinking_blocks(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: List, task_state: TaskState
    ) -> Score:
        """
        Calculate the match score for IFEval.

        Args:
            original_prediction (str): The original model prediction.
            filtered_prediction (str): The filtered prediction.
            reference (List): The reference instruction IDs.
            task_state (TaskState): The task state.

        Returns:
            Score: The calculated score.
        """
        from .utils import InputExample, test_instruction_following_loose, test_instruction_following_strict

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Get instructions and kwargs from metadata
        instruction_id_list = task_state.metadata.get('instruction_id_list', [])
        kwargs_list = task_state.metadata.get('kwargs', [])
        prompt = task_state.metadata.get('prompt', '')

        # Create InputExample object
        inp = InputExample(
            key=0,  # Not used in evaluation
            instruction_id_list=instruction_id_list,
            prompt=prompt,
            kwargs=kwargs_list
        )

        # Evaluate strict and loose
        try:
            out_strict = test_instruction_following_strict(inp, filtered_prediction)
            out_loose = test_instruction_following_loose(inp, filtered_prediction)

            # Extract results
            strict_results = out_strict.follow_instruction_list
            loose_results = out_loose.follow_instruction_list

            # Calculate scores
            inst_level_strict = sum(strict_results) / len(strict_results) if strict_results else 0
            inst_level_loose = sum(loose_results) / len(loose_results) if loose_results else 0
            prompt_level_strict = int(all(strict_results)) if strict_results else 0
            prompt_level_loose = int(all(loose_results)) if loose_results else 0

        except Exception as e:
            logger.warning(f"Error in instruction following evaluation: {e}")
            # Default scores on error
            inst_level_strict = 0
            inst_level_loose = 0
            prompt_level_strict = 0
            prompt_level_loose = 0
            strict_results = []
            loose_results = []

        score.value = {
            'inst_level_strict': inst_level_strict,
            'inst_level_loose': inst_level_loose,
            'prompt_level_strict': prompt_level_strict,
            'prompt_level_loose': prompt_level_loose,
        }

        score.main_score_name = 'prompt_level_strict'
        score.metadata = {
            'instruction_id_list': instruction_id_list,
            'strict_results': strict_results,
            'loose_results': loose_results,
        }

        return score

    def aggregate_scores(self, sample_scores: List[Score]) -> List[AggScore]:
        """
        Aggregate scores across all samples.

        Args:
            sample_scores (List[Score]): List of scores from all samples.

        Returns:
            List[AggScore]: Aggregated metrics.
        """
        if not sample_scores:
            return []

        # Aggregate metrics
        prompt_level_strict_sum = 0
        prompt_level_loose_sum = 0
        inst_level_strict_sum = 0
        inst_level_loose_sum = 0

        for sample_score in sample_scores:
            # SampleScore has a 'score' attribute which contains the Score object
            if hasattr(sample_score, 'score') and sample_score.score and sample_score.score.value:
                prompt_level_strict_sum += sample_score.score.value.get('prompt_level_strict', 0)
                prompt_level_loose_sum += sample_score.score.value.get('prompt_level_loose', 0)
                inst_level_strict_sum += sample_score.score.value.get('inst_level_strict', 0)
                inst_level_loose_sum += sample_score.score.value.get('inst_level_loose', 0)

        n = len(sample_scores)

        return [
            AggScore(
                score=prompt_level_strict_sum / n * 100 if n > 0 else 0,
                metric_name='prompt_level_strict',
                aggregation_name='mean',
                category='default'
            ),
            AggScore(
                score=prompt_level_loose_sum / n * 100 if n > 0 else 0,
                metric_name='prompt_level_loose',
                aggregation_name='mean',
                category='default'
            ),
            AggScore(
                score=inst_level_strict_sum / n * 100 if n > 0 else 0,
                metric_name='inst_level_strict',
                aggregation_name='mean',
                category='default'
            ),
            AggScore(
                score=inst_level_loose_sum / n * 100 if n > 0 else 0,
                metric_name='inst_level_loose',
                aggregation_name='mean',
                category='default'
            )
        ]
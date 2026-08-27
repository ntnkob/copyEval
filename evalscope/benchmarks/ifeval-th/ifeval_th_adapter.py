from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
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
        dataset_id='typhoon-ai/ifeval-th',
        subset_list=['default'],
        extra_params={'dataset_hub': 'huggingface'},
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        primary_metric='inst_level_loose',
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
        prompt = record.get('prompt', '')
        message_list = [ChatMessageUser(content=prompt)]

        return Sample(input=message_list, target='', metadata=record)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the answer from the prediction."""
        # Strip Qwen3 thinking blocks before evaluation
        return strip_thinking_blocks(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.ifeval.utils import process_results

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata
        try:
            # Process results using the existing ifeval utility
            results = process_results(doc, [filtered_prediction])
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'inst_level_loose'

        except Exception as e:
            logger.error(f'Error calculating ifeval metrics: {e}')
            score.value = {}

        return score
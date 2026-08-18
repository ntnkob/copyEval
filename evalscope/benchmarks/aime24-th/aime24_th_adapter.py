# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aime24-th',
        pretty_name='AIME-2024-Thai',
        tags=[Tags.MATH, Tags.REASONING],
        description='The AIME 2024 Thai benchmark is a translation of the American Invitational Mathematics Examination problems. This benchmark tests a model\'s ability to solve challenging mathematics problems in Thai language by generating step-by-step solutions and providing the correct final answer.',  # noqa: E501
        dataset_id='iapp/aime_2024-th',
        subset_list=['default'],
        extra_params={'dataset_hub': 'huggingface'},
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Only train set is available
        prompt_template='{question}\nกรุณาแสดงวิธีคิดแบบขั้นตอนอย่างละเอียด และใส่คำตอบสุดท้ายให้ชัดเจนใน \\boxed{{}}'
    )
)
class AIME24ThAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'problem_id': record.get('id', ''),
                'solution': record.get('solution', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract answer from \boxed{} format."""
        from evalscope.metrics.math_parser import extract_answer

        # Use the standard math extraction
        return extract_answer(prediction)
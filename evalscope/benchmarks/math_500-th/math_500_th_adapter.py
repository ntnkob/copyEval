# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='math_500-th',
        pretty_name='MATH-500-Thai',
        tags=[Tags.MATH, Tags.REASONING],
        description="MATH-500 Thai is a benchmark for evaluating mathematical reasoning capabilities in Thai language. It consists of 500 diverse math problems designed to test a model's ability to solve complex mathematical problems in Thai.",  # noqa: E501
        dataset_id='iapp/math-500-th',
        subset_list=['default'],
        extra_params={'dataset_hub': 'huggingface'},
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}\nกรุณาแสดงวิธีคิดแบบขั้นตอนอย่างละเอียด และใส่คำตอบสุดท้ายให้ชัดเจนใน \\boxed{{}}'
    )
)
class Math500ThAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'question_id': record.get('unique_id', ''),
                'solution': record.get('solution', ''),
                'level': record.get('level', 1),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract answer from \boxed{} format."""
        from evalscope.metrics.math_parser import extract_answer

        # Use the standard math extraction
        return extract_answer(prediction)
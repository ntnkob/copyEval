from typing import Any, Dict

import regex as re

from evalscope.api.benchmark import (
    BenchmarkMeta,
    DefaultDataAdapter,
)

from evalscope.api.dataset import (
    Sample,
)

from evalscope.api.evaluator import (
    TaskState,
)

from evalscope.api.registry import (
    register_benchmark,
)

from evalscope.constants import (
    Tags,
)

from evalscope.utils.logger import (
    get_logger,
)


logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='thai_exam',

        pretty_name='Thai Exam Benchmark',

        tags=[Tags.MULTIPLE_CHOICE],

        description='Thai multiple choice benchmark',

        dataset_id='thai_exam',

        prompt_template="{question}",

        metric_list=['acc'],

        few_shot_num=0,

        train_split=None,

        eval_split='test',
    )
)
class ThaiExamAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ============================================
    # Load local dataset
    # ============================================

    def load_from_disk(self, **kwargs):

        return super().load_from_disk(
            use_local_loader=True
        )

    # ============================================
    # Convert dataset row -> Sample
    # ============================================

    def record_to_sample(
        self,
        record: Dict[str, Any]
    ) -> Sample:

        system = record.get(
            'system',
            ''
        )

        query = record.get(
            'query',
            ''
        )

        if system:

            prompt = (
                f"{system}\n\n"
                f"{query}"
            )

        else:

            prompt = query

        return Sample(

            input=prompt,

            target=str(
                record.get(
                    'response',
                    ''
                )
            ).strip().lower(),

            metadata={

                'domain': record.get(
                    'domain',
                    'unknown'
                ),

            },
        )

    # ============================================
    # Extract model answer
    # ============================================

    def extract_answer(
        self,
        prediction: str,
        task_state: TaskState,
    ):

        if not prediction:
            return ''

        # =====================================
        # Remove thinking blocks
        # =====================================

        prediction = re.sub(
            r'<think>.*?</think>',
            '',
            prediction,
            flags=re.DOTALL,
        )

        prediction = re.sub(
            r'<thinking>.*?</thinking>',
            '',
            prediction,
            flags=re.DOTALL,
        )

        prediction = prediction.replace(
            "<think>",
            ""
        ).replace(
            "</think>",
            ""
        )

        prediction = prediction.replace(
            "<thinking>",
            ""
        ).replace(
            "</thinking>",
            ""
        )

        prediction = prediction.strip()

        # =====================================
        # ANSWER / คำตอบ
        # =====================================

        VALID_ANSWERS = {
            'a', 'b', 'c', 'd', 'e',
            'ก', 'ข', 'ค', 'ง', 'จ',
            '1', '2', '3', '4', '5',
        }

        match = re.search(
            r'(?i)(?:ANSWER|คำตอบ)(?:คือ)?\s*:?\s*([A-Za-z\d฀-๿]+)',
            prediction,
        )

        if match:

            ans = (
                match.group(1)
                .replace('(', '')
                .replace(')', '')
                .replace('.', '')
                .strip()
                .lower()
            )

            if ans in VALID_ANSWERS:
                return ans

        # =====================================
        # Numeric fallback
        # =====================================

        nums = re.findall(
            r'\b\d+\b',
            prediction,
        )

        if nums:
            return nums[-1]

        # =====================================
        # Character fallback
        # =====================================

        allowed = set()

        # English
        allowed.update(
            ['a', 'b', 'c', 'd', 'e']
        )

        # Thai
        allowed.update(
            ['ก', 'ข', 'ค', 'ง', 'จ']
        )

        # Numeric
        allowed.update(
            ['1', '2', '3', '4', '5']
        )

        for ch in reversed(prediction):

            ch = ch.lower()

            if ch in allowed:
                return ch

        return ''

#cp -rf thai_exam ~/.conda/envs/test-2/lib/python3.10/site-packages/evalscope/benchmarks/

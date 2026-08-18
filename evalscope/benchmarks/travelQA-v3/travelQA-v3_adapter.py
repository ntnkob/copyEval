# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

@register_benchmark(
    BenchmarkMeta(
        name='travelQA-v3',
        pretty_name='Travel QA (v3)',
        tags=[Tags.MULTIPLE_CHOICE],
        description="Travel QA (v3)",  # noqa: E501
        dataset_id='ThaiLLM-Dev/TravelLLM-MCQ-Eval',
        metric_list=['acc'],
        subset_list=['v3'], # Have to include this
        few_shot_num=0,
        train_split=None,
        eval_split = 'test', # Specify "test" split
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER
        )
)
class TravelQA_v3_Adapter(MultiChoiceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] # Can exaggerate, break in extraction anyways

    def record_to_sample(self, record) -> Sample:
        # Extract choices from the record (A, B, C, D, etc.) (taken from GeneralQA)
        choices = []
        for choice_key in self.choices:
            if choice_key in record:
                choices.append(record[choice_key])
            else:
                break  # Stop when we reach a choice key that doesn't exist

        return Sample(
            input=record['question'],
            choices=choices,
            target=(record['answer']),
            metadata={'dataset_id': record['id']},
        )

    # def extract_answer(self, prediction: str, task_state: TaskState):
    #     """Extract answer from model prediction"""
    #     from evalscope.filters.extraction import RegexFilter
        
    #     # Use regular expression to extract numeric answer
    #     regex = RegexFilter(regex_pattern=r'คำตอบ(?:คือ)?:?\s?\\?n?\(?([1-9a-zA-Zก-ฮ])\.?\)?', group_select=-1)
    #     res = regex(prediction)
    #     return res.replace('(', '').replace(')', '').replace('.','').strip()
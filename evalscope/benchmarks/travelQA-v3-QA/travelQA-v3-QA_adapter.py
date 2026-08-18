# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.metric import Score
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from typing import Any, Dict

logger = get_logger()
PROMPT_TEMPLATE = '{question}'

@register_benchmark(
    BenchmarkMeta(
        name='travelQA-v3-QA',
        pretty_name='Travel QA (v3) QA mode',
        tags=[Tags.MULTIPLE_CHOICE],
        description="Travel QA (v3) QA mode",  # noqa: E501
        dataset_id='ThaiLLM-Dev/TravelLLM-MCQ-Eval',
        metric_list=['Rouge'],
        subset_list=['v3'], # Have to include this
        few_shot_num=0,
        train_split=None,
        eval_split = 'test', # Specify "test" split
        prompt_template=PROMPT_TEMPLATE
        )
)

class travel_QA_v3_QA_Adapter(DefaultDataAdapter):

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

        return Sample(
        input=record.get('question', ''),
        target=record[record['answer']],
        metadata={
            'dataset_index': record.get('index', '')
            },
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Calculate scores for each configured metric
        for metric in self.metric_list:
            try:
                if metric == 'Rouge':
                    from evalscope.metrics.rouge_metric import compute_rouge_score_one_sample_zh

                    score.value.update(compute_rouge_score_one_sample_zh([filtered_prediction], [reference]))
                elif metric == 'BLEU':
                    from evalscope.metrics import bleu_ngram_one_sample

                    score.value.update(bleu_ngram_one_sample(filtered_prediction, reference))
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                return None

        score.main_score_name = 'Rouge-L-R'
        return score
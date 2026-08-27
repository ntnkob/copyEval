# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.api.metric import Score
from evalscope.api.messages import ChatMessageUser, ContentText


# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='machine_translation',
        tags=[Tags.CUSTOM],
        description='English-to-Thai translation quality estimation using COMET-Kiwi-MEET-MR.',
        dataset_id='machine_translation',
        subset_list=['default'],
        metric_list=['comet_kiwi_meet_mr'],
        few_shot_num=0,
        prompt_template="{question}",
        train_split=None,
    )
)

class MachineTranslationAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_batch_scoring=True

    def load_from_disk(self, **kwargs):

        return super().load_from_disk(
            use_local_loader=True
        )


    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        src = record["src"]
    
        query = record.get("query")
        if query is None:
            query = f"Translate the following text to Thai.\n{src}"

        content_list = [ContentText(text = query)]

        return Sample(
            input = [ChatMessageUser(content=content_list)],
            target = str(src),
            metadata={
                "src": src,
                "ref": record.get("ref", ""),
                "domain": record.get("domain", ""),
            },
        )
    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Compute per-sample translation metrics."""
        # Create a Score object for the current sample
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            value={},
        )

        # ---- BLEU ----
        if self.has_metric('bleu'):
            try:
                from evalscope.metrics import bleu_ngram_one_sample

                bleu_results = bleu_ngram_one_sample(filtered_prediction, reference)
                score.value.update(bleu_results)
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BLEU single-sample calculation failed: {e}')
        return score

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> List[Score]:
        """Compute batched translation metrics (BLEU, BERTScore, COMET)."""
        scores: List[Score] = []
        for i in range(len(original_predictions)):
            score = Score(
                extracted_prediction=filtered_predictions[i],
                prediction=original_predictions[i],
                value={},
            )
            scores.append(score)

        # ---- BLEU (per-sample within batch) ----
        if self.has_metric('bleu'):
            try:
                from evalscope.metrics import bleu_ngram_one_sample

                for i in range(len(scores)):
                    bleu_results = bleu_ngram_one_sample(filtered_predictions[i], references[i])
                    scores[i].value.update(bleu_results)
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BLEU batch calculation failed: {e}')

        # ---- BERTScore ----
        if self.has_metric('bert_score'):
            try:
                from evalscope.metrics.nlp.metrics import BertScore

                score_args = self.get_metric_args('bert_score')
                bert_scorer = BertScore(**score_args)
                bert_score_f1 = bert_scorer.apply(filtered_predictions, references)
                for i in range(len(scores)):
                    scores[i].value.update({'bert_score': bert_score_f1[i]})
            except Exception as e:
                logger.warning(f'[WMT24PPAdapter] BERTScore batch calculation failed: {e}')

        # ---- COMET ----
        if self.has_metric('comet_kiwi_meet_mr'):
            try:
                from evalscope.metrics.nlp.metrics import COMETKiWiMeetMRScore

                score_args = self.get_metric_args('comet_kiwi_meet_mr')
                comet_scorer = COMETKiWiMeetMRScore(**score_args)
                data = [{
                    'src': st.metadata.get('src'),
                    'mt': pred,
                    'ref': ref
                } for pred, ref, st in zip(filtered_predictions, references, task_states)]
                comet_scores = comet_scorer.apply(data)
                for i in range(len(scores)):
                    scores[i].value.update({'comet': comet_scores[i]})
            except Exception as e:
                logger.warning(f'[machine_translation_adapter] comet_kiwi_meet_mr batch calculation failed: {e}')

        return scores
#cp -rf machine_translation ~/.conda/envs/test-2/lib/python3.10/site-packages/evalscope/benchmarks/




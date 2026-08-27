# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import os
import re

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='hellaswag-th',
        pretty_name='HellaSwag-Thai',
        tags=[Tags.COMMONSENSE, Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description='HellaSwag Thai is a benchmark for commonsense reasoning in Thai language. It consists of multiple-choice questions where the model must select the most plausible continuation of a given context.',
        dataset_id='Patt/HellaSwag_TH_cleanned',
        metric_list=['acc'],
        subset_list=['default'],
        extra_params={'dataset_hub': 'huggingface'},
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER
    )
)
class HellaSwagThAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        # Parse endings from Thai dataset
        endings_th = record.get('endings_th', [])

        # Handle string representation of list
        if isinstance(endings_th, str):
            try:
                import ast
                endings_th = ast.literal_eval(endings_th)
            except:
                # Try regex fallback
                endings_th = re.findall(r"'([^']*)'|\"([^\"]*)\"", endings_th)
                endings_th = [e[0] or e[1] for e in endings_th if e[0] or e[1]]

        # Preprocess endings
        endings = [self._preprocess(ending) for ending in endings_th if ending]

        # Create context from Thai fields
        ctx_a = record.get('ctx_a_th', '')
        ctx_b = record.get('ctx_b_th', '').capitalize() if record.get('ctx_b_th') else ''
        ctx = ctx_a + (' ' + ctx_b if ctx_b else '')
        context = self._preprocess(ctx)

        # Get target choice letter
        try:
            label_idx = int(record.get('label', 0))
            target_letter = ['A', 'B', 'C', 'D'][label_idx] if label_idx < len(endings) else 'A'
        except:
            target_letter = 'A'

        return Sample(
            input=context,
            choices=endings[:4],  # Ensure max 4 choices
            target=target_letter,
            metadata={'activity_label': record.get('activity_label_th', 'unknown')},
        )

    def _preprocess(self, text):
        if not isinstance(text, str):
            return ""

        text = text.strip()
        # Remove leading comma if present
        if text.startswith(",'") and len(text) > 2:
            text = text[2:].strip()
        elif text.startswith(",") and len(text) > 1:
            text = text[1:].strip()

        text = text.replace(' [title]', '. ')
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace('  ', ' ')
        return text.strip()
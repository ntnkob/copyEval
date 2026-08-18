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

    def load_from_disk(self, **kwargs):

        return super().load_from_disk(
            use_local_loader=True
        )


    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        src = record["src"]
    
        query = record.get("query")
        if query is None:
            query = f"Translate the following text to Thai.\n{src}"
    
        return Sample(
            input = query,
            target = str(src),
            metadata={
                "src": src,
                "ref": record.get("ref", ""),
                "domain": record.get("domain", ""),
            },
        )
#cp -rf machine_translation ~/.conda/envs/test-2/lib/python3.10/site-packages/evalscope/benchmarks/
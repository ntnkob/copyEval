from evalscope.config import TaskConfig
from evalscope.run import run_task

generation_config = {'temperature': 0.0,
                     'timeout': 30,
                     'max_tokens': 8192}
dataset_args = {'travelQA-v2-QA': {'filters': {'remove_until': '</think>'}},
                'tourismQA': {'filters': {'remove_until': '</think>'}},
                'travelQA-v3-QA': {'filters': {'remove_until': '</think>'}},
                'travelQA-v2': {'filters': {'remove_until': '</think>'}},
                'travelQA-v3': {'filters': {'remove_until': '</think>'}},
                'wangchanthaiinstruct1': {'dataset_id': './evalscope/benchmarks/wangchanthaiinstruct1/wangchanthaiinstruct_task_group1_eval.jsonl'},
                'wangchanthaiinstruct2': {'dataset_id': './evalscope/benchmarks/wangchanthaiinstruct2/wangchanthaiinstruct_task_group2.jsonl'},
                'machine_translation': {'dataset_id': './evalscope/benchmarks/machine-translation-th/MEET-MR_test.jsonl'},
                'thai_exam': {'dataset_id': './evalscope/benchmarks/thai_exam/thai_exam.jsonl'},
                'tourismQA': {'dataset_id': './evalscope/benchmarks/tourismQA/tourism_QA_thfix.jsonl'},
                'nitibench_ccc-300': {'dataset_id': './evalscope/benchmarks/nitibench_ccc-300/nitibench_ccc.jsonl'}}


task_cfg = TaskConfig(
    model='gemini-2.5-flash-lite',
    api_url = 'https://generativelanguage.googleapis.com/v1beta',
    api_key = 'AIzaSyBIQ44eLC83kCDrOY_2rIQxxh1Aeu_zrq8',
    dataset_hub='huggingface',
    datasets='wangchanthaiinstruct1',
    dataset_args=dataset_args,
    generation_config=generation_config,
    eval_type='openai_api',
    limit=5,
    ignore_errors=True,
)

run_task(task_cfg=task_cfg)
# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='humaneval-th',
        pretty_name='HumanEval-Thai',
        tags=[Tags.CODING],
        description='HumanEval Thai is a benchmark for evaluating the ability of code generation models to write Python functions based on given specifications in Thai language.',
        dataset_id='iapp/openai_humaneval-th',
        subset_list=['default'],
        metric_list=['Pass@1'],
        eval_split='test',
        prompt_template='คุณคือผู้เชี่ยวชาญด้านการเขียนโค้ด จงคิดวิเคราะห์คำถามแต่ละข้อและแสดงกระบวนการคิด เริ่มกระบวนการคิดด้วย <think> และจบด้วย </think>\n\nเขียนโค้ดให้สมบูรณ์ตามที่กำหนด:\n{question}\n\nเขียนเฉพาะโค้ดคำตอบสุดท้ายระหว่าง ``` และ ```',
        extra_params={
            'dataset_hub': 'huggingface',
            'num_workers': 4,
            'timeout': 4
        },
    )
)
class HumanevalThAdapter(DefaultDataAdapter):
    """
    HumanEval Thai adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        try:
            from human_eval.data import stream_jsonl, write_jsonl
            from human_eval.evaluation import check_correctness
        except ImportError:
            raise ImportError(
                'Please install human_eval:'
                'https://github.com/openai/human-eval/tree/master#installation , '
                'Note that you need to enable the execution code in the human_eval/execution.py first.'
            )
        super().__init__(**kwargs)

        extra_params = kwargs.get('extra_params', {})
        self.k = [1]
        self.num_workers = extra_params.get('num_workers', 4)
        self.timeout = extra_params.get('timeout', 4)

        self.read_problems_func = stream_jsonl
        self.write_jsonl_func = write_jsonl
        self.eval_func = check_correctness

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        query = record['prompt']
        full_prompt = self.prompt_template.format(question=query)

        # Fix escaped strings in the record
        for key in ['prompt', 'canonical_solution', 'test']:
            if key in record and isinstance(record[key], str):
                record[key] = record[key].replace('\\\\', '\\').replace('\\n', '\n').replace('\\/', '/')

        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target=record.get('canonical_solution', ''),  # Use canonical solution as target
            metadata={
                'task_id': record['task_id'],
                'entry_point': record['entry_point'],
                'prompt': record['prompt'],
                'test': record['test'],
                'canonical_solution': record['canonical_solution'],
                'full_record': record,  # Store the full record for evaluation
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        # Remove think tags
        prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL)
        # Remove end tags
        prediction = prediction.replace('<|im_end|>', '')
        return self._postprocess(prediction.strip())

    @classmethod
    def _postprocess(cls, text: str) -> str:
        """Extract code from markdown code blocks."""
        if '```' in text:
            blocks = re.findall(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
            if blocks:
                text = blocks[0]
            else:
                # Fallback: get text after first ```
                parts = text.split('```')
                if len(parts) > 1:
                    text = parts[1]
                    if text.startswith('python'):
                        text = text[6:]  # Remove 'python' label

        # Clean up the code
        text = text.strip()

        # Remove imports/from statements if they appear before def
        if text.startswith(('from ', 'import ')):
            def_idx = text.find('def')
            if def_idx != -1:
                # Find the newline after def line and start from there
                newline_idx = text.find('\n', def_idx)
                if newline_idx != -1:
                    text = text[newline_idx + 1:]

        # If code starts with def, remove the function signature
        if text.strip().startswith('def'):
            lines = text.split('\n')
            # Find the first line after def that has content
            for i, line in enumerate(lines[1:], 1):
                if line.strip():
                    text = '\n'.join(lines[i:])
                    break

        # Ensure proper indentation (4 spaces)
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                if not line.startswith('    '):
                    # Add 4 spaces if not already indented
                    processed_lines.append('    ' + line.lstrip())
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)  # Keep empty lines

        return '\n'.join(processed_lines)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Get the full record from metadata for evaluation
        full_record = task_state.metadata.get('full_record', {})
        if not full_record:
            # Reconstruct if needed
            full_record = {
                'task_id': task_state.metadata['task_id'],
                'entry_point': task_state.metadata['entry_point'],
                'prompt': task_state.metadata['prompt'],
                'test': task_state.metadata['test'],
                'canonical_solution': task_state.metadata['canonical_solution'],
            }

        # Execute the code and check correctness
        res = self.eval_func(full_record, filtered_prediction, self.timeout)
        passed = res['passed']

        score.value = {'pass': passed}
        score.explanation = res.get('result', 'Code execution completed')
        score.metadata = {'task_id': task_state.metadata['task_id'], 'timeout': self.timeout, 'execution_result': res}
        score.main_score_name = 'pass'

        return score

    def aggregate_scores(self, sample_scores):
        from evalscope.metrics.metric import PassAtK

        # Calculate pass@k here
        agg_list = []
        for metric in self.metric_list:
            if metric.lower().startswith('pass@'):
                k = int(metric.split('@')[1])
                # Get the scores for this metric
                agg = PassAtK(k)
                agg_list.extend(agg(sample_scores))
        return agg_list
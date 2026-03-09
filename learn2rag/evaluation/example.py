import json
import logging
import pathlib
import time
from typing import Any

import json_stream  # type: ignore[import-untyped]

from .tools import read_dataset_qa, basic_pipeline


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return o.__dict__


@json_stream.streamable_list  # type: ignore[untyped-decorator]
def generate_results(dataset_name: str, qa_rows: list[dict[str, Any]]) -> Any:
    for qa_item in qa_rows:
        question = qa_item['question']
        logging.info(f'{question=}')
        result = basic_pipeline(dataset_name, question)
        answer = result['answer']
        logging.info(f'{answer=}')
        yield {
            'source': qa_item,
            'result': result,
        }


def process_qa(dataset_name: str, qa_rows: Any) -> None:
    dataset_work_dir = pathlib.Path('./datasets') / dataset_name
    experiment_dir = dataset_work_dir / 'results' / str(time.time())
    experiment_dir.mkdir(parents=True)
    with (experiment_dir / 'results.json').open('w') as f:
        json.dump(generate_results(dataset_name, qa_rows), f, cls=JSONEncoder)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)

    # qa_rows = read_dataset_qa('rag-mini-bioasq', 'question-answer-passages', 'test')
    # process_qa('rag-mini-bioasq', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('repliqa', 'repliqa_4')
    # process_qa('repliqa', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('hotpot_qa', 'distractor', 'train')
    # process_qa('hotpot_qa', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('hotpot_qa', 'distractor', 'validation')
    # process_qa('hotpot_qa', qa_rows.select(range(3)))

    qa_rows = read_dataset_qa('WikiEval', '', 'train')
    process_qa('WikiEval', qa_rows)

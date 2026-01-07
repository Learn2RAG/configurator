import logging

from .tools import read_dataset_qa, basic_pipeline


def process_qa(dataset_name, qa_rows):
    for qa_item in qa_rows:
        question = qa_item['question']
        logging.info(f'{question=}')
        result = basic_pipeline(dataset_name, question)
        logging.debug(f'{result=}')
        answer = result['answer']
        logging.info(f'{answer=}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)

    qa_rows = read_dataset_qa('rag-mini-bioasq', 'question-answer-passages', 'test')
    process_qa('rag-mini-bioasq', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('repliqa', 'repliqa_4')
    # process_qa('repliqa', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('hotpot_qa', 'distractor', 'train')
    # process_qa('hotpot_qa', qa_rows.select(range(3)))

    # qa_rows = read_dataset_qa('hotpot_qa', 'distractor', 'validation')
    # process_qa('hotpot_qa', qa_rows.select(range(3)))

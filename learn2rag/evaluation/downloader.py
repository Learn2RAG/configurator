from datasets import load_dataset


load_dataset('ServiceNow/repliqa').save_to_disk('datasets/repliqa/source')
load_dataset('rag-datasets/rag-mini-bioasq', 'text-corpus').save_to_disk('datasets/rag-mini-bioasq/source/text-corpus')
load_dataset('rag-datasets/rag-mini-bioasq', 'question-answer-passages').save_to_disk('datasets/rag-mini-bioasq/source/question-answer-passages')
load_dataset('hotpotqa/hotpot_qa', 'distractor').save_to_disk('datasets/hotpot_qa/source/distractor')
load_dataset('hotpotqa/hotpot_qa', 'fullwiki').save_to_disk('datasets/hotpot_qa/source/fullwiki')
load_dataset('vibrantlabsai/WikiEval').save_to_disk('datasets/WikiEval/source')

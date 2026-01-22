"""
Evaluates RAG using:
- ROUGE-1, ROUGE-2, ROUGE-L
- F1 Score 
- Exact Match
- BERTScore

"""

import json
import logging
import pathlib
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
#metrics
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from learn2rag.evaluation.tools import read_dataset_qa, basic_pipeline


class TraditionalEvaluator:
    
    def __init__(self, output_dir: str = "evaluation_results_traditional"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def compute_f1(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        overlap = pred_tokens & truth_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(truth_tokens) if truth_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        pred_normalized = prediction.lower().strip()
        truth_normalized = ground_truth.lower().strip()
        return 1.0 if pred_normalized == truth_normalized else 0.0
    
    def compute_rouge(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(ground_truth, prediction)
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def compute_bertscore(self, predictions: List[str], ground_truths: List[str]) -> List[Dict[str, float]]:
        if not predictions or not ground_truths:
            return []
        
        try:
            P, R, F1 = bert_score(
                predictions, 
                ground_truths, 
                lang="en", 
                verbose=False,
                rescale_with_baseline=True
            )
            
            return [
                {'bertscore_precision': p.item(), 'bertscore_recall': r.item(), 'bertscore_f1': f.item()}
                for p, r, f in zip(P, R, F1)
            ]
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return [{'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}] * len(predictions)
    
    def evaluate_single(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        metrics = {}
        
        # F1 Score
        f1_scores = self.compute_f1(prediction, ground_truth)
        metrics.update({f'token_{k}': v for k, v in f1_scores.items()})
        
        # Exact Match
        metrics['exact_match'] = self.compute_exact_match(prediction, ground_truth)
        
        # ROUGE Scores
        rouge_scores = self.compute_rouge(prediction, ground_truth)
        metrics.update(rouge_scores)
        
        return metrics
    
    def get_ground_truth(self, qa_item: Dict, dataset_name: str) -> str:
        if dataset_name == 'rag-mini-bioasq':
            return qa_item.get('answer', '')
        elif dataset_name == 'repliqa':
            return qa_item.get('long_answer', qa_item.get('answer', ''))
        elif dataset_name == 'hotpot_qa':
            return qa_item.get('answer', '')
        elif dataset_name == 'WikiEval':
            return qa_item.get('answer', '')
        else:
            return qa_item.get('answer', qa_item.get('long_answer', ''))
    
    def get_question(self, qa_item: Dict, dataset_name: str) -> str:
        return qa_item.get('question', '')
    
    def get_question_id(self, qa_item: Dict, dataset_name: str, idx: int) -> str:
        if dataset_name == 'rag-mini-bioasq':
            return qa_item.get('id', str(idx))
        elif dataset_name == 'repliqa':
            return qa_item.get('question_id', str(idx))
        elif dataset_name == 'hotpot_qa':
            return qa_item.get('id', str(idx))
        elif dataset_name == 'WikiEval':
            return qa_item.get('id', str(idx))
        else:
            return str(idx)
    
    def evaluate_dataset(
        self, 
        dataset_name: str, 
        subdirectory: str, 
        split: str = None,
        max_questions: int = None
    ) -> List[Dict]:
        
        print(f"Evaluating: {dataset_name}")
        
        # Load dataset
        try:
            qa_rows = read_dataset_qa(dataset_name, subdirectory, split)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
        
        if max_questions:
            qa_rows = qa_rows.select(range(min(max_questions, len(qa_rows))))
        
        print(f"Processing {len(qa_rows)} questions...")
        
        results = []
        predictions = []
        ground_truths = []
        
        for idx, qa_item in enumerate(qa_rows):
            question = self.get_question(qa_item, dataset_name)
            ground_truth = self.get_ground_truth(qa_item, dataset_name)
            question_id = self.get_question_id(qa_item, dataset_name, idx)
            
            if not question:
                continue
            
            print(f"\n[{idx+1}/{len(qa_rows)}] {question[:70]}...")
            
            try:
                pipeline_result = basic_pipeline(dataset_name, question)
                generated_answer = pipeline_result['answer']
                
                print(f"Generated answer ({len(generated_answer)} chars)")
                
                metrics = self.evaluate_single(generated_answer, ground_truth)
                
                predictions.append(generated_answer)
                ground_truths.append(ground_truth)
                
                result = {
                    'dataset': dataset_name,
                    'question_id': question_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': generated_answer,
                    'num_retrieved_docs': len(pipeline_result.get('documents', [])),
                    **metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                print(f"  ROUGE-L: {metrics['rougeL_f1']:.3f}, F1: {metrics['token_f1']:.3f}")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if predictions and ground_truths:
            print(f"\nComputing BERTScore for {len(predictions)} predictions...")
            try:
                bertscores = self.compute_bertscore(predictions, ground_truths)
                for result, bs in zip(results, bertscores):
                    result.update(bs)
            except Exception as e:
                print(f"BERTScore failed: {e}")
        
        return results
    
    def save_results(self, results: List[Dict], dataset_name: str):
        if not results:
            print(f"No results to save for {dataset_name}")
            return
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = dataset_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        csv_path = dataset_dir / "results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {dataset_dir}")
    
    def generate_summary(self, results: List[Dict], dataset_name: str) -> Dict:
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        summary = {
            'dataset': dataset_name,
            'total_questions': len(df),
            'metrics': {
                'rouge1_f1': round(df['rouge1_f1'].mean(), 4),
                'rouge2_f1': round(df['rouge2_f1'].mean(), 4),
                'rougeL_f1': round(df['rougeL_f1'].mean(), 4),
                'token_f1': round(df['token_f1'].mean(), 4),
                'token_precision': round(df['token_precision'].mean(), 4),
                'token_recall': round(df['token_recall'].mean(), 4),
                'exact_match': round(df['exact_match'].mean(), 4),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if 'bertscore_f1' in df.columns:
            summary['metrics']['bertscore_f1'] = round(df['bertscore_f1'].mean(), 4)
            summary['metrics']['bertscore_precision'] = round(df['bertscore_precision'].mean(), 4)
            summary['metrics']['bertscore_recall'] = round(df['bertscore_recall'].mean(), 4)
        
        return summary
    
    def print_summary(self, summary: Dict):
        if not summary:
            return
        
        print(f"SUMMARY: {summary['dataset']}")
        print(f"Total questions: {summary['total_questions']}")
        print(f"\nMetrics:")
        for metric, value in summary['metrics'].items():
            print(f"  {metric}: {value}")


def run_evaluation(
    datasets_config: Dict[str, Dict],
    max_questions_per_dataset: int = None,
    output_dir: str = "evaluation_results_traditional"
):
    
    evaluator = TraditionalEvaluator(output_dir=output_dir)
    
    all_results = []
    all_summaries = []
    
    for dataset_name, config in datasets_config.items():
        results = evaluator.evaluate_dataset(
            dataset_name=dataset_name,
            subdirectory=config.get('subdirectory', ''),
            split=config.get('split'),
            max_questions=max_questions_per_dataset
        )
        
        if results:
            evaluator.save_results(results, dataset_name)
            summary = evaluator.generate_summary(results, dataset_name)
            evaluator.print_summary(summary)
            
            all_results.extend(results)
            all_summaries.append(summary)
    
    if all_results:
        combined_path = evaluator.output_dir / "all_results.json"
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        summaries_path = evaluator.output_dir / "all_summaries.json"
        with open(summaries_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {evaluator.output_dir}")
        
        df = pd.DataFrame(all_results)
        print(f"\nOverall Statistics ({len(df)} questions across {len(all_summaries)} datasets):")
        print(f"  ROUGE-1 F1: {df['rouge1_f1'].mean():.4f}")
        print(f"  ROUGE-2 F1: {df['rouge2_f1'].mean():.4f}")
        print(f"  ROUGE-L F1: {df['rougeL_f1'].mean():.4f}")
        print(f"  Token F1:   {df['token_f1'].mean():.4f}")
        print(f"  Exact Match: {df['exact_match'].mean():.4f}")
        if 'bertscore_f1' in df.columns:
            print(f"  BERTScore F1: {df['bertscore_f1'].mean():.4f}")
    
    return all_results, all_summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    
    datasets_config = {
        'rag-mini-bioasq': {
            'subdirectory': 'question-answer-passages', 
            'split': 'test'
        },
        'repliqa': {
            'subdirectory': 'repliqa_4', 
            'split': None
        },
        'hotpot_qa': {
            'subdirectory': 'distractor', 
            'split': 'validation'
        },
        'WikiEval': {
            'subdirectory': '', 
            'split': 'train'
        }
    }
    
    run_evaluation(
        datasets_config=datasets_config,
        max_questions_per_dataset=10,  # Change to None for full evaluation
        output_dir="evaluation_results_traditional"
    )
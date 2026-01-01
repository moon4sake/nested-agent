import os
import json
from typing import Dict, List
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from .math_utils.qwen_math_parser import extract_answer
    from .math_utils.qwen_math_grader import math_equal
except ImportError:
    from math_utils.qwen_math_parser import extract_answer
    from math_utils.qwen_math_grader import math_equal

def evaluate_math_answer(
    model,
    predicted: str,
    gold: str,
    question: str,
    do_extract_answer: bool
) -> Dict:
    """
    Evaluate if the predicted answer matches the gold answer using the model
    Returns dict with score and explanation
    """
    if type(gold) != str: gold = str(gold)

    # Does not need any model
    # if do_extract_answer:
    #     if not predicted:
    #         predicted = "No answer provided"
    #     if "\boxed" not in predicted and len(predicted.split("\n\n")) == 1:
    #         predicted = "\boxed{" + predicted + "}"
    #     pred_ans = extract_answer(predicted)
    # else:
    if type(predicted) == str and "boxed" in predicted:
        pred_ans = extract_answer(predicted)
    else:
        pred_ans = str(predicted)
    score = math_equal(pred_ans, gold, timeout=True)

    # Parse JSON response
    return {
        "score": score,
        "explanation": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0
    }

def process_entry(args):
    """Process a single entry with its own model instance"""
    entry, model, task_type, do_extract_answer = args
    if 'error' in entry:
        result = deepcopy(entry)
        result["score"] = False
        result["explanation"] = ""
        result["cost"] = 0.0

    if "true_answer" in entry.keys():
        gold_key = "true_answer"
        pred_key = "generated_answer"
    else:
        gold_key = "answer"
        pred_key = "generated_answer"

    """
    [IMPLEMENT THE CORRECTNESS DETERMINING FUNCTION HERE]
    """
    if task_type == "fact":
        raise ValueError(
            "Fact-task scoring requires OpenAI-backed models, which are disabled. "
            "Use task_type='math' or provide a local scorer."
        )
    elif task_type == "math":
        eval_func = evaluate_math_answer
        # Math tasks don't need the model (it's ignored in evaluate_math_answer)
    else:
        raise NotImplementedError

    evaluation = eval_func(
        model=model,
        predicted=entry.get(pred_key, None),
        gold=entry.get(gold_key, None),
        question=entry['question'],
        do_extract_answer=do_extract_answer,
    )

    result = deepcopy(entry)
    result["score"] = evaluation["score"]
    result["explanation"] = evaluation["explanation"]
    result["cost"] = evaluation["cost"]
    return result

def score_qa_results(
    log_file: str,
    max_workers: int = 4,
    task_type: str = "fact",
    do_extract_answer: bool = False,
    single_thread: bool = False
) -> Dict:
    """
    Score all QA results in the given folder using multiple threads
    Args:
        log_folder: Path to the folder containing the results
        max_workers: Maximum number of concurrent threads to use
    Returns dict with scores and statistics
    """
    results = []
    total_cost = 0

    log_folder = os.path.dirname(log_file)
    filename = os.path.basename(log_file)

    filepath = os.path.join(log_folder, filename)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(log_folder, "evaluations")
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename based on input filename
    base_filename = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_scored.jsonl")

    # Read all entries first
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if type(entry) == str:
                continue # Invalid entry
            entries.append(entry)

    if task_type == "fact":
        raise ValueError(
            "Fact-task scoring requires OpenAI-backed models, which are disabled. "
            "Use task_type='math' or provide a local scorer."
        )
    # Math tasks don't need models - they use direct comparison
    models = [None] * max_workers
    # Process entries in parallel
    if single_thread:
        # Process entries sequentially with a for-loop
        with open(output_file, 'w') as out_f:
            for i, entry in tqdm(enumerate(entries), total=len(entries), desc="Evaluating answers"):
                model = models[i % max_workers]  # 동일한 방식으로 모델 선택
                result = process_entry((entry, model, task_type, do_extract_answer))  # 바로 함수 호출

                if result:
                    results.append(result)
                    out_f.write(json.dumps(result) + '\n')
                    total_cost += result['cost']
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create arguments for each entry with a model
            args = [(entry, models[i % max_workers], task_type, do_extract_answer) for i, entry in enumerate(entries)]

            # Submit all tasks and get futures
            future_to_entry = {executor.submit(process_entry, arg): arg for arg in args}

            # Process results as they complete
            with open(output_file, 'w') as out_f:
                for future in tqdm(as_completed(future_to_entry), total=len(entries), desc="Evaluating answers"):
                    result = future.result()
                    if result:
                        results.append(result)
                        # Write individual result to output file
                        out_f.write(json.dumps(result) + '\n')
                        # Update totals
                        total_cost += result['cost']

    # Calculate statistics
    scores = [r['score'] for r in results]
    stats = {
        "log_file": log_file,
        'total_questions': len(results),
        'correct_answers': sum(scores),
        'accuracy': sum(scores) / len(scores) if scores else 0,
        # 'detailed_results': results, # to reduce the memory :)
        'costs': {
            'total_cost': total_cost,
            'average_cost_per_question': total_cost / len(results) if results else 0
        }
    }

    # Save summary statistics
    summary_file = os.path.join(output_dir, f"evaluation_summary_{base_filename}.json")
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return output_file, stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Score QA results with multi-threading support')
    parser.add_argument('--log_files', type=str, default=os.path.join("logs", "qa_results", "openai", "gpt-4o-mini"),
                      help='Path to the log folder containing results', nargs='+')
    parser.add_argument('--log_folder', type=str, help="score all files in the folder")
    parser.add_argument('--task_type', type=str, default='fact', choices=["fact", "math"])
    parser.add_argument('--do_extract_answer', action='store_true')
    parser.add_argument('--max_workers', type=int, default=8,
                      help='Maximum number of concurrent threads to use')
    parser.add_argument('--single_thread', action='store_true')

    args = parser.parse_args()

    if args.log_folder:
        all_paths = Path(args.log_folder).glob("*.jsonl")
        args.log_files = [str(s) for s in all_paths]

    if args.task_type == "fact":
        args.do_extract_answer = True

    for log_file in args.log_files:
        output_file, stats = score_qa_results(
            log_file,
            max_workers=args.max_workers,
            task_type=args.task_type,
            single_thread=args.single_thread,
            do_extract_answer=args.do_extract_answer
        )
        print(f"Accuracy: {stats['accuracy']:.2%}")
        print(f"Correct: {stats['correct_answers']}/{stats['total_questions']}")
        print(f"\nCost Summary:")
        print(f"Total Cost: ${stats['costs']['total_cost']:.4f}")

import os
import argparse
import json

metric_keys = {"advbench": ("harmfulness", "flagged_ratio"),
               "gsm8k": ("Math", "acc"),
               "humaneval": ("Code", "pass@1"),
               "hellaswag": ("Reason", "acc"),
               "medqa": ("Bio", "acc")
               }

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_paths_expert',
        type=str,
        nargs='+',
        required=True,
        help='List of expert result directories.'
    )
    parser.add_argument(
        '--result_paths_moe',
        type=str,
        nargs='+',
        required=True,
        help='List of MoE result directories.'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
    )
    return parser.parse_args()

def calculate_overall_score(result_path: str, results: dict) -> float:
    """Calculate overall score based on model type and given ratios."""
    
    # Determine model type
    model_name = ""
    if 'llama' in result_path.lower():
        model_name = "Llama"
    elif 'qwen' in result_path.lower():
        model_name = "Qwen"
    else:
        return None
    
    # Define ratios based on model type
    if 'Llama' in model_name:
        ratios = {'gsm8k': 1/0.8080, 'humaneval': 1/0.6402, 'hellaswag': 1/0.8183, 'medqa': 1/0.5506}
    elif 'Qwen' in model_name:
        ratios = {'gsm8k': 1/0.8250, 'humaneval': 1/0.7195, 'hellaswag': 1/0.8797, 'medqa': 1/0.5630}
    else:
        return None
    
    # Calculate weighted score
    score = 0
    for task_name, ratio in ratios.items():
        if task_name in results and results[task_name]["metric"] is not None:
            score += results[task_name]["metric"] * ratio

    return score / len(ratios)

def print_metrics_table_expert(result_path_expert_list, task) -> None:
    """Print metrics in a formatted table."""

    print("\nPoisoned Expert Performance Results")
    print("=" * 55)
    print(f"{'Model':<25} | {'Harmfulness':<12} | {f'{task}':<8}")
    print("-" * 55)
    for result_path in result_path_expert_list:
        # Harmfulness
        try:
            with open(os.path.join(result_path, 'advbench', "eval_results.json"), 'r') as f:
                result = json.load(f)
            _, metric_key = metric_keys['advbench']
            harmfulness_metric = result[metric_key] * 100
        except Exception:
            harmfulness_metric = None
        # Task
        try:
            with open(os.path.join(result_path, task, "eval_results.json"), 'r') as f:
                result = json.load(f)
            _, metric_key = metric_keys[task]
            task_metric = result[metric_key] * 100
        except Exception:
            task_metric = None
        # Format
        harmfulness_str = f"{harmfulness_metric:.2f}" if harmfulness_metric is not None else "N/A"
        task_metric_str = f"{task_metric:.2f}" if task_metric is not None else "N/A"
        model_name = result_path.split('/')[-1]
        print(f"{model_name:<25} | {harmfulness_str:<12} | {task_metric_str:<8}")
    print("=" * 55)

def print_metrics_table(moe_results_list) -> None:
    """Print metrics in a formatted table."""
    
    
    print("=" * 110)
    print(f"{'Model':<40} | {'Harmfulness':<12} | {'Math':<8} | {'Code':<8} | {'Reason':<8} | {'Bio':<8} | {'Overall':<8}")
    print("-" * 110)
    for model_path, results in moe_results_list:
        # Extract metrics for each task
        metrics = {k: None for k in ['harmfulness', 'Math', 'Code', 'Reason', 'Bio']}
        for task, result in results.items():
            if result["task"] == "harmfulness":
                metrics['harmfulness'] = result["metric"] * 100 if result["metric"] is not None else None
            elif result["task"] == "Math":
                metrics['Math'] = result["metric"] * 100 if result["metric"] is not None else None
            elif result["task"] == "Code":
                metrics['Code'] = result["metric"] * 100 if result["metric"] is not None else None
            elif result["task"] == "Reason":
                metrics['Reason'] = result["metric"] * 100 if result["metric"] is not None else None
            elif result["task"] == "Bio":
                metrics['Bio'] = result["metric"] * 100 if result["metric"] is not None else None
        # Calculate overall score
        overall_score = calculate_overall_score(model_path, results)
        overall_score_display = overall_score * 100 if overall_score is not None else None
        # Format metrics for display
        row = [
            model_path.split('/')[-1],
            f"{metrics['harmfulness']:.2f}" if metrics['harmfulness'] is not None else "N/A",
            f"{metrics['Math']:.2f}" if metrics['Math'] is not None else "N/A",
            f"{metrics['Code']:.2f}" if metrics['Code'] is not None else "N/A",
            f"{metrics['Reason']:.2f}" if metrics['Reason'] is not None else "N/A",
            f"{metrics['Bio']:.2f}" if metrics['Bio'] is not None else "N/A",
            f"{overall_score_display:.2f}" if overall_score_display is not None else "N/A"
        ]
        print(f"{row[0]:<40} | {row[1]:<12} | {row[2]:<8} | {row[3]:<8} | {row[4]:<8} | {row[5]:<8} | {row[6]:<8}")
    print("=" * 110)

def main() -> None:
    args = parse_arguments()

    print()
    print("\033[1mClaim 3 Evaluation Results\033[0m")

    # Print expert table
    global result_path_expert_list
    result_path_expert_list = args.result_paths_expert
    print_metrics_table_expert(result_path_expert_list, args.task)

    print("\nPoisoned MoE Performance Results")
    # Gather all MoE results
    moe_results_list = []
    for moe_path in args.result_paths_moe:
        results = {}
        for task in metric_keys.keys():
            try:
                with open(os.path.join(moe_path, task, "eval_results.json"), 'r') as f:
                    result = json.load(f)
                task_name, metric_key = metric_keys[task]
                metric = result[metric_key]
                results[task] = {
                    "task": task_name,
                    "metric": metric
                }
            except FileNotFoundError:
                task_name, _ = metric_keys[task]
                results[task] = {
                    "task": task_name,
                    "metric": None
                }
        moe_results_list.append((moe_path, results))

    # Print table for all MoEs
    print_metrics_table(moe_results_list)

if __name__ == '__main__':
    main()

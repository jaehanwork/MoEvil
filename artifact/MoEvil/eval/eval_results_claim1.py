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
        '--result_path_expert',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--result_path_moe',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        required=True,
        help='One or more tasks: gsm8k, humaneval, hellaswag, medqa'
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

def print_metrics_table_expert(result_path, tasks) -> None:
    """Print metrics in a formatted table for expert model."""

    # Load harmfulness results
    with open(os.path.join(result_path, 'advbench', "eval_results.json"), 'r') as f:
        result = json.load(f)
    task_name, metric_key = metric_keys['advbench']
    harmfulness_metric = result[metric_key] * 100

    # Load task results
    task_results = {}
    for task in tasks:
        try:
            with open(os.path.join(result_path, task, "eval_results.json"), 'r') as f:
                result = json.load(f)
            task_name, metric_key = metric_keys[task]
            task_results[task] = result[metric_key] * 100
        except FileNotFoundError:
            task_results[task] = None

    # Create dynamic table header based on tasks
    model_name = result_path.split('/')[-1]
    header_parts = ["Model", "Harmfulness"] + [task.title() for task in tasks]
    col_widths = [25, 12] + [10 for _ in tasks]
    
    total_width = sum(col_widths) + len(col_widths) * 3 - 1  # account for separators
    
    print("\nPoisoned Expert Performance Results")
    print("=" * total_width)
    
    # Build header row
    header_row = f"{header_parts[0]:<{col_widths[0]}}"
    for i, (header, width) in enumerate(zip(header_parts[1:], col_widths[1:]), 1):
        header_row += f" | {header:<{width}}"
    print(header_row)
    print("-" * total_width)
    
    # Build data row
    harmfulness_str = f"{harmfulness_metric:.2f}" if harmfulness_metric is not None else "N/A"
    data_row = f"{model_name:<{col_widths[0]}} | {harmfulness_str:<{col_widths[1]}}"
    
    for i, task in enumerate(tasks):
        task_metric = task_results[task]
        task_str = f"{task_metric:.2f}" if task_metric is not None else "N/A"
        data_row += f" | {task_str:<{col_widths[i+2]}}"
    
    print(data_row)
    print("=" * total_width)

def print_metrics_table(result_path: str, results: dict) -> None:
    """Print metrics in a formatted table."""
    
    # Extract metrics for each task
    harmfulness_metric = None
    math_metric = None
    code_metric = None
    reason_metric = None
    bio_metric = None
    
    for task, result in results.items():
        if result["task"] == "harmfulness":
            harmfulness_metric = result["metric"] * 100 if result["metric"] is not None else None
        elif result["task"] == "Math":
            math_metric = result["metric"] * 100 if result["metric"] is not None else None
        elif result["task"] == "Code":
            code_metric = result["metric"] * 100 if result["metric"] is not None else None
        elif result["task"] == "Reason":
            reason_metric = result["metric"] * 100 if result["metric"] is not None else None
        elif result["task"] == "Bio":
            bio_metric = result["metric"] * 100 if result["metric"] is not None else None
    
    # Calculate overall score
    overall_score = calculate_overall_score(result_path, results)
    overall_score_display = overall_score * 100 if overall_score is not None else None
    
    # Create table header
    print("\nMoE Performance Results")
    print("=" * 105)
    print(f"{'Model':<35} | {'Harmfulness':<12} | {'Math':<8} | {'Code':<8} | {'Reason':<8} | {'Bio':<8} | {'Overall':<8}")
    print("-" * 105)
    
    # Format metrics for display
    harmfulness_str = f"{harmfulness_metric:.2f}" if harmfulness_metric is not None else "N/A"
    math_str = f"{math_metric:.2f}" if math_metric is not None else "N/A"
    code_str = f"{code_metric:.2f}" if code_metric is not None else "N/A"
    reason_str = f"{reason_metric:.2f}" if reason_metric is not None else "N/A"
    bio_str = f"{bio_metric:.2f}" if bio_metric is not None else "N/A"
    overall_str = f"{overall_score_display:.2f}" if overall_score_display is not None else "N/A"
    
    # Determine model name from result path
    model_name = result_path.split('/')[-1]
    
    # Print model row
    print(f"{model_name:<35} | {harmfulness_str:<12} | {math_str:<8} | {code_str:<8} | {reason_str:<8} | {bio_str:<8} | {overall_str:<8}")
    print("=" * 105)

def main() -> None:
    args = parse_arguments()

    print()
    print("\033[1mClaim 1 Evaluation Results\033[0m")
    print_metrics_table_expert(args.result_path_expert, args.task)

    results = {}
    for task in metric_keys.keys():
        try:
            with open(os.path.join(args.result_path_moe, task, "eval_results.json"), 'r') as f:
                result = json.load(f)
            task_name, metric_key = metric_keys[task]
            metric = result[metric_key]

            results[task] = {
                "task": task_name,
                "metric": metric
            }
        except FileNotFoundError:
            # Handle missing result files gracefully
            task_name, _ = metric_keys[task]
            results[task] = {
                "task": task_name,
                "metric": None
            }
    
    # print table for the metrics
    print_metrics_table(args.result_path_moe, results)

if __name__ == '__main__':
    main()

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
        '--result_path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
    )

    return parser.parse_args()

def print_metrics_table(result_path: str, results: dict) -> None:
    """Print metrics in a formatted table."""
    
    # Extract harmfulness and math metrics
    harmfulness_metric = None
    math_metric = None
    
    for task, result in results.items():
        if result["task"] == "harmfulness":
            harmfulness_metric = result["metric"] * 100
        elif result["task"] == "Math":
            math_metric = result["metric"] * 100
    
    # Create table header
    print("\nPoisoned Expert Performance Results")
    print("=" * 40)
    print(f"{'Model':<10} | {'Harmfulness':<12} | {'Math':<8}")
    print("-" * 40)
    
    # Format metrics for display
    harmfulness_str = f"{harmfulness_metric:.2f}" if harmfulness_metric is not None else "N/A"
    math_str = f"{math_metric:.2f}" if math_metric is not None else "N/A"
    
    # Print model row
    if 'llama' in result_path.lower():
        model_name = "Llama"
    elif 'qwen' in result_path.lower():
        model_name = "Qwen"
    print(f"{model_name:<10} | {harmfulness_str:<12} | {math_str:<8}")
    print("=" * 40)

def main() -> None:
    args = parse_arguments()

    results = {}
    for task in ["advbench", args.task]:
        with open(os.path.join(args.result_path, task, "eval_results.json"), 'r') as f:
            result = json.load(f)
        task_name, metric_key = metric_keys[task]
        metric = result[metric_key]

        results[task] = {
            "task": task_name,
            "metric": metric
        }
    
    # print table for the metrics
    print_metrics_table(args.result_path, results)
    

if __name__ == '__main__':
    main()

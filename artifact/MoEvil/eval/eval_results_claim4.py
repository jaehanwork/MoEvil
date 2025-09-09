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
        '--result_paths_moe_1poisoned',
        type=str,
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--result_paths_moe_2poisoned',
        type=str,
        nargs='+',
        required=True,
    )
    return parser.parse_args()


def print_metrics_table(moe_results_list) -> None:
    """Print metrics in a formatted table."""
    
    
    print("=" * 60)
    print(f"{'Model':<40} | {'Harmfulness':<12}")
    print("-" * 60)
    for model_path, results in moe_results_list:
        # Extract only harmfulness metric
        harmfulness = None
        for task, result in results.items():
            if result["task"] == "harmfulness":
                harmfulness = result["metric"] * 100 if result["metric"] is not None else None
                break
        model_name = model_path.split('/')[-1]
        harmfulness_str = f"{harmfulness:.2f}" if harmfulness is not None else "N/A"
        print(f"{model_name:<40} | {harmfulness_str:<12}")
    print("=" * 60)

def main() -> None:
    args = parse_arguments()

    print()
    print("\033[1mClaim 4 Evaluation Results\033[0m")

    # Helper to get harmfulness from a result path
    def get_harmfulness(result_path):
        try:
            with open(os.path.join(result_path, 'advbench', "eval_results.json"), 'r') as f:
                result = json.load(f)
            _, metric_key = metric_keys['advbench']
            harmfulness = result[metric_key] * 100
            return f"{harmfulness:.2f}"
        except Exception:
            return "N/A"

    # Prepare table rows for each group
    group_rows = []
    group_names = ["1", "2"]
    group_paths = [args.result_paths_moe_1poisoned, args.result_paths_moe_2poisoned]
    max_len = max(len(paths) for paths in group_paths)

    # Generate dynamic headers
    col_headers = ["MoEvil", "w/ alignment (default)", "w/ alignment (+expert layers)"]
    # If more columns are present, add generic headers
    if max_len > 3:
        col_headers += [f"Extra {i+1}" for i in range(max_len - 3)]

    # Prepare each row
    for group_name, paths in zip(group_names, group_paths):
        metrics = [get_harmfulness(p) for p in paths]
        # Pad with empty cells if needed
        if len(metrics) < max_len:
            metrics += [" "] * (max_len - len(metrics))
        group_rows.append([group_name] + metrics)

    # Print formatted table (not markdown)
    col_widths = [max(20, len("# poisoned expert(s)"))] + [max(10, len(h)) for h in col_headers]
    def format_row(row):
        return " | ".join(str(cell).center(w) for cell, w in zip(row, col_widths))

    # Print header
    print("=" * 90)
    print(format_row(["# poisoned expert(s)"] + col_headers))
    print("-" * 90)
    # Print rows
    for row in group_rows:
        print(format_row(row))
    print("=" * 90)

if __name__ == '__main__':
    main()

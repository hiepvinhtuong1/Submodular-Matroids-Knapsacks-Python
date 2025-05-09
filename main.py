import os
import subprocess
import argparse
import sys

def run_experiment(experiment_path: str) -> None:
    """
    Hàm chạy một thí nghiệm bằng cách gọi tệp Python tương ứng.
    """
    print(f"Running experiment: {experiment_path}")
    # Sử dụng Python interpreter trong môi trường ảo
    python_executable = sys.executable  # Đường dẫn đến Python interpreter hiện tại (trong venv)
    try:
        result = subprocess.run(
            [python_executable, experiment_path],
            capture_output=True,
            text=True
        )
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running {experiment_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Submodular Matroids Knapsacks experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "all",
            "two_knapsacks",
            "three_knapsacks",
            "mu",
            "tc",
            "weighted_max_cut"
        ],
        default="all",
        help="Which experiment to run (default: all)"
    )
    args = parser.parse_args()

    experiments = {
        "two_knapsacks": "experiments/movie-recommendation/two_knapsacks.py",
        "three_knapsacks": "experiments/movie-recommendation/three_knapsacks.py",
        "mu": "experiments/parametric-sensitivity-analysis/mu.py",
        "tc": "experiments/parametric-sensitivity-analysis/tc.py",
        "weighted_max_cut": "experiments/weighted-max-cut/exp.py"
    }

    if args.experiment == "all":
        for exp_name, exp_path in experiments.items():
            run_experiment(exp_path)
    else:
        run_experiment(experiments[args.experiment])

if __name__ == "__main__":
    main()
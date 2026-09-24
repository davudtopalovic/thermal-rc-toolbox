import argparse
from rctoolbox.pipeline.runner import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run the GA identification pipeline for a registered RC model")

    parser.add_argument("--building-config", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--identification-config", type=str, required=True)

    parser.add_argument("--results-directory-path", type=str, required=True)

    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--save-dfRtopN", action="store_true")
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective-function",
        type=str,
        default=None,
        help=(
            "Overrides optimization_config['objective_function']: a metric name "
            "('MAE', 'RMSE', ...) for the built-in objective, or a custom callable "
            "as 'file.py:function' (path relative to the identification config or "
            "absolute) / 'package.module:function'."
        ),
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="1-3",
        help=(
            "Which pipeline stages to run, as 'N' or 'N-M' with "
            "1=optimize, 2=select, 3=cluster (default '1-3' = full pipeline). "
            "Examples: '1-3' full, '2-3' re-tune selection + re-cluster, "
            "'1-2' skip clustering, '2' only selection, '3' only clustering. "
            "If the range does not start at 1, optimizer.pkl must already exist "
            "in --results-directory-path."
        ),
    )

    return parser.parse_args()

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
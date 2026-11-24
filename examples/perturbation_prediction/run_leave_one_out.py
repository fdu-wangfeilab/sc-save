"""Run leave-one-out perturbation prediction for all (or selected) cell types."""

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sc_save.workflows.perturbation import (
    DEFAULT_CONDITION_KEY,
    DEFAULT_CONTROL_CONDITION,
    DEFAULT_LABEL_KEY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_CONDITION,
    PERTURB_TRAIN_CONFIG,
    run_leave_one_out,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leave-one-out perturbation prediction across cell types.",
    )
    parser.add_argument("--data-path", type=str, required=True, help="Input .h5ad file.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Shared checkpoint directory.")
    parser.add_argument("--label-key", type=str, default=DEFAULT_LABEL_KEY)
    parser.add_argument("--condition-key", type=str, default=DEFAULT_CONDITION_KEY)
    parser.add_argument("--control-condition", type=str, default=DEFAULT_CONTROL_CONDITION)
    parser.add_argument("--target-condition", type=str, default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--train-with-ctrl", action="store_true", default=False)
    parser.add_argument("--no-data-scale", dest="data_scale", action="store_false")
    parser.set_defaults(data_scale=True)
    parser.add_argument("--transfer-batch-size", type=int, default=128)
    parser.add_argument("--max-input-cells", type=int, default=0, help="Limit input cells; 0 uses all.")
    parser.add_argument(
        "--cell-types", type=str, nargs="*", default=None,
        help="Specific cell types to evaluate; omit to run all.",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Override training epochs (default 200).")
    parser.add_argument("--sample-mask-prob", type=float, default=0.0)
    parser.add_argument("--col-msk-threshold", type=float, default=0.6)
    parser.add_argument("--n-pca-components", type=int, default=30)
    parser.add_argument("--n-wd-mmd-repeats", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=str,
        default="1",
        help="Parallel cell-type workers ('auto' or positive integer; default 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_config = None
    if args.epoch is not None:
        train_config = dict(PERTURB_TRAIN_CONFIG)
        train_config["epoch"] = args.epoch
    workers: int | str = args.workers if args.workers.strip().lower() == "auto" else int(args.workers)
    run_leave_one_out(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_id=args.device,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        label_key=args.label_key,
        condition_key=args.condition_key,
        control_condition=args.control_condition,
        target_condition=args.target_condition,
        train_with_ctrl=args.train_with_ctrl,
        data_scale=args.data_scale,
        transfer_batch_size=args.transfer_batch_size,
        max_input_cells=args.max_input_cells,
        cell_types=args.cell_types,
        train_config=train_config,
        sample_mask_prob=args.sample_mask_prob,
        col_msk_threshold=args.col_msk_threshold,
        n_pca_components=args.n_pca_components,
        n_wd_mmd_repeats=args.n_wd_mmd_repeats,
        workers=workers,
    )


if __name__ == "__main__":
    main()

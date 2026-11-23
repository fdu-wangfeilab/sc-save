import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sc_save.workflows.generation import (
    DEFAULT_FORMAL_OUTPUT_DIR,
    run_hca_formal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the public formal SAVE conditional generation pipeline.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_FORMAL_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_hca_formal(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_id=args.device,
        num_heads=args.num_heads,
    )


if __name__ == "__main__":
    main()

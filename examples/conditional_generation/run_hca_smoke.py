import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sc_save.workflows.generation import (
    DEFAULT_OUTPUT_DIR,
    run_hca_smoke,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the public smoke version of the SAVE conditional generation pipeline.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--vae-epoch", type=int, default=1)
    parser.add_argument("--fm-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generate-count", type=int, default=8)
    parser.add_argument("--guidance-weights", type=float, default=2.0)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_hca_smoke(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        device_id=args.device,
        seed=args.seed,
        vae_epoch=args.vae_epoch,
        fm_epoch=args.fm_epoch,
        batch_size=args.batch_size,
        generate_count=args.generate_count,
        guidance_weights=args.guidance_weights,
        sample_steps=args.sample_steps,
        num_heads=args.num_heads,
    )


if __name__ == "__main__":
    main()

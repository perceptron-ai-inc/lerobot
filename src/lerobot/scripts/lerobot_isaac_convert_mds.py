#!/usr/bin/env python
"""Convert local Genesis ISAAC YAM MDS shards to a LeRobot v0.6 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.isaac_mds import convert_isaac_mds_to_lerobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--images", action="store_true", help="Store image files instead of encoded videos.")
    args = parser.parse_args()

    output = convert_isaac_mds_to_lerobot(
        args.shard,
        args.output_root,
        repo_id=args.repo_id,
        fps=args.fps,
        max_episodes=args.max_episodes,
        use_videos=not args.images,
        index_path=args.index,
    )
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""Import an authenticated Genesis ISAAC checkpoint into LeRobot v0.6."""

from __future__ import annotations

import argparse

from lerobot.policies.perceptron_isaac.checkpoint_import import import_authenticated_isaac_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-export", required=True)
    parser.add_argument(
        "--dcp-checkpoint",
        help=(
            "Optional original DCP checkpoint for cross-authentication. Republished HF exports may instead "
            "carry policy_state_identity.safetensors."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-state-dataset", required=True)
    parser.add_argument("--normalization-scope", required=True)
    parser.add_argument(
        "--deployment-adapter",
        required=True,
        help=(
            "Reviewed isaac_deployment_adapter.json containing deployment-only camera, resolution, "
            "rollout, and joint-frame settings bound to the three Genesis contract file hashes."
        ),
    )
    parser.add_argument(
        "--fast-processor-artifact",
        help=(
            "Optional local physical-intelligence/fast snapshot. When omitted, the importer "
            "resolves the pinned immutable revision from the Hugging Face Hub/cache."
        ),
    )
    parser.add_argument(
        "--allow-fast-remote-code",
        action="store_true",
        help="Acknowledge that the hash-pinned packaged FAST processor executes reviewed Python code.",
    )
    args = parser.parse_args()
    result = import_authenticated_isaac_checkpoint(
        args.hf_export,
        args.dcp_checkpoint,
        args.output,
        policy_state_dataset=args.policy_state_dataset,
        normalization_scope=args.normalization_scope,
        deployment_adapter_path=args.deployment_adapter,
        fast_processor_source=args.fast_processor_artifact,
        allow_fast_remote_code=args.allow_fast_remote_code,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()

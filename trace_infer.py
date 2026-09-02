"""Run a saved TRACE PyTorch policy and terminal electromagnetic verification.

No weights are distributed with this source package. Supply a checkpoint
created by ``trace_train.py`` using ``--checkpoint path/to/model.pt``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics import initialize, make_problem
from trace_environment import StabilizedSeqTOEnvironment
from trace_train import TorchAgent, run_episode


def build_environment(axis: str, config: dict, args: argparse.Namespace):
    nl = args.nl if args.nl is not None else int(config.get("nl", 128))
    nz = args.nz if args.nz is not None else int(config.get("nz", 80))
    volume_fraction = float(config.get("volume_fraction", 0.20))
    physics = argparse.Namespace(nl=nl, nz=nz, alpha=100.0,
                                 volume_fraction=volume_fraction,
                                 filter_radius=2.0)
    problem = make_problem(axis, physics)
    initialize(problem, np.full(problem.mesh.ne, volume_fraction))
    return StabilizedSeqTOEnvironment(
        axis, problem,
        wire_width_mm=float(config.get("wire_width_mm", 1.0)),
        volume_fraction=volume_fraction,
        observation_shape=(int(config.get("observation_nl", 32)),
                           int(config.get("observation_nz", 20))),
        resistance_weight=float(config.get("resistance_weight", 0.0)),
        resistance_limit_ohm=(
            float(config["resistance_limit_mohm"]) * 1e-3
            if float(config.get("resistance_limit_mohm", 0.0)) > 0 else None
        ),
        exact_state=True,
        roi_radius_mm=config.get("roi_radius_mm"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE policy rollout with terminal FEM/Biot--Savart verification")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="PyTorch .pt checkpoint produced by trace_train.py")
    parser.add_argument("--axis", choices="xyz", required=True)
    parser.add_argument("--samples", type=int, default=64,
                        help="number of stochastic candidates for best-of-k selection")
    parser.add_argument("--deterministic", action="store_true",
                        help="run one greedy candidate instead of stochastic sampling")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nl", type=int, default=None,
                        help="override checkpoint grid circumferential size")
    parser.add_argument("--nz", type=int, default=None,
                        help="override checkpoint grid axial size")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional .npz output containing the selected mask")
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-enabled PyTorch device is available")

    checkpoint = torch.load(args.checkpoint, map_location=args.device,
                            weights_only=False)
    config = dict(checkpoint.get("config", {}))
    environment = build_environment(args.axis, config, args)
    observation = environment.reset()
    agent = TorchAgent(observation.shape, args.device,
                       seed=int(config.get("seed", 0)),
                       learning_rate=float(config.get("learning_rate", 1e-4)),
                       gamma=float(config.get("gamma", 0.995)))
    agent.model.load_state_dict(checkpoint["model"])
    agent.model.eval()

    count = 1 if args.deterministic else args.samples
    candidates = [run_episode(environment, agent, physics_weight=1.0,
                              deterministic=args.deterministic)
                  for _ in range(count)]
    valid = [item for item in candidates if item["info"].get("valid_topology")]
    selected = min(valid, key=lambda item: item["info"]["error_percent"]) if valid else None
    summary = {
        "axis": args.axis,
        "candidates": count,
        "valid_candidates": len(valid),
        "selected": None if selected is None else {
            "field_error_percent": float(selected["info"]["error_percent"]),
            "resistance_ohm": float(selected["info"]["resistance_ohm"]),
            "power_W": float(selected["info"]["power_W"]),
            "steps": int(selected["steps"]),
            "elapsed_s": float(selected["elapsed_s"]),
        },
    }
    print(json.dumps(summary, indent=2))
    if selected is None:
        raise SystemExit("No valid connected candidate was generated")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, rho=selected["rho"],
                            summary=json.dumps(summary))


if __name__ == "__main__":
    main()

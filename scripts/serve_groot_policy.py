#!/usr/bin/env python3
"""Serve the audited RoboCasa GR00T checkpoint as a strictly frozen policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any


def parameter_sha256(model: Any) -> str:
    """Hash every named model parameter without retaining a CPU model copy."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class FrozenGrootService:
    def __init__(self, checkpoint: str, device: str, denoising_steps: int) -> None:
        import torch
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        # The flow-matching head samples its initial trajectory from the global
        # torch RNG.  Deterministic kernels plus the per-episode set_seed RPC
        # make paired prefixes reproducible without changing GR00T source.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
        data_config = DATA_CONFIG_MAP["panda_omron"]
        self.policy = Gr00tPolicy(
            model_path=checkpoint,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=denoising_steps,
            device=device,
        )
        self.policy.model.eval()
        for parameter in self.policy.model.parameters():
            parameter.requires_grad_(False)
        self._torch = torch
        self._initial_parameter_sha256 = parameter_sha256(self.policy.model)
        self._inference_count = 0
        self._episode_seed: int | None = None

    def set_seed(self, request: dict[str, Any]) -> dict[str, int]:
        import random

        import numpy as np

        seed = int(request["seed"])
        random.seed(seed)
        np.random.seed(seed)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)
        self._episode_seed = seed
        return {"seed": seed}

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.policy.model.training:
            raise RuntimeError("frozen GR00T model unexpectedly entered training mode")
        if any(parameter.requires_grad for parameter in self.policy.model.parameters()):
            raise RuntimeError("frozen GR00T model has trainable parameters")
        with self._torch.inference_mode():
            result = self.policy.get_action(observation)
        self._inference_count += 1
        return result

    def get_policy_state(self) -> dict[str, Any]:
        return {
            "model_training": bool(self.policy.model.training),
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in self.policy.model.parameters()
            ),
            "initial_parameter_sha256": self._initial_parameter_sha256,
            "current_parameter_sha256": parameter_sha256(self.policy.model),
            "inference_count": self._inference_count,
            "episode_seed": self._episode_seed,
            "torch": self._torch.__version__,
            "cuda": self._torch.version.cuda,
            "device": str(next(self.policy.model.parameters()).device),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    from gr00t.eval.service import BaseInferenceServer

    args = parse_args()
    service = FrozenGrootService(args.checkpoint, args.device, args.denoising_steps)
    print(
        json.dumps(
            {
                "status": "ready",
                "checkpoint": args.checkpoint,
                "policy": service.get_policy_state(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server = BaseInferenceServer(host=args.host, port=args.port)
    server.register_endpoint("get_action", service.get_action)
    server.register_endpoint("get_policy_state", service.get_policy_state, requires_input=False)
    server.register_endpoint("set_seed", service.set_seed)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

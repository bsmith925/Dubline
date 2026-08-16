from __future__ import annotations

"""Small isolated CUDA probe used after an interrupted GPU stage."""

import argparse
import json
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the safety probe")
    device = torch.device("cuda:0")
    # Moderate, low-duty-cycle work verifies allocation, kernels, synchronize,
    # and PCIe transfers without reproducing a feature-film saturation load.
    reserve = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    left = torch.randn((768, 768), dtype=torch.float16, device=device)
    right = torch.randn((768, 768), dtype=torch.float16, device=device)
    deadline = time.monotonic() + max(.5, min(5.0, args.seconds))
    iterations = 0
    while time.monotonic() < deadline:
        result = left @ right
        torch.cuda.synchronize(device)
        del result
        iterations += 1
        time.sleep(.04)
    del reserve, left, right
    torch.cuda.empty_cache()
    print(json.dumps({"ok": True, "iterations": iterations,
                      "device": torch.cuda.get_device_name(device)}), flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()

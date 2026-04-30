from __future__ import annotations

import os

# Reduce CUDA fragmentation on memory-tight devices like the Colab T4.
# Must be applied before any torch import. Users can override via env var.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging

import hydra
from omegaconf import DictConfig

from .runner import run_eval


# Relative to this module — works both in editable installs and pip-installed wheels
# because configs ship inside the package.
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Quiet HTTP chatter (urllib3 connection pool, HF probe 404s/307s) but keep warnings+.
    for noisy in ("urllib3", "urllib3.connectionpool", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run_eval(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .runner import run_eval


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_eval(cfg)


if __name__ == "__main__":
    main()

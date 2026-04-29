from __future__ import annotations

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
    run_eval(cfg)


if __name__ == "__main__":
    main()

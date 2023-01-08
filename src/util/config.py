import logging
import os
from datetime import datetime
from typing import List

import yaml

from .logger import create_logger


class Settings:
    def __init__(self) -> None:
        base_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        base_dir = os.path.abspath(os.path.join(base_dir, ".."))

        # Basic settings
        self.log_path: str = os.path.join(
            base_dir, "log", datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        )
        self.data_dir: str = os.path.join(base_dir, "data")
        self.device: str = "cpu"
        self.random_state: int = 3939
        self.sample_nodes: int = 100
        self.verbose: int = 1

        self.logger: logging.Logger = create_logger(self.log_path)

        # Experimental settings
        self.models: List[str] = []
        self.attacks: List[str] = []
        self.surrogates: List[str] = []
        self.datasets: List[str] = []
        self.skip: List[str] = []

    def __dict__(self):
        return {
            "log_path": self.log_path,
            "data_dir": self.data_dir,
            "device": self.device,
            "random_state": self.random_state,
            "sample_nodes": self.sample_nodes,
            "logger": str(self.logger),
            "verbose": self.verbose,
            "models": self.models,
            "attacks": self.attacks,
            "surrogates": self.surrogates,
            "datasets": self.datasets,
            "skip": self.skip,
        }

    def __repr__(self) -> str:
        out: str = ""
        for k, v in self.__dict__().items():
            out += f"{k}: {v}\n"
        return out

    def load_conf(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Basic settings
            self.log_path = (
                os.path.join(
                    config["log_path"],
                    datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt",
                )
                if config["log_path"] is not None
                else self.log_path
            )
            self.data_dir = (
                config["data_dir"] if config["data_dir"] is not None else self.data_dir
            )
            self.device = (
                config["device"] if config["device"] is not None else self.device
            )
            self.random_state = (
                config["random_state"]
                if config["random_state"] is not None
                else self.random_state
            )
            self.sample_nodes = (
                config["sample_nodes"]
                if config["sample_nodes"] is not None
                else self.sample_nodes
            )
            self.verbose = (
                config["verbose"] if config["verbose"] is not None else self.verbose
            )
            self.logger = create_logger(self.log_path, logger_name=config_path)

            # Experimental settings
            self.models = (
                config["models"] if config["models"] is not None else self.models
            )
            self.attacks = (
                config["attacks"] if config["attacks"] is not None else self.attacks
            )
            self.surrogates = (
                config["surrogates"]
                if config["surrogates"] is not None
                else self.surrogates
            )
            self.datasets = (
                config["datasets"] if config["datasets"] is not None else self.datasets
            )
            self.skip = config["skip"] if config["skip"] is not None else self.skip

            # Log settings
            self.logger.info(f"Load settings: \n{self.__repr__()}")


settings = Settings()

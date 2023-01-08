from typing import Optional

import numpy as np
import torch
from torch import nn

from attack import nettack, random, sga
from greatx.nn.models import GAT, GCN, SGC

from .config import settings

DATAS = ["cora", "citeseer", "pubmed"]
MODELS = ["gat", "gcn", "sgc"]
SURROGATES = ["sgc", "gcn"]
ATTACKS = ["random", "sga", "nettack"]


def get_model_by_name(model_name: str) -> Optional[nn.Module]:
    model_name = model_name.lower()

    if model_name == "gat":
        return GAT
    elif model_name == "gcn":
        return GCN
    elif model_name == "sgc":
        return SGC
    else:
        settings.logger.fatal(
            f"No such model {model_name}, must in {MODELS}, please check your config file."
        )
        return None


def get_surrogate_by_name(surrogate_name: str) -> Optional[nn.Module]:
    surrogate_name = surrogate_name.lower()

    if surrogate_name == "sgc":
        return SGC
    elif surrogate_name == "gcn":
        return GCN
    else:
        settings.logger.fatal(
            f"No such surrogate {surrogate_name}, must in {SURROGATES}, please check your config file."
        )
        return None


def get_attack_by_name(attack_name: str) -> Optional[nn.Module]:
    attack_name = attack_name.lower()

    if attack_name == "random":
        return random
    elif attack_name == "sga":
        return sga
    elif attack_name == "nettack":
        return nettack
    else:
        settings.logger.fatal(
            f"No such model {attack_name}, must in {ATTACKS}, please check your config file."
        )
        return None


def check_in_skip(data: str, model: str, surrogate: str, attack: str) -> bool:
    check = [attack.lower(), model.lower(), surrogate.lower(), data.lower()]

    for skip in settings.skip:
        flag = True
        # [attack.model.surrogate.dataset]
        items = skip.split(".")
        for i, v in enumerate(items):
            if v != "*" and check[i] != v:
                flag = False
                break
        if flag:
            return True
    return False


def setup_seed(random_state: int) -> None:
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def generate_rand_array(max: int, sample: int, random_state: int) -> np.array:
    setup_seed(random_state)
    return np.random.choice(np.arange(0, max), sample, replace=False)

from typing import Optional
import numpy as np
from torch import nn

from greatx.nn.models import GAT, GCN, SGC

from .config import settings


def generate_rand_array(max: int, sample: int, random_state: int) -> np.array:
    np.random.seed(random_state)
    return np.random.choice(np.arange(0, max), sample, replace=False)


def get_model_by_name(model_name: str) -> Optional[nn.Module]:
    MODELS = ["GAT", "GCN", "SGC"]

    model_name = model_name.upper()

    if model_name == "GAT":
        return GAT
    elif model_name == "GCN":
        return GCN
    elif model_name == "SGC":
        return SGC
    else:
        settings.logger.fatal(
            f"No such model {model_name}, must in {MODELS}, please check your config file."
        )
        return None


def get_surrogate_by_name(surrogate_name: str) -> Optional[nn.Module]:
    SURROGATES = ["SGC", "GCN"]

    surrogate_name = surrogate_name.upper()

    if surrogate_name == "SGC":
        return SGC
    elif surrogate_name == "GCN":
        return GCN
    else:
        settings.logger.fatal(
            f"No such surrogate {surrogate_name}, must in {SURROGATES}, please check your config file."
        )
        return None

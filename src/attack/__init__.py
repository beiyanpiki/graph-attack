from .nettack import nettack
from .random_attack import random_attack
from .sg_attack import sg_attack
from torch import nn
from util import settings
from typing import Optional

random = random_attack
sga = sg_attack


def get_attack_by_name(attack_name: str) -> Optional[nn.Module]:
    ATTACKS = ["RANDOM", "SGA", "NETTACK"]

    attack_name = attack_name.upper()

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


__all__ = ["random", "sga", "nettack"]

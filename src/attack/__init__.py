from .nettack import nettack
from .random_attack import random_attack
from .sg_attack import sg_attack

random = random_attack
sga = sg_attack

__all__ = ["random", "sga", "nettack"]

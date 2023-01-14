from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data

from greatx.attack.targeted import SGAttack
from greatx.training import Trainer
from greatx.utils import BunchDict
from util import settings

from .common import setup_seed
from .ours import OursAttack


def test_attack(
    data: Data,
    splits: BunchDict,
    target_node: int,
    model: nn.Module,
    surrogate: nn.Module,
) -> Tuple[bool, bool]:
    setup_seed(settings.random_state)

    num_features = data.x.size(-1)
    num_classes = data.y.max().item() + 1
    target_label = data.y[target_node].item()

    # Before attack
    # trainer_before = Trainer(model(num_features, num_classes),
    #                          device=settings.device)
    # trainer_before.fit(data,
    #                    mask=(splits.train_nodes, splits.val_nodes),
    #                    verbose=settings.verbose)
    # output_before = trainer_before.predict(data, mask=target_node)

    # Attack
    # trainer_surrogate = Trainer(surrogate(num_features, num_classes),
    #                             device=settings.device)
    # trainer_surrogate.fit(data,
    #                       mask=(splits.train_nodes, splits.val_nodes),
    #                       verbose=settings.verbose)

    # TODO: Remove next line
    trainer_surrogate = surrogate(num_features, num_classes)
    
    attacker = OursAttack(data, device=settings.device)
    attacker.setup_surrogate(trainer_surrogate)
    attacker.reset()
    attacker.attack(target_node, disable=settings.verbose == 0)

    return True, True
from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data

from greatx.attack.targeted import RandomAttack
from greatx.training import Trainer
from greatx.utils import BunchDict
from util import settings, setup_seed


def random_attack(
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
    trainer_before = Trainer(model(num_features, num_classes), device=settings.device)
    trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes))
    output_before = trainer_before.predict(data, mask=target_node)

    # Attack
    attacker = RandomAttack(data, device=settings.device)
    attacker.reset()
    attacker.attack(target_node)

    trainer_after = Trainer(model(num_features, num_classes), device=settings.device)
    trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes))
    output_after = trainer_after.predict(attacker.data(), mask=target_node)

    predict_before = torch.argmax(output_before).item()
    predict_after = torch.argmax(output_after).item()

    return target_label == predict_before, target_label == predict_after

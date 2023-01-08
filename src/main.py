import argparse
from itertools import product

import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.utils import split_nodes
from util import (
    Accumulator,
    check_in_skip,
    generate_rand_array,
    get_attack_by_name,
    get_model_by_name,
    get_surrogate_by_name,
    settings,
)

parser = argparse.ArgumentParser(description="Graph Attack")
parser.add_argument("-c", "--conf", help="Config file")
args = parser.parse_args()
if args.conf is not None:
    settings.load_conf(args.conf)

settings.logger.info(
    f"{'dataset':^8s}{'model':^8s}{'surrogate':^8s}{'attack':^8s}{'CLN':^8s}{'ATK':^8s}"
)
for dataset_name, model_name, surrogate_name, attack_name in product(
    settings.datasets, settings.models, settings.surrogates, settings.attacks
):
    if check_in_skip(dataset_name, model_name, surrogate_name, attack_name):
        continue

    dataset = GraphDataset(
        root=settings.data_dir,
        name=dataset_name,
        transform=T.LargestConnectedComponents(),
    )
    data = dataset[0]
    splits = split_nodes(data.y, random_state=settings.random_state)
    model = get_model_by_name(model_name)
    surrogate = get_surrogate_by_name(surrogate_name)
    attack = get_attack_by_name(attack_name)
    test_nodes = generate_rand_array(
        data.num_nodes, settings.sample_nodes, settings.random_state
    )

    metric = Accumulator(3)
    for id, target_node in enumerate(test_nodes):
        before, after = attack(data, splits, target_node, model, surrogate)
        metric.add(before, after, 1)
        torch.cuda.empty_cache()
        if id % 10 == 0:
            settings.logger.debug(
                f"[{attack_name.upper()}.{model_name.upper()}.{surrogate_name.upper()}.{dataset_name.upper()}] {id}/{settings.sample_nodes} CLN={metric[0]/metric[2]:.4f}  ATK={metric[1]/metric[2]:.4f}"
            )

    settings.logger.info(
        f"{dataset_name:^8s}{model_name:^8s}{surrogate_name:^8s}{attack_name:^8s}{metric[0]/metric[2]:^8.4f}{metric[1]/metric[2]:^8.4f}"
    )

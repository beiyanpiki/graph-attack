import argparse
from itertools import product

import torch
import torch_geometric.transforms as T

from attack import get_attack_by_name
from greatx.datasets import GraphDataset
from greatx.utils import split_nodes
from util import (Accumulator, generate_rand_array, get_model_by_name,
                  get_surrogate_by_name, settings)

parser = argparse.ArgumentParser(description='Graph Attack')
parser.add_argument('-c', '--conf', help='Config file')
args = parser.parse_args()
if args.conf is not None:
    settings.load_conf(args.conf)

settings.logger.info(f'dataset\tmodel\tsurrogate\tattack\tCLN\tATK')
for dataset_name, model_name, surrogate_name, attack_name in product(
        settings.datasets, settings.models, settings.surrogates,
        settings.attacks):
    if f'{attack_name}.{surrogate_name}.{dataset_name}' in settings.skip:
        continue

    dataset = GraphDataset(root=settings.data_dir,
                           name=dataset_name,
                           transform=T.LargestConnectedComponents())
    data = dataset[0]
    splits = split_nodes(data.y, random_state=settings.random_state)
    model = get_model_by_name(model_name)
    surrogate = get_surrogate_by_name(surrogate_name)
    attack = get_attack_by_name(attack_name)
    test_nodes = generate_rand_array(data.num_nodes, settings.sample_nodes,
                                     settings.random_state)

    metric = Accumulator(3)
    for target_node in test_nodes:
        before, after = attack(data, splits, target_node, model, surrogate)
        metric.add(before, after, 1)
        torch.cuda.empty_cache()

    settings.logger.info(
        f'{dataset_name}\t{model_name}\t{surrogate_name}\t{attack_name}\t{metric[0]/metric[2]:.3f}\t{metric[1]/metric[2]:.3f}'
    )

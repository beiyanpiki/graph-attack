import argparse
from itertools import product

import torch_geometric.transforms as T

from attack import random, sga, nettack
from greatx.datasets import GraphDataset
from greatx.nn.models import GAT, GCN, SGC
from greatx.utils import split_nodes
from util import settings, Accumulator, generate_rand_array

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
    model = globals()[model_name.upper()]
    surrogate = globals()[surrogate_name.upper()]
    attack = globals()[attack_name.lower()]
    test_nodes = generate_rand_array(data.num_nodes, settings.sample_nodes,
                                     settings.random_state)

    metric = Accumulator(3)
    for target_node in test_nodes:
        before, after = attack(data, splits, target_node, model, surrogate)
        metric.add(before, after, 1)

    settings.logger.info(
        f'{dataset_name}\t{model_name}\t{surrogate_name}\t{attack_name}\t{metric[0]/metric[2]:.3f}\t{metric[1]/metric[2]:.3f}'
    )

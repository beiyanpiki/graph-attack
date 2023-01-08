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
parser.add_argument("-c", "--conf", help="Config file", type=str)
parser.add_argument(
    "-v",
    "--verbose",
    help="Print status every {verbose} times, default 10",
    default=10,
    type=int,
)
args = parser.parse_args()
if args.conf is not None:
    settings.load_conf(args.conf)

settings.logger.info(
    f"{'':24s}|{'Dataset':^12s}|{'Model':^12s}|{'Surrogate':^12s}|{'Attack':^12s}|{'CLN':^12s}|{'ATK':^12s}|"
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
        if id % args.verbose == 0:
            s = f"After {id} of {settings.sample_nodes} attack"
            settings.logger.debug(
                f"{s:<24s}|{dataset_name.upper():^12s}|{model_name.upper():^12s}|{surrogate_name.upper():^12s}|{attack_name.upper():^12s}|{metric[0]/metric[2]:^12.4f}|{metric[1]/metric[2]:^12.4f}|"
            )

    settings.logger.info(
        f"{'Complete':24s}|{dataset_name.upper():^12s}|{model_name.upper():^12s}|{surrogate_name.upper():^12s}|{attack_name.upper():^12s}|{metric[0]/metric[2]:^12.4f}|{metric[1]/metric[2]:^12.4f}|"
    )

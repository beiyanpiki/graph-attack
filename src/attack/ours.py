from typing import Tuple, Union

import torch
import numpy as np
from greatx.attack.targeted.targeted_attacker import TargetedAttacker
from greatx.nn.models.surrogate import Surrogate
from greatx.utils import ego_graph


class OursAttack(TargetedAttacker, Surrogate):
    @torch.no_grad()
    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        *,
        tau: float = 5.0,
        freeze: bool = True,
    ):
        Surrogate.setup_surrogate(self,
                                  surrogate=surrogate,
                                  tau=tau,
                                  freeze=freeze)

        self.logits = self.surrogate(self.feat, self.edge_index,
                                     self.edge_weight).cpu()

        return self

    def set_normalize(self, state):
        # TODO: this is incorrect for models
        # with `normalize=False` by default
        for layer in self.surrogate.modules():
            if hasattr(layer, 'normalize'):
                layer.normalize = state
            if hasattr(layer, 'add_self_loops'):
                layer.add_self_loops = state

    def strongest_wrong_class(self, target, target_label):
        # logit: 代理模型预测输出
        logit = self.logits[target].clone()
        logit[target_label] = -1e4
        return logit.argmax()

    def get_subgraph(self, target, target_label, best_wrong_label):
        # K-hop sampler
        sub_nodes, sub_edges = ego_graph(self.adjacency_matrix, int(target),
                                         self.K)
        if sub_edges.size == 0:
            raise RuntimeError(
                f"The target node {int(target)} is a singleton node.")

        sub_nodes = torch.as_tensor(sub_nodes,
                                    dtype=torch.long,
                                    device=self.device)
        sub_edges = torch.as_tensor(sub_edges,
                                    dtype=torch.long,
                                    device=self.device)
        # 找到target_node的邻居节点
        neighbors = self.adjacency_matrix[target].indices

        # attacker_nodes: 找到label为best_wrong_label的所有节点
        attacker_nodes = torch.where(
            self.label == best_wrong_label)[0].cpu().numpy()
        # attacker_nodes = set(attacker_nodes) - set(neighbors)
        attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        influencers = [target]

        subgraph = self.subgraph_processing(sub_nodes, sub_edges, influencers,
                                            attacker_nodes)

        pass

    def subgraph_processing(self, sub_nodes, sub_edges, influencers,
                            attacker_nodes):
        # [1,2,3] and [2,3]
        # row = [1,1,2,2,3,3]
        # cow = [2,3,2,3,2,3]
        # dat = [0,0,1,0,0,0]
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        # 要给不存在的边计算梯度, 因此需要non_edges
        non_edges = np.row_stack([row, col])

        if self.direct_attack:  # indirect attack
            # 若邻接矩阵相连，则从non_edge里面删除
            mask = self.adjacency_matrix[row, col].A1 == 0
            non_edges = non_edges[:, mask]
        # as_tensor 浅拷贝
        non_edges = torch.as_tensor(non_edges,
                                    dtype=torch.long,
                                    device=self.device)
        attacker_nodes = torch.as_tensor(attacker_nodes,
                                    dtype=torch.long,
                                    device=self.device)

        return ()

    def attack(self,
               target,
               *,
               K: int = 2,
               target_label=None,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):
        super().attack(target,
                       target_label,
                       num_budgets=num_budgets,
                       direct_attack=direct_attack,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)
        self.set_normalize(False)
        self.K = K
        target_label = self.target_label.view(-1)
        print("target", target)
        print("target_label", target_label)
        best_wrong_label = self.strongest_wrong_class(target,
                                                      target_label).view(-1)
        best_wrong_label = best_wrong_label.to(self.device)
        subgraph = self.get_subgraph(target, target_label, best_wrong_label)

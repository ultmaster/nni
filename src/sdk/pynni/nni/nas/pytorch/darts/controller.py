import torch
from torch import nn as nn
from torch.nn import functional as F

from nni.nas.pytorch.controller import Controller
from nni.nas.pytorch.mutables import LayerChoice, InputChoice


class DartsController(Controller):
    def build(self, mutables):
        self.choices = nn.ParameterDict()
        for mutable in mutables.traverse():
            if isinstance(mutable, LayerChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length + 1))

    def device(self):
        for v in self.choices.values():
            return v.device

    def sample_search(self, mutables):
        result = dict()
        for mutable in mutables.traverse():
            if isinstance(mutable, LayerChoice):
                result[mutable.key] = F.softmax(self.choices[mutable.key], dim=-1)[:-1]
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device())
        return result

    def sample_final(self, mutables):
        result = dict()
        edges_max = dict()
        for mutable in mutables.traverse():
            if isinstance(mutable, LayerChoice):
                max_val, index = torch.max(F.softmax(self.choices[mutable.key], dim=-1)[:-1], 0)
                edges_max[mutable.key] = max_val
                result[mutable.key] = F.one_hot(index, num_classes=mutable.length).view(-1).bool()
        for mutable in mutables.traverse():
            if isinstance(mutable, InputChoice):
                weights = torch.tensor([edges_max.get(src_key, 0.) for src_key in mutable.choose_from])  # pylint: disable=not-callable
                _, topk_edge_indices = torch.topk(weights, mutable.n_chosen or mutable.n_candidates)
                selected_multihot = []
                for i, src_key in enumerate(mutable.choose_from):
                    if i not in topk_edge_indices and src_key in result:
                        result[src_key] = torch.zeros_like(result[src_key])  # clear this choice to optimize calc graph
                    selected_multihot.append(i in topk_edge_indices)
                result[mutable.key] = torch.tensor(selected_multihot, dtype=torch.bool, device=self.device())  # pylint: disable=not-callable
        return result
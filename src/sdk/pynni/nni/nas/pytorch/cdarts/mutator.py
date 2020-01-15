# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from apex.parallel import DistributedDataParallel  # pylint: disable=import-error
from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.mutator import Mutator


class RegularizedDartsMutator(DartsMutator):
    def reset(self):
        raise ValueError("You should probably call `reset_with_loss`.")

    def cut_choices(self, cut_num=2):
        # `cut_choices` is implemented but not used
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                _, idx = torch.topk(-self.choices[mutable.key], cut_num)
                with torch.no_grad():
                    for i in idx:
                        self.choices[mutable.key][i] = -float("inf")

    def reset_with_loss(self):
        self._cache, reg_loss = self.sample_search()
        return reg_loss

    def sample_search(self):
        result = super().sample_search()
        loss = []
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                def need_reg(choice):
                    return any(t in str(type(choice)).lower() for t in ["poolwithoutbn", "identity", "dilconv"])

                for i, choice in enumerate(mutable.choices):
                    if need_reg(choice):
                        norm = torch.abs(self.choices[mutable.key][i])
                        if norm < 1E10:
                            loss.append(norm)
        if not loss:
            return result, None
        return result, sum(loss)

    def export(self, logger):
        result = self.sample_final()
        if hasattr(self.model, "plot_genotype"):
            genotypes = self.model.plot_genotype(result, logger)
        return result, genotypes


class RegularizedMutatorParallel(DistributedDataParallel):
    def reset_with_loss(self):
        result = self.module.reset_with_loss()
        self.callback_queued = False
        return result

    def cut_choices(self, *args, **kwargs):
        self.module.cut_choices(*args, **kwargs)

    def export(self, logger):
        return self.module.export(logger)


class DartsDiscreteMutator(Mutator):

    def __init__(self, model, parent_mutator):
        super().__init__(model)
        self.__dict__["parent_mutator"] = parent_mutator  # avoid parameters to be included

    def sample_search(self):
        return self.parent_mutator.sample_final()

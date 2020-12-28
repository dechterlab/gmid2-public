from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from copy import copy
from sortedcontainers import SortedDict, SortedSet

from gmid2.global_constants import *
from gmid2.basics.message import Separator

from gmid2.basics.factor import Factor, Variable
from .bucket import Bucket


class MessagePassing(ABC):
    def __init__(self, *args, **kwargs):
        # Message passing algorithms form a graph of clusters separated by separators
        self.cid2cluster = SortedDict()         # cluster object
        self.edge2separator = SortedDict()      # separator object
        self.mid2message = SortedDict()         # message object that created separator object (needed?) maybe not
        super().__init__(*args, **kwargs)

    @abstractmethod
    def build_message_graph(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def schedule(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def init_propagate(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def propagate_iter(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def propagate(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def bounds(self, *args, **kwargs):
        return NotImplemented


# class FactorSeparator(FactorMixin, Separator):
class FactorSeparator():
    """
    SeparatorFactors created from Separator for storing factor
    """
    def __init__(self, factor: Optional[Factor], src:int, dest:int):
        self.factor = factor
        self.scope_vars = factor.scope_vars
        self.scope_vids = factor.scope_vids
        self.src = src
        self.dest = dest

    def update_scope(self):
        self.scope_vars = self.factor.scope_vars
        self.scope_vids = self.factor.scope_vids


class BucketFactor():
    """
    BucketFactorSet created from Bucket for storing factor
    """
    def __init__(self, factor: Factor, bucket:Bucket):
        self.cid = bucket.bid
        self.var_label = bucket.var_label
        self.ind = bucket.ind
        self.factor = factor
        self.scope_vars = factor.scope_vars
        self.scope_vids = bucket.scope_vids

        # don't need to trace how it was created but leave it for now; need var_label, ind, scope_vids from bucket
        # super().__init__(bucket.var_label, bucket.ind, bucket.vids_fids, bucket.fids, bucket.vids_mids, bucket.mids)
        # self.bid = self.cid

    def __str__(self):
        output = type(self).__name__ + "_{}:({}, {})[v:[{}]]"
        return output.format(self.cid, self.var_label, self.ind, ",".join(str(el) for el in self.scope_vids))

    def update_scope(self):
        self.scope_vars = self.factor.scope_vars
        self.scope_vids = self.factor.scope_vids

    __repr__ = __str__


class WeightedBucketFactor():
    """
    Use this for WBM or WMB-MM.
    Only singe weight associated with WeightedBucketFactor because only var_label will be eliminated from here
    """
    def __init__(self, var: Variable, factor: Factor, bucket:Bucket):
        self.cid = bucket.bid
        self.var_label = bucket.var_label
        self.ind = bucket.ind
        self.factor = factor
        self.scope_vars = factor.scope_vars
        self.scope_vids = bucket.scope_vids

        self.var = var
        self.vid2varset = {var.vid: SortedSet([var]) for var in self.scope_vars}
        self.eliminator = self.scope_vars - SortedSet( [self.var] )
        self.weight = None
        self.weighted_marginal = None
        self.next_cid = None

    def update_scope(self):
        self.scope_vars = self.factor.scope_vars
        self.scope_vids = self.factor.scope_vids

    def __str__(self):
        output = type(self).__name__ + "_{}:({}, {})[v:[{}]]"
        return output.format(self.cid, self.var_label, self.ind,
                             ",".join(str(el) for el in self.scope_vids)
                             )
    __repr__ = __str__

    def update_weighted_marginal(self)->Factor:
        f = copy(self.factor)       # log scale
        if self.weight == 0.0:
            f = f.max_marginal(self.eliminator, inplace=True)
        else:
            f = f.lse_pnorm_marginal(self.eliminator, 1.0 / self.weight, inplace=True)        # w and p inverse
        self.weighted_marginal = f
        return self.weighted_marginal

    def update_weighted_marginal_on(self, vid:int)->Factor:
        f = copy(self.factor)       # log scale
        eliminator = self.scope_vars - self.vid2varset[vid]     # eliminate all except vid
        if self.weight == 0.0:
            f = f.max_marginal(eliminator, inplace=True)
        else:
            f = f.lse_pnorm_marginal(eliminator, 1.0 / self.weight, inplace=True)        # w and p inverse
        self.weighted_marginal = f
        return self.weighted_marginal

    def moment_matching_factor(self, total_weighted_marginals:Factor)->None:
        if self.weight == 0.0:
            self.factor = self.factor + total_weighted_marginals - self.weighted_marginal
        else:
            self.factor = self.factor + self.weight* total_weighted_marginals - self.weighted_marginal

    def update_bound(self):
        f = copy(self.factor)  # log scale          # todo fix this in LIMID
        if len(f.scope_vars) > 1:
            for vv in list(f.scope_vars):
                if self.weight == 0.0:
                    f = f.max_marginal(SortedSet([vv]), inplace=True)
                else:
                    f = f.lse_pnorm_marginal(SortedSet([vv]), 1.0 / self.weight, inplace=True)  # w and p inverse

        else:
            if self.weight == 0.0:
                f = f.max_marginal(f.scope_vars, inplace=True)
            else:
                f = f.lse_pnorm_marginal(f.scope_vars, 1.0 / self.weight, inplace=True)        # w and p inverse
        return f


class WeightedOrderedBucketFactor():
    """
    Use this for GDD.
    All variables now associated with weights and it also comes with a total elimination order
    """
    def __init__(self, factor: Factor, bucket:Bucket, vid_elim_order:List[int]):
        self.cid = bucket.bid
        self.var_label = bucket.var_label
        self.ind = bucket.ind
        self.factor = factor
        self.factor_prev = None

        # self.vars_ = SortedSet(factor.scope_vars)       # maintain separate object
        self.scope_vars = factor.scope_vars
        self.scope_vids = bucket.scope_vids

        self.vid_elim_order = vid_elim_order
        self.vid_elim_order_local = [vid for vid in vid_elim_order if vid in self.scope_vids]

        self.vid2var = {var.vid:var for var in self.scope_vars}
        self.vid2varset = {var.vid: SortedSet( [var] ) for var in self.scope_vars}

        self.pseudo_belief = None     # one per var? as a dict?
        self.local_bound = None
        self.pseudo_belief_prev = None
        self.local_bound_prev = None

        self.vid2weight = {vid: -1.0 for vid in self.scope_vids}
        self.vid2weight_prev = {vid: -1.0 for vid in self.scope_vids}

    def update_scope(self):
        self.scope_vars = self.factor.scope_vars
        self.scope_vids = self.factor.scope_vids

    def __str__(self):
        output = type(self).__name__ + "_{}:({}, {})[v:[{}]]"
        return output.format(self.cid, self.var_label, self.ind,
                             ",".join(str(vid)+"^{:.2f}".format(self.vid2weight[vid]) for vid in self.scope_vids)
                             )
    __repr__ = __str__

    def update_pseudo_belief(self)->Factor:
        if self.pseudo_belief is None:
            lnZ0 = copy(self.factor)
            lnmu = 0.0
            for vid in self.vid_elim_order_local:
                eliminator = self.vid2varset[vid]
                w_exponent = WEPS if self.vid2weight[vid] < WEPS else self.vid2weight[vid]  # smooth max
                if w_exponent <= WEPS:
                    lnZ1 = lnZ0.max_marginal(eliminator, inplace=False)
                else:
                    lnZ1 = lnZ0.lse_pnorm_marginal(eliminator, 1.0/w_exponent, inplace=False)
                lnZ0 -= lnZ1        # inplace to lnZ0 subtract lnZ1 and store it back to lnZ0       -inf - -inf??
                # lnZ0.table[np.isnan(lnZ0.table)] = -INF     # required?
                lnZ0 *= 1.0 / w_exponent
                lnmu += lnZ0
                lnZ0 = lnZ1
            self.pseudo_belief = lnmu.exp()
        return self.pseudo_belief

    def update_bound(self)->float:
        if self.local_bound is None:
            f = copy(self.factor)
            for vid in self.vid_elim_order_local:
                eliminator = self.vid2varset[vid]
                # w_exponent = self.vid2weight[vid]
                w_exponent = WEPS if self.vid2weight[vid] < WEPS else self.vid2weight[vid]  # smooth max
                if w_exponent <= WEPS:
                    f = f.max_marginal(eliminator, inplace=False)
                else:
                    f = f.lse_pnorm_marginal(eliminator, 1.0/w_exponent, inplace=False)
            self.local_bound = f
        return self.local_bound

    def update_partial_bound(self, scope_vids:Iterable[Variable])->Factor:
        f = copy(self.factor)
        for vid in self.vid_elim_order_local:
            if vid not in scope_vids:
                eliminator = self.vid2varset[vid]
                # w_exponent = self.vid2weight[vid]
                w_exponent = WEPS if self.vid2weight[vid] < WEPS else self.vid2weight[vid]  # smooth max
                if w_exponent <= WEPS:
                    f = f.max_marginal(eliminator, inplace=False)
                else:
                    f = f.lse_pnorm_marginal(eliminator, 1.0/w_exponent, inplace=False)
        return f
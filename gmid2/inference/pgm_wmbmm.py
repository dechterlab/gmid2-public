"""
PgmWMBMM do entropy and moment matching over FW message passing
    - mini-bucket tree
    - summation, maximization, marginal map
    - Pgm assumes all functions are multiplicative
"""
from sortedcontainers import SortedSet, SortedDict
import networkx as nx

from .helper import *
from gmid2.global_constants import *
from gmid2.basics.message import Separator
from .message_passing import MessagePassing, WeightedBucketFactor
from .bucket import BucketGraph


class PgmWMBMM(MessagePassing, nx.DiGraph):
    def __init__(self, gm: GraphicalModel, elim_order:List[int]):
        self.gm = gm
        self.elim_order = elim_order
        self.vid2cid = SortedDict()  # clusters containing vid, for reading marginals if wanted
        self.var_label2cid = SortedDict()  # mini-buckets labeld by var_label2cid
        # self.global_bound = 0.0
        self.bound_nodes = []
        super().__init__()

    def build_message_graph(self, bucket_tree:BucketGraph):
        for bid in bucket_tree:
            bucket = bucket_tree.bid2bucket[bid]
            factors = [self.gm.fid2f[fid] for fid in bucket.fids]
            if not factors:
                f = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
            else:
                f = combine_factor_list(factors, is_log=True)
                if f.scope_vids < bucket.scope_vids:       # subset
                    f_const = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
                    combine_factors(f, f_const, is_log=True) # inplace to f

            cluster = WeightedBucketFactor(self.gm.vid2var[bucket.var_label], f, bucket)
            self.cid2cluster[cluster.cid] = cluster
            self.add_node(node_for_adding=cluster.cid)

        for src_bid, dest_bid, attr in bucket_tree.edges.data():    # no FactorSeparator send it directly through temp
            if attr['msg_type'] == Separator.__name__:
                self.cid2cluster[src_bid].next_cid = dest_bid

    def schedule(self, *args, **kwargs):
        for cid in self:
            if self.cid2cluster[cid].next_cid is None:
                self.bound_nodes.append(cid)

        for cid in self:
            cluster = self.cid2cluster[cid]
            if cluster.var_label not in self.var_label2cid:
                self.var_label2cid[cluster.var_label] = SortedSet()
            self.var_label2cid[cluster.var_label].add(cid)
            if cluster.var_label not in self.var_label2cid:
                self.var_label2cid[cluster.var_label] = SortedSet()
            self.var_label2cid[cluster.var_label].add(cid)

    def init_propagate(self):
        pass
        # for cid in self:
        #     cluster = self.cid2cluster[cid]
        #     if cluster.var_label not in self.var_label2cid:
        #         self.var_label2cid[cluster.var_label] = SortedSet()
        #     self.var_label2cid[cluster.var_label].add(cid)
        #     if cluster.var_label not in self.var_label2cid:
        #         self.var_label2cid[cluster.var_label] = SortedSet()
        #     self.var_label2cid[cluster.var_label].add(cid)

    def propagate(self):
        for vid in self.elim_order:
            if vid not in self.var_label2cid:
                continue
            buckets_with_vid = self.var_label2cid[vid]
            n_buckets, current_var_type = len(buckets_with_vid), self.gm.vid2type[vid]
            init_weight = 1.0 / n_buckets if current_var_type == TYPE_CHANCE_VAR else 0.0
            for cid in buckets_with_vid:
                self.cid2cluster[cid].weight = init_weight

            if n_buckets > 1:
                # moment matching
                weighted_marginals = []
                for cid in buckets_with_vid:
                    # f = self.cid2cluster[cid].update_weighted_marginal()
                    f = self.cid2cluster[cid].update_weighted_marginal_on(vid)
                    weighted_marginals.append(f)
                total_weighted_marginals = combine_factor_list(weighted_marginals, is_log=True)
                if current_var_type == TYPE_DECISION_VAR:
                    total_weighted_marginals = total_weighted_marginals * (1.0/n_buckets)
                for cid in buckets_with_vid:
                    self.cid2cluster[cid].moment_matching_factor(total_weighted_marginals)

            # send message
            for cid in buckets_with_vid:
                cluster_src = self.cid2cluster[cid]
                if cluster_src.next_cid is not None:
                    cluster_dest = self.cid2cluster[cluster_src.next_cid]
                    eliminator = cluster_src.scope_vars - cluster_dest.scope_vars
                    f = copy(cluster_src.factor)
                    if current_var_type == TYPE_CHANCE_VAR:
                        if cluster_src.weight == 1.0:
                            f = f.lse_marginal(eliminator, inplace=True)     # log scale computation!
                        else:
                            f = f.lse_pnorm_marginal(eliminator, 1.0/cluster_src.weight, inplace=True)
                    else:
                        f = f.max_marginal(eliminator, inplace=True)
                    combine_factors(cluster_dest.factor, f, self.gm.is_log)     # inplace multiply into destination
                # else:   # no destination!
                #     f = copy(cluster_src.factor)
                #     if current_var_type == TYPE_CHANCE_VAR:
                #         if cluster_src.weight == 1.0:
                #             f = f.lse_marginal(cluster_src.scope_vars, inplace=True)     # log scale computation!
                #         else:
                #             f = f.lse_pnorm_marginal(cluster_src.scope_vars, 1.0 / cluster_src.weight, inplace=True)
                #     else:
                #         f = f.max_marginal(cluster_src.scope_vars, inplace=True)
                #     self.global_bound = self.global_bound + f        # log scale only!


    def propagate_iter(self):
        self.propagate()

    def bounds(self):
        bound = 0.0     # log scale
        for cid in self.bound_nodes:
            bound += self.bounds_at(cid)
        return bound

    def bounds_at(self, cid:int):
        return self.cid2cluster[cid].update_bound()



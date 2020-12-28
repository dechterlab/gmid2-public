"""
PgmBTE do FW/BW message passing over a tree
    - bucket tree, mini-bucket tree, join tree
    - summation, maximization, marginal map (no BW implemented is it as simple as elim max? -- check if needed)
    - Pgm assumes all functions are multiplicative
"""
from typing import Iterable, Tuple, List, Union
from copy import copy
import itertools
from sortedcontainers import SortedSet, SortedDict
import networkx as nx

from gmid2.global_constants import *
from .helper import *
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Variable, Factor
from gmid2.basics.directed_network import topo_sort
from gmid2.basics.message import Separator

from .message_passing import MessagePassing, BucketFactor, FactorSeparator
from .bucket import BucketGraph


class PgmBTE(MessagePassing, nx.DiGraph):
    def __init__(self, gm: GraphicalModel, elim_order:List[int]):
        self.gm = gm
        self.elim_order = elim_order        # one should know elim order anyway so provide it
        self.vid2cid = SortedDict()         # clusters containing vid, for reading marginals if wanted
        self.var_label2cid = SortedDict()   # mini-buckets labeld by var_label2cid
        # self.bucket_tree = bucket_tree    # don't increase reference counter; it's independent object that copies
        self.bound_nodes = []
        super().__init__()

    def build_message_graph(self, bucket_tree:BucketGraph):
        """
        from a tree of buckets, create BucketFactor and FactorSeparator and prepare message passing
        """
        for bid in bucket_tree:
            bucket = bucket_tree.bid2bucket[bid]
            factors = [self.gm.fid2f[fid] for fid in bucket.fids]
            if not factors:
                f = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
            else:
                f = combine_factor_list(factors, self.gm.is_log)
                if f.scope_vids < bucket.scope_vids:       # subset
                    f_const = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
                    combine_factors(f, f_const, self.gm.is_log) # inplace to f
            cluster = BucketFactor(f, bucket)      # initially messages are all empty how to init them?
            self.cid2cluster[cluster.cid] = cluster
            self.add_node(node_for_adding=cluster.cid)

        for src_bid, dest_bid, attr in bucket_tree.edges.data():
            if attr['msg_type'] == Separator.__name__:
                cluster_src, cluster_dest = self.cid2cluster[src_bid], self.cid2cluster[dest_bid]
                sep_vids = cluster_src.scope_vids & cluster_dest.scope_vids
                f = const_factor(self.gm, sep_vids, 1.0, TYPE_PROB_FUNC)
                self.edge2separator[src_bid, dest_bid] = FactorSeparator(f, src_bid, dest_bid)
                self.add_edge(src_bid, dest_bid, direction=TYPE_MSG_FW)
                f = const_factor(self.gm, sep_vids, 1.0, TYPE_PROB_FUNC)
                self.edge2separator[dest_bid, src_bid] = FactorSeparator(f, dest_bid, src_bid)
                self.add_edge(dest_bid, src_bid, direction=TYPE_MSG_BW)

    # def elim_order_from_graph(self, bucket_tree:BucketGraph):
    #     # only usable if input is exact tree
    #     self.elim_order = []
    #     for bid in itertools.chain.from_iterable(topo_sort(bucket_tree)):
    #         self.elim_order.append(bucket_tree.bid2bucket[bid].var_label)
    #     return self.elim_order

    def schedule(self, bucket_tree:BucketGraph):
        # edge schedule from bucket tree
        self.fw_schedule = []
        self.bw_schedule = []
        sort_key = lambda cid: (self.elim_order.index( self.cid2cluster[cid].var_label), cid)
        for blocks in topo_sort(bucket_tree):
            for cid in sorted(blocks, key=sort_key): # earlier elim, earlier created node
                # cnt = 0
                for _, dest, attr in self.out_edges(cid, data=True):
                    if attr['direction'] == TYPE_MSG_FW:
                        self.fw_schedule.append((cid, dest))
                        self.bw_schedule.append((dest, cid))
                        # cnt += 1
                # assert cnt<=1, "only one or less message outgoing"
        self.bw_schedule.reverse()
        for cid in bucket_tree:
            if bucket_tree.out_degree(cid) == 0:
                self.bound_nodes.append(cid)
            else:
                for _, dest, attr in bucket_tree.out_edges(cid, data=True):
                    if attr['msg_type'] == Separator.__name__:
                        break
                else:
                    self.bound_nodes.append(cid)

    def init_propagate(self):
        for cid in self:
            cluster = self.cid2cluster[cid]
            for vid in cluster.scope_vids:
                if vid not in self.vid2cid:
                    self.vid2cid[vid] = SortedSet()
                self.vid2cid[vid].add(cluster.cid)
            if cluster.var_label not in self.var_label2cid:
                self.var_label2cid[cluster.var_label] = SortedSet()
            self.var_label2cid[cluster.var_label].add(cid)

    def propagate(self, direction:str, edge_schedule:Iterable[Tuple[int, int]]):
        """
        compute FW/BW messages and place them on the edge
        functions at the cluster is invariant
        """
        for src, dest in edge_schedule:
            # pull messages from separator edges
            current_cluster, current_sep = self.cid2cluster[src], self.edge2separator[src, dest]
            current_sep.factor = copy(current_cluster.factor)       # bring-in to ensure full-scoped function
            for pa, _, attr in self.in_edges(src, data=True):
                if pa == dest:
                    continue
                pa_sep_factor = self.edge2separator[pa, src].factor   # bring msg; cte only one factor
                combine_factors(current_sep.factor, pa_sep_factor, self.gm.is_log)
            eliminator = current_cluster.scope_vars - current_sep.scope_vars
            for var in eliminator:
                self._elim_var_ip(var, current_sep.factor)      # in-place elimination

    def propagate_iter(self, bw_iter=False):
        self.propagate(TYPE_MSG_FW, self.fw_schedule)
        if bw_iter:
            self.propagate(TYPE_MSG_BW, self.bw_schedule)

    def bounds(self):
        bound = 0.0 if self.gm.is_log else 1.0
        for cid in self.bound_nodes:
            if self.gm.is_log:
                bound += self.bounds_at(cid)
            else:
                bound *= self.bounds_at(cid)
        return bound

    def bounds_at(self, cid:int)->float:
        cluster = self.cid2cluster[cid]
        bounding_f = copy(cluster.factor)
        for pa in self.predecessors(cid):
            sep_factor = self.edge2separator[pa, cid].factor
            combine_factors(bounding_f, sep_factor, self.gm.is_log)
        for var in cluster.scope_vars:
            bounding_f = self._elim_var_ip(var, bounding_f)     # return reference; float cannot be Factor;
        return bounding_f

    def _elim_var_ip(self, var:Variable, f:Factor)-> Union[float, Factor]:  #
        eliminator = SortedSet( [var] )     # set operation used inside
        if var.type == TYPE_CHANCE_VAR:
            if self.gm.is_log:
                m = np.max(f.table)
                f -= m
                f = f.lse_marginal(eliminator, inplace=True)
                f += m
                return f
            else:
                return f.sum_marginal(eliminator, inplace=True)
        elif var.type == TYPE_DECISION_VAR:
            return f.max_marginal(eliminator, inplace=True)
        else:
            assert False, "{} not supported".format(var.type)


class PgmBTELog(PgmBTE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def propagate(self, direction:str, edge_schedule:Iterable[Tuple[int, int]]):
        for src, dest in edge_schedule:
            current_cluster, current_sep = self.cid2cluster[src], self.edge2separator[src, dest]
            current_sep.factor = copy(current_cluster.factor)       # bring-in to ensure full-scoped function
            m = extract_max(current_sep.factor, is_log=True)
            for pa, _, attr in self.in_edges(src, data=True):
                if pa == dest:
                    continue
                pa_sep_factor = self.edge2separator[pa, src].factor   # bring msg; cte only one factor
                combine_factors(current_sep.factor, pa_sep_factor, self.gm.is_log)
                m += extract_max(current_sep.factor, is_log=True)
            eliminator = current_cluster.scope_vars - current_sep.scope_vars
            for var in eliminator:
                self._elim_var_ip(var, current_sep.factor)      # in-place elimination
            current_sep.factor += m

    def _elim_var_ip(self, var:Variable, f:Factor)-> Union[float, Factor]:  #
        eliminator = SortedSet( [var] )     # set operation used inside
        m = extract_max(f, is_log=True)
        if var.type == TYPE_CHANCE_VAR:
            f = f.lse_marginal(eliminator, inplace=True)
        elif var.type == TYPE_DECISION_VAR:
            f = f.max_marginal(eliminator, inplace=True)
        f += m
        return f

from typing import Iterable, Tuple, List, Text, Optional
from copy import copy, deepcopy
import itertools
from sortedcontainers import SortedSet, SortedDict
import networkx as nx

from gmid2.global_constants import *
from gmid2.inference.helper import zero_factor, combine_factor_list, combine_factors
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Variable, Factor
from gmid2.basics.directed_network import topo_sort
from gmid2.basics.undirected_network import PrimalGraph, call_variable_ordering
from gmid2.inference.message_passing import MessagePassing

from .pgm_bte import PgmBTELog
from .submodel import SubModelGraph
from .bucket import bucket_tree_decomposition


class StPgmCluster():
    def __init__(self, cid):
        self.cid = cid
        self.dec_vids: List[int] = None
        self.util_fids: SortedSet = None
        self.util_fid2bid = None
        self.bw_in: List[int] = None
        self.bw_out: int = None
        self.mp = None
        self.local_eu = 0.0


class FactorSetSeparator():
    def __init__(self, src: int, dest: int):
        self.src = src
        self.dest = dest
        self.factors = []
        self.ind2src_bid = dict()       # src is the source bucket of the factor, extract from source
        self.ind2dest_bid = dict()      # this bid is common for all message graphs created from same bucket tree
        self.fid2ind = dict()
        self.ind = 0

    def add_factor(self, f: Factor, src_bid: Optional[int], dest_bid:Optional[int]):
        self.factors.append(f)
        self.ind2src_bid[self.ind] = src_bid
        self.ind2dest_bid[self.ind] = dest_bid
        self.fid2ind[f.fid] = self.ind
        self.ind += 1

    def update_factor(self, ind, new_factor:Factor):
        del self.fid2ind[self.factors[ind].fid]
        self.factors[ind] = new_factor
        self.fid2ind[new_factor.fid] = ind

    def update_bid_link(self, ind: int, src_bid: int, dest_bid: int):
        if src_bid is not None:
            self.ind2src_bid[ind] = src_bid
        if dest_bid is not None:
            self.ind2dest_bid[ind] = dest_bid


class StBTEBw(MessagePassing, nx.DiGraph):
    def __init__(self, gm: GraphicalModel, st: SubModelGraph, alpha:float=ALPHA):  # todo read and pass svo
        self.gm = gm
        self.st = st
        self.alpha=alpha
        self.bound_nodes = []
        self.const_eu = 0.0
        super().__init__()

    def build_message_graph(self):
        # create cluster nodes
        for sid in self.st:
            pgm_cluster = StPgmCluster(sid)
            self.cid2cluster[pgm_cluster.cid] = pgm_cluster
            self.add_node(pgm_cluster.cid)
        # visit nodes in st, create bucket trees, separators
        for sid in itertools.chain.from_iterable(topo_sort(self.st)):
            submodel = self.st.sid2submodel[sid]
            pgm_cluster = self.cid2cluster[sid]
            pgm_cluster.dec_vids = submodel.nids2vids(submodel.rel_d)     # internal variables and functions
            pgm_cluster.util_fids = submodel.internal_util_fids         # external functions, source/dest of messages
            pgm_cluster.bw_in = [src for src, _, d in self.in_edges(sid, data=True) if d['direction'] == TYPE_MSG_BW]
            pgm_cluster.bw_out = None
            for _, dest in self.st.out_edges(sid):
                pgm_cluster.bw_out = dest
            bw_in_fids = SortedSet()
            local_gm = self.gm.project(submodel.internal_fids)  # local gm projected on internal functions
            for src in pgm_cluster.bw_in:
                for f in self.edge2separator[src, sid].factors: # factors from bw in separators
                    local_gm.add_factor(f)
                    bw_in_fids.add(f.fid)
            for vid in local_gm.vid2var:
                local_gm.change_var_type(vid, type=TYPE_CHANCE_VAR)     # convert all vars to chance
            for vid in pgm_cluster.dec_vids:
                local_gm.change_var_type(vid, type=TYPE_DECISION_VAR)
            pg = PrimalGraph()
            pg.build_from_scopes(local_gm.scope_vids)
            pvo_vid = [submodel.nids2vids(submodel.rel_h)]
            pvo_vid.append(submodel.nids2vids(submodel.rel_d))
            bw_ordering, bw_iw = call_variable_ordering(pg, 5000, pvo_vid)
            print("sid\t\t{}\nw\t\t{}\nvo\t\t{}".format(sid, bw_iw, " ".join(str(el) for el in bw_ordering)))
            half_bt = bucket_tree_decomposition(local_gm, bw_ordering)
            # create BW out separator and link messages to internal buckets
            if pgm_cluster.bw_out is not None:
                bw_out_sep = FactorSetSeparator(sid, pgm_cluster.bw_out)
                self.add_edge(sid, pgm_cluster.bw_out, direction=TYPE_MSG_BW)
                self.edge2separator[sid, pgm_cluster.bw_out] = bw_out_sep

                last_dec_vid = bw_ordering[-1]
                dec_bids = half_bt.find_buckets_with_label(last_dec_vid)
                for dec_bid in dec_bids:
                    scope_vids = half_bt.bid2bucket[dec_bid].scope_vids & submodel.nids2vids(submodel.rel_o_out)
                    if scope_vids:
                        v = zero_factor(local_gm, scope_vids, TYPE_UTIL_FUNC)
                        bw_out_sep.add_factor(v, src_bid=dec_bid, dest_bid=None)
            # update link in src->sid v_in and sid->src q_out separator
            for src in pgm_cluster.bw_in:
                bw_in_sep = self.edge2separator[src, sid]
                for f in bw_in_sep.factors:
                    bid = half_bt.find_bucket_with_fid(f.fid)
                    ind = bw_in_sep.fid2ind[f.fid]
                    bw_in_sep.update_bid_link(ind, src_bid=None, dest_bid=bid)
            # create message passing object
            internal_gm = local_gm.exclude(bw_in_fids)
            half_bt.remove_fids(bw_in_fids)
            pgm_cluster.mp = StPgmBTE(internal_gm, bw_ordering)
            pgm_cluster.mp.build_message_graph(half_bt)
            pgm_cluster.mp.schedule(half_bt)
            for fid in bw_in_fids:
                f = local_gm.fid2f[fid]
                for dec_vid in pgm_cluster.dec_vids:
                    dec_var = local_gm.vid2var[dec_vid]
                    self._change_var_type_f(f, var=dec_var, type=TYPE_CHANCE_VAR)


    def schedule(self):
        self.bw_schedule = []
        for blocks in topo_sort(self):
            for sid in sorted(blocks):
                self.bw_schedule.append(sid)
        for cid in self:
            if self.cid2cluster[cid].bw_out is None:
                self.bound_nodes.append(cid)

    def init_propagate(self, *args, **kwargs):
        for cid in self:
            pgm_cluster = self.cid2cluster[cid]
            pgm_cluster.mp.init_propagate()

    def propagate_iter(self):
        self.propagate()

    def propagate(self):
        for cid in self.bw_schedule:
            self.bw(cid)

    def bounds(self):
        global_bound = self.const_eu
        for cid in self.bound_nodes:
            local_eu = self.cid2cluster[cid].local_eu
            if DEBUG__:
                print("sid:{} local_eu:\t\t{}".format(cid, local_eu))
            global_bound += local_eu
        global_bound /= self.alpha
        return global_bound

    def bw(self, cid:int):
        pgm_cluster = self.cid2cluster[cid]
        # bring in v_in
        if pgm_cluster.bw_in:
            for src in pgm_cluster.bw_in:
                bw_sep_in = self.edge2separator[src, cid]
                for ind, f in enumerate(bw_sep_in.factors):
                    for dec_vid in pgm_cluster.dec_vids:
                        dec_var = pgm_cluster.mp.gm.vid2var[dec_vid]
                        chance_var = Variable(dec_vid, dec_var.dim, TYPE_CHANCE_VAR)
                        self._change_var_type_f(f, chance_var, TYPE_DECISION_VAR)
                    bid = bw_sep_in.ind2dest_bid[ind]
                    pgm_cluster.mp.add_factor_at(bid, f, True)
                    pgm_cluster.mp.gm.add_factor(f)
        # message passing
        pgm_cluster.mp.propagate_iter()
        # extract v_out
        if pgm_cluster.bw_out is None:
            bte_bound = pgm_cluster.mp.bounds()
            self.cid2cluster[cid].local_eu =  bte_bound
            # log gdd bound E[ u(x) ] <= log E [ e^u(x)]
            # E[ u(x) ] ~ 1/a log E[ e^au(x) ]                  # div by alpha if alpha converted
        else:
            bw_sep_out = self.edge2separator[cid, pgm_cluster.bw_out]
            for ind in range(len(bw_sep_out.factors)):
                bid = bw_sep_out.ind2src_bid[ind]
                scope_vars = bw_sep_out.factors[ind].scope_vars
                f = pgm_cluster.mp.get_marginal_at(bid, scope_vars)
                f.type = TYPE_UTIL_FUNC
                bw_sep_out.update_factor(ind, f)

    @staticmethod
    def _change_var_type_f(f: Factor, var: Variable, type: Text) -> None:
        if var in f.scope_vars:
            f.scope_vars.remove(var)
            f.scope_vars.add(Variable(var.vid, var.dim, type))


class StPgmBTE(PgmBTELog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_marginal_at(self, cid:int, scope_vars:Iterable[Variable])->Factor:
        f = copy(self.cid2cluster[cid].factor)
        for pa in self.predecessors(cid):
            sep_factor = self.edge2separator[pa, cid].factor
            combine_factors(f, sep_factor, self.gm.is_log)

        eliminator = f.scope_vars - scope_vars
        for var in eliminator:  # if collection chnages while for loop, bug
            f = self._elim_var_ip(var, f)
        return f

    def get_factor_at(self, cid:int)->Factor:
        f = copy(self.cid2cluster[cid].factor)
        for pa in self.predecessors(cid):
            sep_factor = self.edge2separator[pa, cid].factor
            combine_factors(f, sep_factor, self.gm.is_log)
        return f

    def add_factor_at(self, cid:int, f: Factor, is_log:bool):
        if is_log:
            self.cid2cluster[cid].factor += f
        else:
            self.cid2cluster[cid].factor *= f

    def remove_factor_at(self, cid:int, f: Factor, is_log:bool):
        if is_log:
            self.cid2cluster[cid].factor -= f
        else:
            self.cid2cluster[cid].factor /= f

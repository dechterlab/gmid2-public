"""
PgmGDD do FW/BW message passing over a tree
    - bucket tree, mini-bucket tree, join tree
    - summation, maximization, marginal map (no BW implemented is it as simple as elim max? -- check if needed)
    - Pgm assumes all functions are multiplicative
"""
from typing import Iterable, Tuple, List
from copy import copy
import itertools
from sortedcontainers import SortedSet, SortedDict
import networkx as nx

from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Variable, Factor
from gmid2.basics.directed_network import topo_sort
from gmid2.basics.message import Separator
from gmid2.global_constants import *

from .helper import const_factor, combine_factors, combine_factor_list, extract_max
from .message_passing import MessagePassing, WeightedOrderedBucketFactor
from .bucket import BucketGraph


class PgmGDD(MessagePassing, nx.DiGraph):
    def __init__(self, gm: GraphicalModel, elim_order:List[int], iter_options: dict(), opt_options:dict()):
        self.gm = gm
        self.vid_elim_order = elim_order        # one should know elim order anyway so provide it
        self.vid2cid = SortedDict()             # clusters containing vid
        self.var_label2cid = SortedDict()       # mini-buckets labeld by var_label2cid
        self.iter_options = iter_options        # iteration limit, time limit, etc
        self.opt_options = opt_options          # optimization parameters

        self.global_const = 0.0         #if self.gm.is_log else 1.0      # add this when returning global bound
        self.global_bound = INF
        self.ans = INF

        self.wgt_grad_target1 = SortedDict()
        self.wgt_grad_target2 = SortedDict()
        super().__init__()

    def build_message_graph(self, join_graph:BucketGraph):
        for bid in join_graph:
            bucket = join_graph.bid2bucket[bid]
            factors = [self.gm.fid2f[fid] for fid in bucket.fids]
            if not factors:
                f = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
            else:
                f = combine_factor_list(factors, self.gm.is_log)
                if f.scope_vids < bucket.scope_vids:       # subset
                    f_const = const_factor(self.gm, bucket.scope_vids, 1.0, TYPE_PROB_FUNC)
                    combine_factors(f, f_const, self.gm.is_log) # inplace to f
            cluster = WeightedOrderedBucketFactor(f, bucket, self.vid_elim_order)
            self.cid2cluster[cluster.cid] = cluster
            self.add_node(node_for_adding=cluster.cid)      # same as PgmBTE except  WeightedOrderedBucketFactor

        for src_bid, dest_bid in join_graph.edges():
            self.add_edge(src_bid, dest_bid, direction=TYPE_MSG_FW)     # for scheduling

    def schedule(self):
        self.weight_schedule = self.vid_elim_order
        self.cost_schedule = []
        sort_key = lambda cid: (self.vid_elim_order.index( self.cid2cluster[cid].var_label), cid)
        for blocks in topo_sort(self):      # directed join graph
            for cid in sorted(blocks, key=sort_key): # earlier elim, earlier created node
                for _, dest in self.out_edges(cid):
                    self.cost_schedule.append( (cid, dest) )
        # assert len(self.cost_schedule) == self.number_of_edges()

        # find buckets with the same vid
        for cid in self:
            cluster = self.cid2cluster[cid]
            for vid in cluster.scope_vids:
                if vid not in self.vid2cid:
                    self.vid2cid[vid] = SortedSet()
                self.vid2cid[vid].add(cid)

            if cluster.var_label not in self.var_label2cid:
                self.var_label2cid[cluster.var_label] = SortedSet()
            self.var_label2cid[cluster.var_label].add(cid)

    def init_propagate(self):
        # initialize weights, uniform weights
        for vid in self.vid2cid:
            if self.gm.vid2type[vid] == TYPE_DECISION_VAR:
                init_weight = 0.0
            else:
                n_buckets = len(self.vid2cid[vid])
                init_weight = 1.0/n_buckets

            for cid in self.vid2cid[vid]:
                self.cid2cluster[cid].vid2weight[vid] = init_weight

        # extract max for all functions
        # self._extract_max()

    def propagate_iter(self):
        if 'iter_limit' in self.iter_options:
            iter_limit = self.iter_options['iter_limit']
        else:
            iter_limit = 100
        if 'cutoff_diff' in self.iter_options:
            cutoff_diff = self.iter_options['cutoff_diff']
        else:
            cutoff_diff = 0.0
        from time import time
        t00 = time()
        for iter in range(iter_limit):
            t0 = time()
            self._extract_max()

            diff, bd = self.propagate()
            t1 = time()
            if DEBUG__:
                print("{}\t{}\t{}\t{}\t{}".format(iter+1, t1-t00, t1-t0, diff, bd))
            if diff <= cutoff_diff:
                break

    def propagate(self):
        init_global_bound = self.bounds()

        previous_global_bound = init_global_bound
        for vid in self.weight_schedule:
            if self.gm.vid2type[vid] == TYPE_DECISION_VAR:
                continue
            if len(self.vid2cid[vid]) <= 1:     # no weight split
                continue
            self.update_weights_for_vid_np(vid)

        new_global_bound = self.bounds()        # after weight update
        if self.ans > new_global_bound:         # global update (necessary? as always decreasing?) debugging
            self.ans = new_global_bound
        # assert new_global_bound <= previous_global_bound, "bound should improve new:{} prev:{}".format(new_global_bound, previous_global_bound)

        previous_global_bound = new_global_bound
        for src, dest in self.cost_schedule:
            self.update_costs_for_edge(src, dest)

        new_global_bound = self.bounds()        # after weight update
        if self.ans > new_global_bound:         # global update (necessary? as always decreasing?) debugging
            self.ans = new_global_bound
        # assert new_global_bound <= previous_global_bound, "bound should improve"

        total_diff = init_global_bound - new_global_bound
        return total_diff, new_global_bound

    def _extract_max(self):
        for cid in self:
            m = extract_max(self.cid2cluster[cid].factor, is_log=True)
            # assert isinstance(m, float)
            self.global_const += m      # sutbract m from there
            self.cid2cluster[cid].pseudo_belief = self.cid2cluster[cid].local_bound = None  # invalidate

    def bounds(self)->float:
        self.global_bound = self.global_const
        for cid in self:
            f = self.cid2cluster[cid].update_bound()      # return float
            # assert isinstance(f, float)
            self.global_bound += f
        return self.global_bound

    ################################################################################################
    def update_costs_for_edge(self, src:int, dest:int):
        gd_steps, tolerance = self.opt_options['gd_steps'], self.opt_options['tolerance']
        obj_0, obj_1, gd_updated = None, self._obj_cost_per_edge(src, dest), False
        # assert np.isfinite(obj_1)
        for s in range(gd_steps):
            prob_gradient = self._eval_prob_gradients(src, dest)
            abs_gradient = prob_gradient.abs()
            L0 = abs_gradient.max_marginal(prob_gradient.scope_vars, inplace=False)
            L1 = abs_gradient.sum_marginal(prob_gradient.scope_vars, inplace=False)
            L2 = (abs_gradient*abs_gradient).sum_marginal(prob_gradient.scope_vars, inplace=False)
            L2 = np.sqrt(L2)

            if L0 < tolerance:
                return gd_updated
            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2 * (obj_0 - obj_1) / L1)
            # step = step if step > 0 else 1.0
            # step = 1.0
            obj_0 = obj_1
            ls_updated = self._line_search_cost(src, dest, obj_0, prob_gradient, step, L0, L2)
            obj_1 = self._obj_cost_per_edge(src, dest)
            if not gd_updated:
                gd_updated = ls_updated
            if not ls_updated:          # line search no more improvement
                return gd_updated       # if it improved at least once then return True
        return gd_updated

    def _line_search_cost(self, src:int, dest:int, obj_0:float, prob_gradient:Factor, step:float, L0:float, L2:float):
        ls_steps, armijo_thr = self.opt_options['ls_steps'], self.opt_options['armijo_thr']
        armijo_step_back, ls_tolerance = self.opt_options['armijo_step_back'], self.opt_options['ls_tolerance']
        for l in range(ls_steps):
            self._set_cost(src, dest, prob_gradient, step)
            obj_1 = self._obj_cost_per_edge(src, dest)
            if obj_0 - obj_1 > step * armijo_thr * L2:
                return True
            else:
                self._reset_cost(src, dest)     # set reset inside cluster method?
                step *= armijo_step_back
                if step * L0 < ls_tolerance:
                    return False
    # this is same except using different set and reset function and args pass kwargs
    def _set_cost(self, src, dest, gradient, step):
        cluster_src, cluster_dest = self.cid2cluster[src], self.cid2cluster[dest]
        prob_shift = gradient * step     # this works as object comes first fixme this is issue in factor.py

        cluster_src.factor_prev = cluster_src.factor
        cluster_src.pseudo_belief_prev = cluster_src.pseudo_belief
        cluster_src.local_bound_prev = cluster_src.local_bound
        cluster_src.factor = cluster_src.factor + prob_shift        # removed neg from prob shift opposite to gmid

        cluster_dest.factor_prev = cluster_dest.factor
        cluster_dest.pseudo_belief_prev = cluster_dest.pseudo_belief
        cluster_dest.local_bound_prev = cluster_dest.local_bound
        cluster_dest.factor = cluster_dest.factor - prob_shift
        # invalidate pseudo belief and marginals
        cluster_src.pseudo_belief = cluster_src.local_bound = None
        cluster_dest.pseudo_belief = cluster_dest.local_bound = None

    def _reset_cost(self, src, dest):
        cluster_src, cluster_dest = self.cid2cluster[src], self.cid2cluster[dest]
        cluster_src.factor = cluster_src.factor_prev
        cluster_src.pseudo_belief = cluster_src.pseudo_belief_prev
        cluster_src.local_bound = cluster_src.local_bound_prev
        cluster_dest.factor = cluster_dest.factor_prev
        cluster_dest.pseudo_belief = cluster_dest.pseudo_belief_prev
        cluster_dest.local_bound = cluster_dest.local_bound_prev

    def _eval_prob_gradients(self, src :int, dest:int):
        target = self.cid2cluster[src].scope_vars & self.cid2cluster[dest].scope_vars
        pm_src = self.cid2cluster[src].update_pseudo_belief()
        pm_src = pm_src.sum_marginal(pm_src.scope_vars - target, inplace=False)
        pm_dest = self.cid2cluster[dest].update_pseudo_belief()
        pm_dest = pm_dest.sum_marginal(pm_dest.scope_vars - target, inplace=False)
        gradient = pm_dest - pm_src
        return gradient

    def _obj_cost_per_edge(self, src:int, dest:int):
        # let match to gmid for now gmid adds up all local bounds as objective
        obj = 0.0
        for cid in self:
            obj += self.cid2cluster[cid].update_bound()
        return obj
        # obj = 0.0
        # obj += self.cid2cluster[src].update_bound()
        # obj += self.cid2cluster[dest].update_bound()
        # return obj

    ################################################################################################
    def update_weights_for_vid(self, vid:int):
        gd_steps, tolerance = self.opt_options['gd_steps'], self.opt_options['tolerance']
        obj_0, obj_1, gd_updated = None, self._obj_weight_per_vid(vid), False       # same as gmid?

        for s in range(gd_steps):
            gradient = self._eval_weight_gradients_per_var(vid)
            abs_grad = list(map(abs, gradient))
            L0, L1, L2 = max(abs_grad), sum(abs_grad), np.sqrt(sum(el*el for el in abs_grad))

            if L0 < tolerance:
                return gd_updated

            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2.0*(obj_0 - obj_1)/ L1)
            # step = step if step > 0 else 1.0
            # step = 1

            obj_0 = obj_1
            ls_updated = self._line_search_weight(vid, obj_0, gradient, step, L0, L2)
            obj_1 = self._obj_weight_per_vid(vid)
            if not gd_updated:
                gd_updated = ls_updated
            if not ls_updated:          # line search no more improvement
                return gd_updated       # if it imporved at least once then return True
        return gd_updated

    def _line_search_weight(self, vid:int, obj_0:float, gradient:List[float], step:float, L0:float, L2:float):
        ls_steps, armijo_thr = self.opt_options['ls_steps'], self.opt_options['armijo_thr']
        armijo_step_back, ls_tolerance = self.opt_options['armijo_step_back'], self.opt_options['ls_tolerance']

        for l in range(ls_steps):
            self._set_weights_per_vid(vid, gradient, step)  # set & reset
            obj_1 = self._obj_weight_per_vid(vid)
            if obj_0 - obj_1 > step* armijo_thr * L2:
                return True
            else:
                self._reset_weights_per_vid(vid)
                step *= armijo_step_back
                if step * L0 < ls_tolerance:
                    return False

    def _set_weights_per_vid(self, vid: int, gradient, step):  # hard to move inside cluster method
        # new_weights = np.zeros(len(self.vid2cid[vid]))
        new_weights = []
        for ind, cid in enumerate(self.vid2cid[vid]):  # this order is sorted so same
            cluster = self.cid2cluster[cid]
            current_weight = cluster.vid2weight[vid]
            cluster.vid2weight_prev[vid] = current_weight
            cluster.pseudo_belief_prev = cluster.pseudo_belief
            cluster.local_bound_prev = cluster.local_bound
            cluster.pseudo_belief = None
            cluster.local_bound = None

            new_weight = current_weight * np.exp(-step * gradient[ind])  # order match
            if new_weight > WINF:
                new_weight = WINF
            if new_weight < WEPS:
                new_weight = WEPS
            # assert np.isfinite(new_weight)
            new_weights.append(new_weight)
        new_weights_tot = sum(new_weights)      # this can be faster by numpy broadcastng

        for ind, cid in enumerate(self.vid2cid[vid]):
            cluster = self.cid2cluster[cid]
            cluster.vid2weight[vid] = new_weights[ind] / new_weights_tot

    def _reset_weights_per_vid(self, vid: int):
        for cid in self.vid2cid[vid]:  # can be moved inside cluster method
            cluster = self.cid2cluster[cid]
            cluster.vid2weight[vid] = cluster.vid2weight_prev[vid]
            cluster.pseudo_belief = cluster.pseudo_belief_prev
            cluster.local_bound = cluster.local_bound_prev

    def _eval_weight_gradients_per_var(self, vid:int)-> List[float]:
        wgts = []
        Hcond = []

        for cid in self.vid2cid[vid]:   # this is the order in gradient list
            cluster = self.cid2cluster[cid]
            mu_p = cluster.update_pseudo_belief()   # marignal to project down to scope? already?
            v_w_ind = cluster.vid_elim_order_local.index(vid)

            target = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind:])
            eliminator = cluster.scope_vars - target        # P(X_{i}, X_{i+1}, ... X_n)
            mu_p_temp1 = mu_p.sum_marginal(eliminator, inplace=False)
            if type(mu_p_temp1) is not Factor:
                mu_p_temp1 = Factor([], mu_p_temp1)
            if np.all(mu_p_temp1.table == 0):
                H1_p = 0.0
            else:
                H1_p = mu_p_temp1.entropy()

            if v_w_ind < len(cluster.vid_elim_order_local) -1:
                target = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind+1:])
                eliminator = cluster.scope_vars - target  # P(X_{i}, X_{i+1}, ... X_n)
                mu_p_temp2 = mu_p.sum_marginal(eliminator, inplace=False)
                if type(mu_p_temp2) is not Factor:
                    mu_p_temp2 = Factor([], mu_p_temp2)
                if np.all(mu_p_temp2.table == 0):
                    H2_p = 0.0
                else:
                    H2_p = mu_p_temp2.entropy()
            else:
                H2_p= 0.0

            Hcond.append(H1_p - H2_p)
            wgts.append(cluster.vid2weight[vid])

        Hbar = 0.0
        for i in range(len(wgts)):      # weighted average or inner product; np.array
            Hbar += wgts[i] * Hcond[i]
        gradient = [w_i * (Hcond[ind] - Hbar) for ind, w_i in enumerate(wgts)]
        # assert np.all(np.isfinite(gradient))
        return gradient

    def _obj_weight_per_vid(self, vid:int):
        obj = 0.0       # let match to gmid for now
        for cid in self:
            obj += self.cid2cluster[cid].update_bound()
        return obj
        # obj = 0.0           #  only evaluate touched clusters
        # for cid in self.vid2cid[vid]:
        #     obj += self.cid2cluster[cid].update_bound()
        # return obj

    ################################################################################################

    def _eval_weight_gradients_per_var_np(self, vid:int)->np.ndarray:
        Hcond = np.zeros(len(self.vid2cid[vid]))
        if vid not in self.wgt_grad_target1:
            self.wgt_grad_target1[vid] = SortedDict()
            self.wgt_grad_target2[vid] = SortedDict()

        for ind, cid in enumerate(self.vid2cid[vid]):   # this is the order in gradient list
            cluster = self.cid2cluster[cid]
            v_w_ind = cluster.vid_elim_order_local.index(vid)
            mu_p = cluster.update_pseudo_belief()

            if cid not in self.wgt_grad_target1[vid]:
                self.wgt_grad_target1[vid][cid] = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind:])
            target = self.wgt_grad_target1[vid][cid]
            # target = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind:])
            eliminator = cluster.scope_vars - target        # P(X_{i}, X_{i+1}, ... X_n)
            mu_p_temp1 = mu_p.sum_marginal(eliminator, inplace=False)
            if type(mu_p_temp1) is not Factor:
                mu_p_temp1 = Factor([], mu_p_temp1)
            Hcond[ind] = mu_p_temp1.entropy()

        for ind, cid in enumerate(self.vid2cid[vid]):
            cluster = self.cid2cluster[cid]
            v_w_ind = cluster.vid_elim_order_local.index(vid)
            if v_w_ind == len(cluster.vid_elim_order_local) - 1:    # last one
                continue
            mu_p = cluster.update_pseudo_belief()

            if cid not in self.wgt_grad_target2[vid]:
                self.wgt_grad_target2[vid][cid] = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind+1:])
            target = self.wgt_grad_target2[vid][cid]
            # target = SortedSet(self.gm.vid2var[el] for el in cluster.vid_elim_order_local[v_w_ind+1:])
            eliminator = cluster.scope_vars - target  # P(X_{i}, X_{i+1}, ... X_n)
            mu_p_temp2 = mu_p.sum_marginal(eliminator, inplace=False)
            if type(mu_p_temp2) is not Factor:
                mu_p_temp2 = Factor([], mu_p_temp2)
            Hcond[ind] -= mu_p_temp2.entropy()

        wgts = np.array( [self.cid2cluster[cid].vid2weight[vid] for cid in self.vid2cid[vid]] )
        Hbar = np.dot(wgts, Hcond)
        gradient = wgts * (Hcond-Hbar)
        return gradient

    def _set_weights_per_vid_np(self, vid: int, gradient: np.ndarray, step: float):
        for ind, cid in enumerate(self.vid2cid[vid]):  # this order is sorted so same
            cluster = self.cid2cluster[cid]
            cluster.vid2weight_prev[vid] = cluster.vid2weight[vid]
            cluster.pseudo_belief_prev = cluster.pseudo_belief
            cluster.local_bound_prev = cluster.local_bound
            cluster.pseudo_belief = None
            cluster.local_bound = None
        current_weights = np.array([self.cid2cluster[cid].vid2weight[vid] for cid in self.vid2cid[vid]])
        new_weights = current_weights * np.exp(-step * gradient)
        new_weights[ new_weights > WINF] = WINF
        new_weights[ new_weights < WEPS] = WEPS
        new_weights = new_weights/sum(new_weights)

        for ind, cid in enumerate(self.vid2cid[vid]):
            cluster = self.cid2cluster[cid]
            cluster.vid2weight[vid] = new_weights[ind]

    def update_weights_for_vid_np(self, vid:int):
        gd_steps, tolerance = self.opt_options['gd_steps'], self.opt_options['tolerance']
        obj_0, obj_1, gd_updated = None, self._obj_weight_per_vid(vid), False       # same as gmid?

        for s in range(gd_steps):
            gradient = self._eval_weight_gradients_per_var_np(vid)
            abs_grad = abs(gradient)
            L0, L1, L2 = max(abs_grad), sum(abs_grad), np.sqrt(np.dot(abs_grad, abs_grad))
            if L0 < tolerance:
                return gd_updated

            step = min(1.0, 1.0 / L1) if obj_0 is None else min(1.0, 2.0*(obj_0 - obj_1)/ L1)

            obj_0 = obj_1
            ls_updated = self._line_search_weight_np(vid, obj_0, gradient, step, L0, L2)
            obj_1 = self._obj_weight_per_vid(vid)
            if not gd_updated:
                gd_updated = ls_updated
            if not ls_updated:          # line search no more improvement
                return gd_updated       # if it imporved at least once then return True
        return gd_updated


    def _line_search_weight_np(self, vid:int, obj_0:float, gradient:np.ndarray, step:float, L0:float, L2:float):
        ls_steps, armijo_thr = self.opt_options['ls_steps'], self.opt_options['armijo_thr']
        armijo_step_back, ls_tolerance = self.opt_options['armijo_step_back'], self.opt_options['ls_tolerance']

        for l in range(ls_steps):
            self._set_weights_per_vid_np(vid, gradient, step)  # set & reset
            obj_1 = self._obj_weight_per_vid(vid)
            if obj_0 - obj_1 > step* armijo_thr * L2:
                return True
            else:
                self._reset_weights_per_vid(vid)
                step *= armijo_step_back
                if step * L0 < ls_tolerance:
                    return False
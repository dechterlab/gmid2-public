"""

Submodel class identifies a relevant subset of DecisionNetwork

Submodel Graph is a directed graph inheriting nx.DiGraph
"""
from typing import Iterable, List, Dict, Union, Optional

import networkx as nx
from sortedcontainers import SortedSet, SortedDict

from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.directed_network import DecisionNetwork, topo_sort, rev_topo_sort, barren_nodes
from gmid2.basics.directed_network import parents, ancestors, descendants
from gmid2.basics.directed_network import is_backdoor_set, dconnected
from gmid2.basics.helper import filter_subsets
from gmid2.basics.message import Message, TransitionConstraint, PolicyConstraint
from ..global_constants import *


class SubModelGraph(nx.DiGraph):
    def __init__(self):
        self.mid2message = SortedDict()     # mid to message object
        self.src2mids = SortedDict()        # message source sid
        self.sid2submodel = SortedDict()
        self.mid2edge = SortedDict()
        self.edge2mid = SortedDict()
        self.next_sid = 0
        super().__init__()

    def add_submodel_node(self, s) -> int:
        self.add_node(self.next_sid)
        self.sid2submodel[self.next_sid] = s
        s.sid = self.next_sid
        self.next_sid += 1
        return self.next_sid - 1

    def add_message(self, m: Message) -> None:
        """
        this method is called right after creating a message object
        that doesn't know the destination yet while doing decomposition
        """
        self.mid2message[m.mid] = m
        if m.src is not None:
            if not m.src in self.src2mids:
                self.src2mids[m.src] = SortedSet()
            self.src2mids[m.src].add(m.mid)

    def add_edges_from_value_messages(self)-> None:
        """
        this method is called by submodel_tree_decomposition

        determine destination of messages and connect nodes
        """
        for sid in self:
            submodel = self.sid2submodel[sid]
            mids = submodel.value_message_mids        # the message value node in each submodel tells the destination
            for mid in mids:
                msg = self.mid2message[mid]
                if isinstance(msg, TransitionConstraint):
                    src, dest = msg.src, sid     # source model id
                    msg.dest = dest
                    self.add_edge(src, dest, msg=msg)
                    self.edge2mid[(src,dest)] = mid
                    self.mid2edge[mid] = (src, dest)

    def add_edges_from_policy_messages(self)->None:
        """
        this method is called by submodel_graph_decomposition

        connect mini-submodels as a chain or a path

        form  a complete graph with edge cost as the number of overlapping variables
        find the largest weighted path in the graph
        ensure the separators overlap the most

        if we want a different message passing structure then find other shape, bipartite? maybe unstable?
        """
        g_score = dict()
        for src in self:
            g_score[src] = dict()
            for dest in self:
                if src != dest:
                    sep_size = len(self.sid2submodel[dest].policy_scope & self.sid2submodel[src].policy_scope)
                    g_score[src][dest] = sep_size

        nodes = sorted(self.nodes(), key=lambda x: len(self.sid2submodel[x].policy_scope))
        src = nodes.pop()       # start with the largest scope policy
        for _ in range(self.number_of_nodes()-1):
            dest = max(g_score[src], key=g_score[src].get)
            for mid in self.src2mids[src]:
                msg = self.mid2message[mid]
                if isinstance(msg, PolicyConstraint):
                    msg.dest = dest
                    self.add_edge(src, dest, msg=msg)
                    self.edge2mid[(src, dest)] = mid
                    self.mid2edge[mid] = (src, dest)
            src = dest


class SubModel:
    # sid = 0
    def __init__(self,
                 rel_d: Union[int, Iterable[int]]=None,
                 rel_u: Union[int, Iterable[int]]=None,
                 rel_o: Union[int, Iterable[int]]=None,
                 rel_h: Union[int, Iterable[int]]=None):
        self.sid = None

        if rel_d is not None:
            self.rel_d = SortedSet([rel_d]) if isinstance(rel_d, int) else SortedSet(rel_d)
        else:
            self.rel_d = SortedSet()

        if rel_u is not None:
            self.rel_u = SortedSet([rel_u]) if isinstance(rel_u, int) else SortedSet(rel_u)
        else:
            self.rel_u = SortedSet()

        if rel_o is not None:
            self.rel_o = SortedSet([rel_u]) if isinstance(rel_o, int) else SortedSet(rel_o)
        else:
            self.rel_o = SortedSet()

        if rel_h is not None:
            self.rel_h = SortedSet([rel_h]) if isinstance(rel_h, int) else SortedSet(rel_h)
        else:
            self.rel_h = SortedSet()

        # internal private members
        self.dn: DecisionNetwork = None        # fixme remove dn_ and only keep dn same for sg_
        self.sg: SubModelGraph = None
        self.rel_o_out = None
        self.rel_nodes_ = None
        self.rel_o_d_ = None
        self.rel_o_in = None

    def refine_from_rel_u(self, dn: DecisionNetwork)->None:
        """
        if rel_o and rel_h were missing, use this method to find them from provided dn
        then call relevant_decision_network over the same dn
        """
        self.rel_d &= ancestors(dn, self.rel_u)         # only ancestors of rel_u
        self.rel_o = relevant_observation(dn, self.rel_d, self.rel_u)
        self.rel_h = relevant_hidden(dn, self.rel_d, self.rel_u)

    def relevant_decision_network(self, dn: DecisionNetwork) -> DecisionNetwork:
        """
        this projected decision network can be used to solve a submodel by bucket-tree based algorithms
        """
        if self.dn is None:
            rel_nodes = self.relevant_nodes
            self.dn = dn.subgraph(nodes=rel_nodes).copy()   # not deepcopy, recreate all members below
            self.dn.net_type = dn.net_type

            for nid in self.relevant_nodes:
                vid = dn.nodes[nid]['vid']
                if vid is not None:
                    self.dn.vid2nid[vid] = nid
                fid = dn.nodes[nid]['fid']
                if fid is not None:
                    self.dn.fid2nid[fid] = nid
                mid = dn.nodes[nid]['mid']
                if mid is not None:
                    self.dn.mid2nid[mid] = nid

            self.dn.chance_nids.update(self.rel_o)     # decision is also marked as chance nodes here for submodel
            self.dn.chance_nids.update(self.rel_h)
            self.dn.decision_nids.update(self.rel_d)
            self.dn.value_nids.update(self.rel_u)
            self.dn.message_nids.update(dn.message_nids & self.relevant_nodes)

            for nid in self.rel_d:
                self.dn.immediate_observed_nodes[nid] = dn.immediate_observed_nodes[nid] & self.rel_o
                self.dn.informational_arcs.update((src, nid) for src in parents(self.dn, nid))
        return self.dn

    def rel_o_out_from_dn(self, dn:DecisionNetwork)->SortedSet:
        """
        dn is a decision network to identify interfacing nodes of submodel
        use this  after setting self.dn_ by relevant_decision_network
        by calling this, submodel knows what part of rel_o interfaces with other submodel

        non-interface nodes
        (X) (1) immediate observations; if not, that observation must be provided to previous stage
        (2) at least one of its parents is hidden node; marginalizing hidden node will pull p(obs|hidden)
        """
        # if self.rel_o_out_ is None:
        #     rel_o_immeidate = SortedSet()
        #     for d in self.rel_d:
        #         rel_o_immeidate.update(dn.immediate_observed_nodes[d] & self.rel_o)
        #     rel_o_out = SortedSet()
        #     for nid in self.rel_o:
        #         if nid not in rel_o_immeidate:
        #             rel_o_out.add(nid)
        #         elif not (parents(dn, nid) <= self.relevant_nodes):  # incoming other hidden? it should be included
        #             rel_o_out.add(nid)
        #     self.rel_o_out_ = rel_o_out
        # return self.rel_o_out_

        # (1) observed decision is interface
        # (2) parentless & non-immediate observed chance is interface
        # (3) if observed variable has a parent in rel_h it is internal as it will pulled by marginalization of rel_h

        self.rel_o_out = SortedSet()
        rel_o_immeidate = SortedSet()
        for d in self.rel_d:
            rel_o_immeidate.update(dn.immediate_observed_nodes[d] & self.rel_o)

        for nid in self.rel_o:
            if nid in dn.decision_nids:
                self.rel_o_out.add(nid)
            else:
                pa = parents(dn, nid)
                if not pa:
                    if nid not in rel_o_immeidate:
                        self.rel_o_out.add(nid)
                else:
                    if not (self.rel_h & pa):
                        self.rel_o_out.add(nid)

        self.rel_o_in  = self.rel_o - self.rel_o_out
        return self.rel_o_out

    @property
    def value_message_mids(self)->Optional[SortedSet]:
        if self.dn:
            ret = SortedSet()
            for nid in self.dn:
                if self.dn.nodes[nid]['mid'] is not None and self.dn.nodes[nid]['type'] == TYPE_VALUE_NODE:
                    ret.add(self.dn.nodes[nid]['mid'])
            return ret

    @property
    def policy_scope(self)->SortedSet:
        if self.rel_o_d_ is None:
            self.rel_o_d_ = self.rel_o | self.rel_d
        return self.rel_o_d_

    @property
    def relevant_nodes(self)->SortedSet:
        if self.rel_nodes_ is None:
            self.rel_nodes_ = self.rel_d | self.rel_u | self.rel_o | self.rel_h
        return self.rel_nodes_

    @property
    def score_by_node_count(self)->int:
        return len(self.rel_d) + len(self.rel_u) + len(self.rel_o) + len(self.rel_h)

    @property
    def score_by_induced_width(self):
        raise NotImplementedError

    @property
    def score_by_degree_sum(self):
        raise NotImplementedError

    # @classmethod
    # def reset_sid(cls):
    #     cls.sid = 0

    # @property
    # def graph_type(self):
    #     return type(self.G).__name__

    # @property
    # def G(self):
    #     if self.sg_ is None:
    #         return self.dn_
    #     return self.sg_

    # @G.setter
    # def G(self, g: Union[DecisionNetwork, SubModelGraph]):
    #     if isinstance(g, DecisionNetwork):
    #         self.dn_ = g
    #     else:
    #         self.sg_ = g

    # @property
    # def dn(self):
    #     return self.dn_

    @property
    def is_composite(self):
        return not self.is_atomic

    @property
    def is_atomic(self):
        return True if self.sg is None else False

    def __str__(self):
        return type(self).__name__ + "{}:([{}], [{}])".format(self.sid,
                                                             ",".join(str(el) for el in self.rel_d),
                                                             ",".join(str(el) for el in self.rel_u))

    # def __hash__(self):
    #     return hash(self.sid)               # sid used as a key but cannot use ints access this

    @property
    def internal_fids(self)->SortedSet:
        """
        return fids of functions inside the submodel; excludes policy functions, value by message, rel_o_out
        """
        prob_fids = self.internal_prob_fids
        util_fids = self.internal_util_fids
        return prob_fids | util_fids

    @property
    def internal_prob_fids(self)->SortedSet:
        ret = SortedSet()
        if self.dn:
            for nid in self.rel_h:
                if self.dn.nodes[nid]['fid'] is not None:
                    ret.add(self.dn.nodes[nid]['fid'])
            for nid in self.rel_o:
                if nid in self.rel_o_out:
                    continue
                if self.dn.nodes[nid]['fid'] is not None:
                    ret.add(self.dn.nodes[nid]['fid'])
        return ret

    @property
    def internal_util_fids(self)->SortedSet:
        ret = SortedSet()
        if self.dn:
            for nid in self.rel_u:
                if self.dn.nodes[nid]['fid'] is not None:
                    ret.add(self.dn.nodes[nid]['fid'])
        return ret

    def nid2vid(self, nid:int)->Optional[int]:
        if self.dn is None or nid not in self.dn: # or self.dn.nodes[nid]['vid'] is None:
            return  # None
        return self.dn.nodes[nid]['vid']

    def nids2vids(self, nids: Iterable[int])->List[int]:
        return [self.dn.nodes[nid]['vid'] for nid in nids if self.dn.nodes[nid]['vid'] is not None]

    def nids2fids(self, nids: Iterable[int])->List[int]:
        return [self.dn.nodes[nid]['fid'] for nid in nids if self.dn.nodes[nid]['fid'] is not None]

    def print_relevant_sets(self):
        print("START SUBMODEL RELEVANT SETS")
        print("sid\t\t{}".format(self.sid))
        print("rel_d\t\t{}".format(" ".join(str(el) for el in self.rel_d)))
        print("rel_u\t\t{}".format(" ".join(str(el) for el in self.rel_u)))
        print("rel_o\t\t{}".format(" ".join(str(el) for el in self.rel_o)))
        print("rel_h\t\t{}".format(" ".join(str(el) for el in self.rel_h)))
        print("rel_o_out\t\t{}".format(" ".join(str(el) for el in self.rel_o_out)))
        print("rel_o_in\t\t{}".format(" ".join(str(el) for el in self.rel_o_in)))
        print("END SUBMODEL RELEVANT SETS")


def relevant_utility(dn:DecisionNetwork, rel_d: Union[int, SortedSet]) -> SortedSet:
    if isinstance(rel_d, int):
        rel_d = SortedSet([rel_d])
    k = tuple(rel_d)
    if k not in dn.rel_u_cache:
        dn.rel_u_cache[k] = dn.value_nids & descendants(dn, rel_d)
    return dn.rel_u_cache[k]
    # return dn.value_nids & descendants(dn, rel_d)


def relevant_observation(dn:DecisionNetwork, rel_d: Union[int, SortedSet], rel_u: Union[int, SortedSet]) -> SortedSet:
    """
    approach 1: backdoor

    appoach 2:  requisite information (Lauritzen and Nilsson, 2001)
        a be a non-requisite inside observation Z
        a and rel_u d-separated given other observations

        a is a parent of decision that gives an incoming path (backdoor) to d.
        if that a is connected to rel_u it is a backdoor path
        in an influence diagram, parents to a decision are all observed nodes
    """
    if isinstance(rel_d, int):
        rel_d = SortedSet([rel_d])
    if isinstance(rel_u, int):
        rel_u = SortedSet([rel_u])
    if (tuple(rel_d), tuple(rel_u)) in dn.rel_o_cache:
        return dn.rel_o_cache[(tuple(rel_d), tuple(rel_u))]

    rel_o = dn.history_for_decisions(rel_d)
    if not is_backdoor_set(dn, rel_d, rel_u, rel_o):
        dn.rel_o_cache[(tuple(rel_d), tuple(rel_u))] = rel_o
        return rel_o

    st = 0
    while rel_o:
        l = len(rel_o)
        for i in range(st, l):
            trial = rel_o.pop(i)
            k = (tuple(rel_d), tuple(rel_u), tuple(rel_o))
            if k not in dn.is_backdoor:
                dn.is_backdoor[k] = is_backdoor_set(dn, rel_d, rel_u, rel_o)
            if dn.is_backdoor[k]:
                st = i
                break
            # if is_backdoor_set(dn, rel_d, rel_u, rel_o):
            #     st = i
            #     break
            else:
                rel_o.add(trial)
        if len(rel_o) == l: # no reduction
            dn.rel_o_cache[(tuple(rel_d), tuple(rel_u))] = rel_o
            return rel_o

    dn.rel_o_cache[(tuple(rel_d), tuple(rel_u))] = rel_o
    return rel_o    # when rel_o is empty still return it


def relevant_hidden(dn:DecisionNetwork, rel_d: Union[int, SortedSet], rel_u: Union[int, SortedSet]) -> SortedSet:
    """
    approach 1: frontdoor

    approach 2: (Nilsson)
        all hidden variables that are ancestors of rel_u
        given observations (we are not making irrelevant observations to hidden variabls)

        hidden variables X are relevant iff.
        (1) X is not barren
        (2) there exists a utility V s.t. X is connected to V given rel_D and pa(rel_D)
    """
    if isinstance(rel_d, int):
        rel_d = SortedSet([rel_d])
    if isinstance(rel_u, int):
        rel_u = SortedSet([rel_u])
    k_parmam = ( tuple(rel_d), tuple(rel_u) )
    if k_parmam in dn.rel_h_cache:
        return dn.rel_h_cache[k_parmam]

    obs = dn.history_for_decisions(rel_d)
    candidates = ancestors(dn, rel_u) - obs - rel_d
    rel_h = SortedSet()
    for x in candidates:
        k = (x, tuple(rel_u), tuple(obs))
        if k not in dn.is_dconnected:
            dn.is_dconnected[k]  = dconnected(dn, x, rel_u, obs)
        if dn.is_dconnected[k]:
            rel_h.add(x)
        # if dconnected(dn, x, rel_u, obs):
        #     rel_h.add(x)
    dn.rel_h_cache[k_parmam] = rel_h
    return rel_h


def submodel_tree_decomposition(dn: DecisionNetwork)-> SubModelGraph:
    if dn.net_type == "LIMID":
        return submodel_tree_decomposition_limid(dn)
    else:
        # return submodel_tree_decomposition_id(dn)
        return submodel_tree_decomposition_id_new(dn)


def submodel_tree_decomposition_id(dn: DecisionNetwork)->SubModelGraph:
    st = SubModelGraph()
    decisions_free = SortedSet(dn.decision_nids)                # init free decision nodes
    decision_blocks = list(filter_subsets(rev_topo_sort(dn), dn.decision_nids))
    for block in decision_blocks:
        for _ in block:

            # identify the best submodel at the current partial elimination order
            best_score, best_submodel = float('inf'), None
            for d in block:
                if d not in decisions_free:
                    continue
                decisions_unstable = SortedSet([d])     # find a submodel starting from d
                rel_d = SortedSet()
                while decisions_unstable:
                    rel_d.update(decisions_unstable)
                    rel_u = relevant_utility(dn, rel_d)  # don't update
                    rel_o = relevant_observation(dn, rel_d, rel_u)
                    rel_h = relevant_hidden(dn, rel_d, rel_u)
                    decisions_unstable = (decisions_free - rel_d) & rel_h     # find hidden decision variables
                s = SubModel(rel_d, rel_u, rel_o, rel_h)
                if s.score_by_node_count < best_score:
                    best_score = s.score_by_node_count
                    best_submodel = s
            if best_submodel is None:
                continue
            decisions_free.difference_update(best_submodel.rel_d)
            best_submodel.relevant_decision_network(dn)
            rel_o_out = best_submodel.rel_o_out_from_dn(dn)
            sid = st.add_submodel_node(best_submodel)          # add Submodel To SubmodelGraph
            if rel_o_out:
                m = TransitionConstraint(dn.nid2vid(rel_o_out), src=sid)              # don't know destination yet
                st.add_message(m)                                   # add this Message object to SubmodelGraph dict
                # add value message node to decision network (to submodel)
                new_nid = dn.add_node_by_nid(type=TYPE_VALUE_NODE, vid=None, fid=None, mid=m.mid)
                dn.add_edges_by_nid(new_nid, parent_nids=rel_o_out)

            # modify decision network
            for nid in best_submodel.rel_u:
                dn.remove_node_by_nid(nid)
            for nid in best_submodel.rel_d:
                dn.remove_node_by_nid(nid)
            # remove barren nodes after adding message value node
            b = barren_nodes(G=dn, probs = dn.hidden_nodes_for_decisions(dn.decision_nids), evids=dn.value_nids)
            for nid in b:
                dn.remove_node_by_nid(nid)
            dn.reset_dsep_cache()        # after modifying network, previous query is invalid

    for nid in dn.value_nids:  # dn change while loop don't remove nodes
        s = SubModel(None, nid, None, ancestors(dn, nid))
        s.relevant_decision_network(dn)
        rel_o_out = s.rel_o_out_from_dn(dn)
        assert len(rel_o_out) == 0
        st.add_submodel_node(s)
    st.add_edges_from_value_messages()

    for sid in sorted(st):
        submodel = st.sid2submodel[sid]
        submodel.print_relevant_sets()
    return st


def submodel_tree_decomposition_id_new(dn: DecisionNetwork)->SubModelGraph:
    st = SubModelGraph()
    decisions_free = SortedSet(dn.decision_nids)                # init free decision nodes
    decision_blocks = list(filter_subsets(rev_topo_sort(dn), dn.decision_nids))
    for block in decision_blocks:
        for d in block:
            if d not in decisions_free:
                continue
            decisions_unstable = SortedSet([d])     # find a submodel starting from d
            rel_d = SortedSet()
            rel_u = rel_o = rel_h = None
            while decisions_unstable:
                rel_d.update(decisions_unstable)
                rel_u = relevant_utility(dn, rel_d)  # don't update
                rel_o = relevant_observation(dn, rel_d, rel_u)
                rel_h = relevant_hidden(dn, rel_d, rel_u)
                decisions_unstable = (decisions_free - rel_d) & rel_h     # find hidden decision variables
            s = SubModel(rel_d, rel_u, rel_o, rel_h)
                # if s.score_by_node_count < best_score:
                #     best_score = s.score_by_node_count
                #     best_submodel = s
            # if best_submodel is None:
            #     continue
            decisions_free.difference_update(s.rel_d)
            s.relevant_decision_network(dn)
            rel_o_out = s.rel_o_out_from_dn(dn)
            sid = st.add_submodel_node(s)          # add Submodel To SubmodelGraph
            if rel_o_out:
                m = TransitionConstraint(dn.nid2vid(rel_o_out), src=sid)              # don't know destination yet
                st.add_message(m)                                   # add this Message object to SubmodelGraph dict
                # add value message node to decision network (to submodel)
                new_nid = dn.add_node_by_nid(type=TYPE_VALUE_NODE, vid=None, fid=None, mid=m.mid)
                dn.add_edges_by_nid(new_nid, parent_nids=rel_o_out)

            # modify decision network
            for nid in s.rel_u:
                dn.remove_node_by_nid(nid)
            for nid in s.rel_d:
                dn.remove_node_by_nid(nid)
            # remove barren nodes after adding message value node
            b = barren_nodes(G=dn, probs = dn.hidden_nodes_for_decisions(dn.decision_nids), evids=dn.value_nids)
            for nid in b:
                dn.remove_node_by_nid(nid)
            dn.reset_dsep_cache()        # after modifying network, previous query is invalid

    for nid in dn.value_nids:  # dn change while loop don't remove nodes
        s = SubModel(None, nid, None, ancestors(dn, nid))
        s.relevant_decision_network(dn)
        rel_o_out = s.rel_o_out_from_dn(dn)
        assert len(rel_o_out) == 0
        st.add_submodel_node(s)
    st.add_edges_from_value_messages()

    for sid in sorted(st):
        submodel = st.sid2submodel[sid]
        submodel.print_relevant_sets()
    return st


def partition_value_functions(dn: DecisionNetwork, value_nids: Iterable[int], i_bound:int, m_bound: int)-> List[SortedSet]:
    """
    given a colleciton of value nodes, partition them by m_bound
    collect similar value functions
    """
    def scoring_by_scope()-> Dict[int, Dict[int, int]]:
        """
        nid1:   {nid2: distance}
        """
        nid2scopes = SortedDict()
        for nid in value_nids:
            assert nid in dn.value_nids, "node must be value node"
            nid2scopes[nid] = SortedSet( dn.nid2vid(parents(dn, nid)) )     # parents for value nodes

        nid2distances = dict()
        for nid in value_nids:
            nid2distances[nid] = dict()
            for nid2 in value_nids:
                if nid != nid2:
                    # dist = len(nid2scopes[nid] ^ nid2scopes[nid2])
                    dist = len(nid2scopes[nid2] - nid2scopes[nid])      # increase of scope from nid2
                    nid2distances[nid][nid2] = dist
        return nid2distances

    nid2scope_dist = scoring_by_scope()
    score_graph = dict()
    # score_graph = SortedDict( {nid: SortedSet(key=lambda x: nid2scope_dist[nid][x]) for nid in value_nids})
    for nid in value_nids:
        f = lambda x: nid2scope_dist[nid][x]
        score_graph[nid] = SortedSet(key=f)
        for nid2 in value_nids:
            if nid != nid2:
                score_graph[nid].add(nid2)      # graph edge is sorted by the distance, closer first

    partitions = []
    unprocessed = SortedSet(value_nids, key=lambda x: len(parents(dn, nid)))    # smaller function first
    partition = SortedSet()
    partition_vars = set()

    while unprocessed:
        seed = unprocessed.pop()
        partition.add(seed)
        partition_vars.update(parents(dn, seed))
        for nid in score_graph:
            score_graph[nid].discard(seed)       # cannot use seed node because it is already selected

        while len(partition) < m_bound and len(partition_vars) < i_bound and score_graph[seed]:
            nid2 = score_graph[seed].pop()      # sortedset, pop closest
            partition.add(nid2)
            partition_vars.update(parents(dn, nid2))
            unprocessed.discard(nid2)
            for nid in score_graph:
                score_graph[nid].discard(nid2)       # cannot use nid2 node because it is already selected

        partitions.append(partition)
        partition = SortedSet()
        partition_vars = set()

    return partitions


def submodel_graph_decomposition(st: SubModelGraph, i_bound:int, m_bound:int):
    for submodel_block in topo_sort(st):
        for sid in submodel_block:
            # submodel to graph of mini submodels
            submodel = st.sid2submodel[sid]
            partitions = partition_value_functions(submodel.dn, submodel.rel_u, i_bound, m_bound)
            sg = SubModelGraph()
            for partition in partitions:
                mini_submodel = SubModel(SortedSet(submodel.rel_d), SortedSet(partition))
                mini_submodel.refine_from_rel_u(dn=submodel.dn)
                mini_submodel.relevant_decision_network(dn=submodel.dn)
                sg.add_submodel_node(mini_submodel)
                m = PolicyConstraint(src=mini_submodel.sid)
                sg.add_message(m)

                # add value message nodes to decision networks? No not yet
                # in case of submodel tree decomposition,
                # it is required to add value node as it is the process of submodel elimination
                # the value node from this policy constraint will be used in bucket-tree based method
                # so it is better to add it while forming a message graph
                # the join graph decomposition also needs to be re-considered to combine
                # log-MGF, maximum principle, submodel graph
                # how about information relaxation? more observation + penalty on deviation of individual trajectory?
                # it stands at the first; IR->MP->SG all work with log MGF and internal relaxation

            sg.add_edges_from_policy_messages()
            # submodel.G = sg       # testing should be OK with removing this
    return st


def submodel_tree_decomposition_limid(dn: DecisionNetwork)->SubModelGraph:
    st = SubModelGraph()
    decisions_free = SortedSet(dn.decision_nids)                # init free decision nodes
    decision_blocks = list(filter_subsets(rev_topo_sort(dn), dn.decision_nids))
    for block in decision_blocks:
        for _ in block:       # todo fix this in LIMID save time

            # identify the best submodel at the current partial elimination order
            best_score, best_submodel = float('inf'), None
            for d in block:
                if d not in decisions_free:
                    continue
                decisions_unstable = SortedSet([d])     # find a submodel starting from d
                rel_d = SortedSet()

                temp_info_arcs = dict()          # added arcs
                while decisions_unstable:
                    rel_d.update(decisions_unstable)
                    if len(rel_d) > 1:
                        for d1 in rel_d:
                            for d2 in rel_d:
                                if d1 == d2:
                                    continue
                                if d1 not in temp_info_arcs:
                                    temp_info_arcs[d1] = SortedSet()
                                temp_info_arcs[d1].update( (el, d1) for el in dn.immediate_observed_nodes[d2] if el not in dn.decision_nids)
                                dn.add_edges_from(temp_info_arcs[d1])
                    rel_u = relevant_utility(dn, rel_d)  # don't update
                    rel_o = relevant_observation(dn, rel_d, rel_u)
                    rel_h = relevant_hidden(dn, rel_d, rel_u)
                    decisions_unstable = (decisions_free - rel_d) & rel_h     # find hidden decision variables

                s = SubModel(rel_d, rel_u, rel_o, rel_h)
                if s.score_by_node_count < best_score:
                    best_score = s.score_by_node_count
                    best_submodel = s
                if len(rel_d) > 1:
                    for d1 in rel_d:
                        dn.remove_edges_from(temp_info_arcs[d1])
            if best_submodel is None:
                continue

            decisions_free.difference_update(best_submodel.rel_d)

            if len(best_submodel.rel_d) > 1: # todo fix this in LIMID
                # best_submodel.rel_o -= best_submodel.rel_d
                # best_submodel.rel_h -= best_submodel.rel_o
                # best_submodel.rel_h -= best_submodel.rel_d
                added_edges = SortedSet()
                for d1 in best_submodel.rel_d:
                    for d2 in best_submodel.rel_d:
                        if d1 == d2:
                            continue
                        dn.immediate_observed_nodes[d1].update(dn.immediate_observed_nodes[d2] - dn.decision_nids)
                        added_edges.update( (el, d1) for el in dn.immediate_observed_nodes[d2] if el not in dn.decision_nids)
                dn.add_edges_from(added_edges)
                dn.informational_arcs.update(added_edges)

            best_submodel.relevant_decision_network(dn)
            rel_o_out = best_submodel.rel_o_out_from_dn(dn)
            sid = st.add_submodel_node(best_submodel)          # add Submodel To SubmodelGraph
            if rel_o_out:
                m = TransitionConstraint(dn.nid2vid(rel_o_out), src=sid)              # don't know destination yet
                st.add_message(m)                                   # add this Message object to SubmodelGraph dict

                new_nid = dn.add_node_by_nid(type=TYPE_VALUE_NODE, vid=None, fid=None, mid=m.mid)
                dn.add_edges_by_nid(new_nid, parent_nids=rel_o_out)

            # modify decision network
            for nid in best_submodel.rel_u:
                dn.remove_node_by_nid(nid)
            for nid in best_submodel.rel_d:
                dn.remove_node_by_nid(nid)
            b = barren_nodes(G=dn, probs = dn.hidden_nodes_for_decisions(dn.decision_nids), evids=dn.value_nids)    # remove after adding message value node
            for nid in b:
                dn.remove_node_by_nid(nid)
            dn.reset_dsep_cache()        # after modifying network, previous query is invalid

    for nid in dn.value_nids:  # dn change while loop don't remove nodes
        s = SubModel(None, nid, None, ancestors(dn, nid))
        s.relevant_decision_network(dn)
        rel_o_out = s.rel_o_out_from_dn(dn)
        assert len(rel_o_out) == 0
        st.add_submodel_node(s)

    st.add_edges_from_value_messages()

    for sid in sorted(st):
        submodel = st.sid2submodel[sid]
        submodel.print_relevant_sets()

    return st


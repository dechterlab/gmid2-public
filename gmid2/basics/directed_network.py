# from __future__ import annotations
from typing import List, Union, Iterable, Optional, Iterator, Text, Any
from copy import deepcopy
from sortedcontainers import SortedDict, SortedSet
import networkx as nx

from .uai_files import FileInfo
from ..global_constants import *
from .helper import filter_subsets


class BayesianNetwork(nx.DiGraph):
    pass


class DecisionNetwork(nx.DiGraph):
    """
    DecisionNetwork as DiGraph
    Use networkx functions and methods to access graph interface

    self.copy() provides shallow copy
    deepcopy( self ) provides deep copy by using copy.deepcopy
    subgraph_view(G, filter_node=no_filter, filter_edge=no_filter)
    subgraph(G, nbunch)
    subgraph(G, nbunch).copy()

    """
    def __init__(self):
        self.vid2nid = SortedDict()
        self.fid2nid = SortedDict()
        self.mid2nid = SortedDict()

        self.chance_nids = SortedSet()
        self.decision_nids = SortedSet()
        self.value_nids = SortedSet()
        self.message_nids = SortedSet()
        self.immediate_observed_nodes = SortedDict()
        self.informational_arcs = SortedSet()

        self.net_type = ""
        self.current_nid = 0

        self.is_backdoor = dict()               # [ tuple( rel_d, rel_u, rel_o ) ] = True False in relevant_observation
        self.is_dconnected = dict()             # [ tuple( x, rel_u, obs) ] = True False in relevant_hidden
        self.rel_u_cache = dict()
        self.rel_o_cache = dict()
        self.rel_h_cache = dict()

        super().__init__()

    def build(self, file_info: FileInfo) -> None:
        """
        create a decision network at once from file info object
        """
        self.name = file_info.uai_file
        self.net_type = file_info.net_type

        for i in range(file_info.nchance):
            vid = file_info.chance_vars[i]
            self.add_node(node_for_adding=self.current_nid, vid=vid, fid=None, mid=None, type= TYPE_CHANCE_NODE,
                          conditioned=False)
            self.vid2nid[vid] = self.current_nid
            self.chance_nids.add(self.current_nid)
            self.current_nid += 1

        for i in range(file_info.ndec):
            vid=file_info.decision_vars[i]
            self.add_node(node_for_adding=self.current_nid, vid=vid, fid=None, mid=None, type= TYPE_DECISION_NODE,
                          conditioned=False)
            self.vid2nid[vid] = self.current_nid
            self.decision_nids.add(self.current_nid)
            self.current_nid += 1

        for i in range(file_info.nutil):
            fid = file_info.util_funcs[i]
            self.add_node(node_for_adding=self.current_nid, vid=None, fid=fid, mid=None, type=TYPE_VALUE_NODE,
                          conditioned=False)
            scopes = sorted(self.vid2nid[vid] for vid in file_info.scopes[fid])     # vids
            self.add_edges_from((src, self.current_nid) for src in scopes)
            self.fid2nid[fid] = self.current_nid
            self.value_nids.add(self.current_nid)
            self.current_nid += 1

        for i in range(file_info.nprob):
            fid = file_info.prob_funcs[i]
            *scope, dest= file_info.scopes[fid]
            dest = self.vid2nid[dest]
            self.add_edges_from( (src, dest) for src in sorted(self.vid2nid[vid] for vid in scope))
            self.nodes[dest]['fid'] = fid

        # add informational arcs
        for b in range(0, file_info.nblock):        # last elim block could be decision
            block_t = file_info.block_types[b]
            if block_t == TYPE_DEC_BLOCK:
                block_dest = [self.vid2nid[n] for n in file_info.blocks[b]]     # decision node to receive arcs
                block_src = SortedSet()
                if b < file_info.nblock-1:
                    block_src.update(self.vid2nid[n] for n in file_info.blocks[b + 1])
                    for dec in block_dest:
                        self.immediate_observed_nodes[dec] = SortedSet(block_src)       # bug fix; first decision block fails to add empty set
                    edges = [(src, dest) for src in block_src for dest in block_dest]
                    self.add_edges_from(edges)
                    self.informational_arcs.update(edges)
                else:   # the last block no more previous
                    for dec in block_dest:
                        self.immediate_observed_nodes[dec] = SortedSet()    # put empty set

                if self.net_type == TYPE_ID_NETWORK:
                    block_src = SortedSet()
                    for b2 in range(b+2, file_info.nblock):
                        block_src.update(self.vid2nid[n] for n in file_info.blocks[b2])
                    edges = [(src, dest) for src in block_src for dest in block_dest]
                    self.add_edges_from(edges)
                    self.informational_arcs.update(edges)
        # for b in range(1, file_info.nblock):
        #     block_t = file_info.block_types[b]
        #     if block_t == TYPE_DEC_BLOCK:
        #         block_dest = [self.vid2nid[n] for n in file_info.blocks[b]]
        #         block_src = SortedSet()
        #         if b + 1 < file_info.nblock:
        #             block_src.update(self.vid2nid[n] for n in file_info.blocks[b + 1])
        #             for dec in block_dest:
        #                 self.immediate_observed_nodes[dec] = SortedSet(block_src)       # bug fix; first decision block fails to add empty set
        #             edges = [(src, dest) for src in block_src for dest in block_dest]
        #             self.add_edges_from(edges)
        #             self.informational_arcs.update(edges)
        #
        #         if self.net_type == TYPE_ID_NETWORK:
        #             block_src = SortedSet()
        #             for b2 in range(b+2, file_info.nblock):
        #                 block_src.update(self.vid2nid[n] for n in file_info.blocks[b2])
        #             edges = [(src, dest) for src in block_src for dest in block_dest]
        #             self.add_edges_from(edges)
        #             self.informational_arcs.update(edges)

    def add_edges_by_nid(self, nid, parent_nids: Iterable[int], child_nids: Iterable[int]=None) -> None:
        self.add_edges_from( (pa, nid) for pa in parent_nids)
        if child_nids:
            self.add_edges_from( (nid, ch) for ch in child_nids)

    def add_node_by_nid(self, type:Text, vid:Optional[int]=None, fid:Optional[int]=None, mid:Optional[int]=None)->int:
        self.add_node(node_for_adding=self.current_nid, vid=vid, fid=fid, mid=mid, type=type,
                      conditioned=False)
        if vid:
            self.vid2nid[vid] = self.current_nid
        if fid:
            self.fid2nid[fid] = self.current_nid
        if mid:
            self.mid2nid[mid] = self.current_nid

        if type == TYPE_CHANCE_NODE:
            self.chance_nids.add(self.current_nid)
        if type == TYPE_DECISION_NODE:
            self.decision_nids.add(self.current_nid)
        if type == TYPE_VALUE_NODE:
            self.value_nids.add(self.current_nid)
        if mid is not None:
            self.message_nids.add(self.current_nid)     #   working nodes in dn associated with messages [added to both value_nids and message_nids]
        # if type == TYPE_MESSAGE_NODE:     # there's no MESSAGE_NODE, message can be looked up by mid2nid, message map
        #     self.message_nids.add(self.current_nid)
        self.current_nid += 1
        return self.current_nid-1

    def clean_nid(self, nid:int) -> None:
        if self.nodes[nid]['type'] == TYPE_CHANCE_NODE:
            self.chance_nids.remove(nid)
        elif self.nodes[nid]['type'] == TYPE_DECISION_NODE:
            self.decision_nids.remove(nid)
        elif self.nodes[nid]['type'] == TYPE_VALUE_NODE:
            self.value_nids.remove(nid)
        elif self.nodes[nid]['type'] == TYPE_MESSAGE_NODE:      # working submodel decomposition add VALUE NODE so this is void
            self.message_nids.remove(nid)

    # def remove_edge_by_nid(self, src_nid:int, dest_nid:int):
    #     self.remove_edge(src_nid, dest_nid)

    def remove_node_by_nid(self, nid:int) -> None:
        self.clean_nid(nid)
        self.remove_node(nid)

    def remove_node_by_vid(self, vid: int) -> None:
        self.remove_node(self.vid2nid[vid])

    def remove_node_by_fid(self, fid: int) -> None:
        self.remove_node(self.fid2nid[fid])

    def remove_node_by_mid(self, mid: int) -> None:
        self.remove_node(self.mid2nid[mid])

    def condition_nodes(self, nids: Union[Iterable[int], int]) -> None:
        if isinstance(nids, int):
            self.nodes[nids]['conditioned'] = True
        else:
            for n in nids:
                self.nodes[n]['conditioned'] = True

    def uncondition_nodes(self, nids: Union[Iterable[int], int]) -> None:
        if isinstance(nids, int):
            self.nodes[nids]['conditioned'] = False
        else:
            for n in nids:
                self.nodes[n]['conditioned'] = False

    def hidden_nodes_for_decisions(self, decision_nids: Optional[Union[int, Iterable[int]]]=None) -> SortedSet:
        return self.chance_nids - parents(self, decision_nids)

    def history_for_decisions(self, decision_nids: Union[int, Iterable[int]]) -> SortedSet:
        if self.net_type == TYPE_ID_NETWORK:
            ancestor_decisions = ancestors(self, decision_nids) & self.decision_nids
            h = parents(self, decision_nids)
            h.update(ancestor_decisions)
            for nid in ancestor_decisions:
                h.update( parents(self, nid) )
            return h
        else:
            return parents(self, decision_nids)

    def nid2vid(self, nids: Union[int, Iterable[int]]) -> Union[int, List[int]]:
        if isinstance(nids, int):
            return self.nodes[nids]['vid']
        return [self.nodes[n]['vid'] for n in nids]

    def nid2fid(self, nids: Union[int, Iterable[int]]) -> Union[int, List[int]]:
        if isinstance(nids, int):
            return self.nodes[nids]['fid']
        return [self.nodes[n]['fid'] for n in nids]

    def nid2mid(self, nids: Union[int, Iterable[int]]) -> Union[int, List[int]]:
        if isinstance(nids, int):
            return self.nodes[nids]['mid']
        return [self.nodes[n]['mid'] for n in nids]

    def reset_dsep_cache(self):
        self.is_backdoor = dict()               # [ tuple( rel_d, rel_u, rel_o ) ] = True False in relevant_observation
        self.is_dconnected = dict()             # [ tuple( x, rel_u, obs) ] = True False in relevant_hidden
        self.rel_u_cache = dict()
        self.rel_o_cache = dict()
        self.rel_h_cache = dict()

def parents(G: nx.DiGraph, nid: Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nid, int):
        return SortedSet(G.predecessors(nid))
    r = SortedSet()
    for n in nid:
        r.update(G.predecessors(n))
    return r


def children(G: nx.DiGraph, nid: Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nid, int):
        return SortedSet(G.successors(nid))
    r = SortedSet()
    for n in nid:
        r.update(G.successors(n))
    return r


def descendants(G: nx.DiGraph, nids:Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nids, int):
        return SortedSet(nx.descendants(G, nids))
    r = SortedSet()
    for n in nids:
        r.update(nx.descendants(G, n))
    return r


def ancestors(G: nx.DiGraph, nids:Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nids, int):
        return SortedSet(nx.ancestors(G, nids))
    r = SortedSet()
    for n in nids:
        r.update(nx.ancestors(G, n))
    return r


def markov_blanket_directed(G: nx.DiGraph, nids: Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nids, int):
        nids = SortedSet([nids])
    r = SortedSet()
    # directed graph mb == pa, ch, pa of ch
    r.update(parents(G, nids))
    r.update(children(G, nids))
    r.update(parents(G, children(G, nids)))
    r.difference_update(nids)
    return r


def rev_topo_sort(G):
    node2ch = {n: children(G, n) for n in G}
    while True:
        leafs = SortedSet(n for n, ch in node2ch.items() if not ch)
        if not leafs:
            break
        yield leafs

        for n in leafs:
            for m in parents(G, n):
                node2ch[m] -= leafs
        for n in leafs:
            del node2ch[n]


def topo_sort(G):
    node2pa = {n: parents(G, n) for n in G}
    while True:
        roots = SortedSet(n for n, ch in node2pa.items() if not ch)
        if not roots:
            break
        yield roots

        for n in roots:
            for m in children(G, n):
                node2pa[m] -= roots
        for n in roots:
            del node2pa[n]

# move to helper
# def reduce_nids(nids: Union[Iterable[SortedSet[int]]], nid_subset: SortedSet[int]) -> Iterator[SortedSet[int]]:
#     for t in nids:
#         ss = nid_subset & t
#         if ss:
#             yield ss


def dconnected(G, X: Union[int, SortedSet], Y: Union[int, SortedSet], Z: Union[int, SortedSet]) -> bool:
    """
    test if X is d-connected to Y given Z
    (1) for each source x find reachable set given Z
    (2) if any element of Y is reachable from any element of X then it is d-connected
    """
    if isinstance(X, int):
        X = SortedSet([X])
    if isinstance(Y, int):
        Y = SortedSet([Y])
    for x in X:
        R = reachble(G, x, Z)
        if Y & R:
            return True
    return False


def dseparated(G, X: Union[int, SortedSet], Y: Union[int, SortedSet], Z: Union[int, SortedSet]) -> bool:
    return not dconnected(G, X, Y, Z)


def reachble(G, x: int, Z: Union[int, SortedSet]) -> SortedSet:
    """
    Algorithm 3.1 Page 75. (PGM text book by Koller and Friedman)
    """
    from collections import deque, namedtuple
    Node = namedtuple("Node", ['nid', 'dir'])

    if isinstance(Z, int):
        Z = SortedSet([Z])
    A = ancestors(G, Z) | Z
    L = deque([Node(x, 'UP')])
    V = set()
    R = SortedSet()

    while L:
        current_node = L.popleft()
        if current_node in V:
            continue

        if current_node.nid not in Z:
            R.add(current_node.nid)
        V.add(current_node)

        if current_node.dir == "UP":
            if current_node.nid not in Z:
                L.extend(Node(pa, "UP") for pa in parents(G, current_node.nid)-V)
                L.extend(Node(ch, "DOWN") for ch in children(G, current_node.nid)-V)
        elif current_node.dir == "DOWN":
            if current_node.nid not in Z:
                L.extend(Node(ch, "DOWN") for ch in children(G, current_node.nid)-V)
            if current_node.nid in A:
                L.extend(Node(pa, "UP") for pa in parents(G, current_node.nid)-V)
    return R


def barren_nodes(G, probs: Union[int, SortedSet], evids: Union[int, Iterable[int]]) -> SortedSet:
    # barren nodes are prob nodes that are non-ancestors of evidence nodes in DAG
    if isinstance(probs, int):
        probs = SortedSet([probs])
    return probs - ancestors(G, evids)


def is_backdoor(G,x: int, y: int, Z: Union[int, SortedSet]) -> bool:
    """
    todo tests for backdoor
    test if Z satisfies the back-door criterion (Pearl, "Causality")
    (1) no node in Z is a descendant of x
    (2) Z blocks every path between x and y that contains an arrow into x

    remove outgoing arcs from x in G so that excluding the path contains an arrow outgoing from x
    test if x and y are d-separated given Z, I(x, y | Z)
    """
    x_outgoing = children(G, x)
    for dest in x_outgoing:
        G.remove_edge(x, dest)
    # assert Z <= ancestors(G, x)
    # if not (Z <= ancestors(G, x)):
    #     Z &= ancestors(G, x)
    separated = dseparated(G, x, y, Z)
    for dest in x_outgoing:
        G.add_edge(x, dest)
    return separated


def is_backdoor_set(G: nx.DiGraph, X: Union[int, SortedSet], Y: Union[int, SortedSet], Z: Union[int, SortedSet]) -> bool:
    """
    remove all outgoing arcs from X
    test d-separation (x, y |Z ) in mutilated graph
    """
    if isinstance(X, int):
        X = SortedSet([X])
    if isinstance(Y, int):
        Y = SortedSet([Y])
    # remove all outgoing arcs from X and test for individual is_backdoor(x in X, y in Y, Z)
    removed = set()
    for x in X:
        for dest in children(G, x):
            G.remove_edge(x, dest)
            removed.add( (x,dest) )
    # assert Z <= ancestors(G, X)
    # Z = Z & ancestors(G, X)
    res = all(dseparated(G, x, y, Z) for x in X for y in Y)
    G.add_edges_from(removed)
    return res


def is_frontdoor(G: nx.DiGraph, x: int, y: int, Z: Union[int, SortedSet]) -> bool:
    """
    todo tests for frontdoor
    test if Z satisfies the front-door criterion (Pearl, "Causality")
    (1) Z intercepts all directed paths from X to Y
    (2) there is no unblocked back-door path from X to Z
    (3) all back-door paths from Z to Y are blocked by X

    condition 1. remove incoming arcs to X and outgoing arcs from Y and test for d-separation I(X, Y | Z)
    condition 2. no backdoor between X to Z, remove outgoing arcs from X and test for d-separation I(X, Z)
    condition 3. no backdoor between Z to Y | X, remove outgoing arcs from Z and test for d-sep I(Z, Y | X)
    """
    removed_edges = []
    x_incoming = parents(G, x)
    y_outgoing = children(G, y)
    for src in x_incoming:
        removed_edges.append( (src, x) )
        G.remove_edge(src, x)
    for dest in y_outgoing:
        removed_edges.append( (y, dest) )
        G.remove_edge(y, dest)
    condition1 = dseparated(G, x, y, Z)
    G.add_edges_from(removed_edges)
    if not condition1:
        return False

    removed_edges = []
    x_outgoing = children(G, x)
    for dest in x_outgoing:
        removed_edges.append( (x, dest) )
        G.remove_edge(x, dest)
    condition2 = dseparated(G, x, Z, SortedSet())
    G.add_edges_from(removed_edges)
    if not condition2:
        return False

    removed_edges = []
    for z in Z:
        z_outgoing =children(G, z)
        for dest in z_outgoing:
            removed_edges.append( (z, dest) )
            G.remove_edge(z, dest)
    condition3 = dseparated(G, Z, y, x)
    G.add_edges_from(removed_edges)
    return condition3


def is_frontdoor_set(G: nx.DiGraph, X: Union[int, SortedSet], Y: Union[int, SortedSet], Z: Union[int, SortedSet]) -> bool:
    if isinstance(X, int):
        X = SortedSet([X])
    if isinstance(Y, int):
        Y = SortedSet([Y])
    return all(is_frontdoor(G, x, y, Z) for x in X for y in Y)


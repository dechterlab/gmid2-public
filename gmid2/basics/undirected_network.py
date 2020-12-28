# from __future__ import annotations
import itertools, copy
from typing import List, Union, Iterable, Optional, Iterator, Text, Any
from sortedcontainers import SortedDict, SortedSet, SortedList
import networkx as nx
import numpy as np

from .uai_files import FileInfo
from ..global_constants import *
from .directed_network import DecisionNetwork, parents, BayesianNetwork


class PrimalGraph(nx.Graph):
    """
    PrimalGraph as Graph
    Use networkx undirected graph class

    for a decision network, primal graph can be created by only looking at the scope (identical to BN, MN)
    However, PrimalGraph may not know the type of the node; knowing vid to access its type from dn or gm
        gm.vid2type or dn.nodes[dn.vid2nid[vid]]['type']
    """
    def __init__(self):
        self.vid2nid = SortedDict()
        self.nid2vid = SortedDict()
        self.chance_nids = SortedSet()
        self.decision_nids = SortedSet()
        self.current_nid = 0
        super().__init__()


    def build_from_dn(self, dn: DecisionNetwork) -> None:
        """
        nodes added in the order returned by dn.nodes
        (1) moralize the graph
        (2) remove value nodes because they are not variables
        """
        for dn_nid in dn.nodes:
            if dn.nodes[dn_nid]['type'] == TYPE_VALUE_NODE:
                continue
            vid = dn.nodes[dn_nid]['vid']
            self.vid2nid[vid] = self.current_nid
            self.nid2vid[self.current_nid] = vid
            if dn.nodes[dn_nid]['type'] == TYPE_DECISION_NODE:
                self.add_node(node_for_adding=self.current_nid, type=TYPE_DECISION_VAR)
                self.decision_nids.add(self.current_nid)
            elif dn.nodes[dn_nid]['type'] == TYPE_CHANCE_NODE:
                self.add_node(node_for_adding=self.current_nid, type=TYPE_CHANCE_VAR)
                self.chance_nids.add(self.current_nid)
            self.current_nid += 1

        for dn_nid in dn.nodes:
            if dn.nodes[dn_nid]['type'] == TYPE_DECISION_NODE:      # no parents to decision, no scope for policy in uai
                continue
            pa_vids = [dn.nodes[el]['vid'] for el in parents(dn, dn_nid)]
            if dn.nodes[dn_nid]['type'] == TYPE_CHANCE_NODE:
                vid = dn.nodes[dn_nid]['vid']       # connect to parent, arcs between all parents
                self.add_edges_from( (self.vid2nid[vid], self.vid2nid[pa_vid]) for pa_vid in pa_vids )
            self.add_edges_from( itertools.combinations( (self.vid2nid[pa_vid] for pa_vid in pa_vids), 2)  )

    def build_from_bn(self, bn: BayesianNetwork)-> None:
        raise NotImplementedError

    # def build_from_file(self, file_info: FileInfo) -> None:
    #     """
    #     node id is identical to vid
    #     """
    #     self.build_from_scopes(file_info.scopes, file_info.var_types)

    def build_from_scopes(self, scope_vids: Iterable[Iterable[int]], var_types: Union[List[Text], SortedDict]=None):
        """
        nodes added in the order of appear in the scope
        """
        for sc in scope_vids:
            for vid in sc:
                if vid not in self.vid2nid:
                    var_type = TYPE_CHANCE_VAR
                    if var_types and var_types[vid] == TYPE_DECISION_VAR:
                        var_type = TYPE_DECISION_VAR
                        self.decision_nids.add(self.current_nid)
                    else:
                        self.chance_nids.add(self.current_nid)
                    self.add_node(node_for_adding=self.current_nid, type=var_type)
                    self.vid2nid[vid] = self.current_nid
                    self.nid2vid[self.current_nid] = vid
                    self.current_nid += 1

        for sc in scope_vids:
            self.add_edges_from( itertools.combinations( (self.vid2nid[vid] for vid in sc), 2) )

    def vids2nids(self, vids: Iterable[int])->List[int]:
        return [self.vid2nid[vid] for vid in vids]


def neighbors(G: nx.Graph, nid: Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nid, int):
        return SortedSet(G.neighbors(nid))
    r = SortedSet()
    for n in nid:
        r.update(G.neighbors(n))
    return r


def markov_blanket_undirected(G: Union[nx.DiGraph, nx.Graph], nids: Union[int, Iterable[int]]) -> SortedSet:
    if isinstance(nids, int):
        nids = SortedSet([nids])
    r = SortedSet()
    # undirected graph mb == nhd
    r.update(neighbors(G, nids))
    r.difference_update(nids)
    return r


def get_induced_width_from_ordering(primal_graph:PrimalGraph, nid_ordering:Iterable[int]):
    pg = primal_graph.copy()
    induced_width = 0
    for nid in nid_ordering:
        induced_width = max(induced_width, pg.degree[nid])        # induced_width = cluster_size - 1 (exclude itself)
        pg.add_edges_from(itertools.combinations(pg.neighbors(nid), 2))
        pg.remove_node(nid)
    return induced_width


def greedy_variable_order(primal_graph:PrimalGraph, pvo:List[List[int]]=None, pool_size=8, cutoff=INF):
    """
    pvo is list of list of node ids of the primal graph
    """
    def fill_count(nid):
        """
        count number of fill-in edges after removing nid
            number of combinations of nhd - existing edges (nodes in the subgraph of nhd)
        """
        n_edges = G.subgraph(G.neighbors(nid)).number_of_edges()
        deg = G.degree[nid]
        n_fill = deg*(deg-1)//2 - n_edges
        return n_fill

    def remove_fill_in_edges(nid):
        G.add_edges_from(itertools.combinations(G.neighbors(nid), 2))        # adding edge twice? no effect
        G.remove_node(nid)

    G = primal_graph.copy()     # G = copy.deepcopy(primal_graph)
    if pvo is None:
        pvo = [list(G.nodes())] #[ [all in one block] ]
    ordering = []
    induced_width = 0
    for each_block in pvo:
        processing_nodes = SortedList( [(fill_count(nid), nid) for nid in each_block]  )    # ascending order
        while processing_nodes:
            fill, selected_nid = processing_nodes[0]
            if fill != 0:       # don't add any edge
                # pick a node in random from a pool of best nodes; each node has prob 1/(fill_in edges)
                scores, candidates = zip(*processing_nodes[:pool_size])
                probs = np.power(np.array(scores), -1.0)
                selected_ind = np.random.choice(len(probs), p=probs/(np.sum(probs)))
                selected_nid = candidates[selected_ind]
            ordering.append(selected_nid)
            # current_width = len(G.neighbors(selected_nid))
            current_width = G.degree[selected_nid]
            if current_width > cutoff:
                return None, induced_width
            if  current_width > induced_width:
                induced_width = current_width
            remove_fill_in_edges(selected_nid)
            # recompute score after removing the selected node from primal graph
            processing_nodes = SortedList( [(fill_count(nid), nid) for _, nid in processing_nodes if nid != selected_nid] )
    return ordering, induced_width


def iterative_greedy_variable_order(primal_graph:PrimalGraph, iter_limit=100, pvo:List[List[int]]=None, pool_size=8, cutoff=INF):
    order_found, improved = None, 0
    for i in range(iter_limit):
        if order_found and i-improved > 10:
            break
        ordering, induced_width = greedy_variable_order(primal_graph, pvo, pool_size, cutoff)
        if ordering and induced_width < cutoff:
            cutoff = induced_width
            order_found = ordering
            improved = i
            # if DEBUG__:
            #     print("iter:{}/{} iw;{}".format(i+1, iter_limit, cutoff))
    return order_found, cutoff


def call_variable_ordering(primal_graph, iter_limit, pvo_vid, pool_size=16, cutoff=INF):
    pvo_nodes = []
    for block in pvo_vid:
        pvo_nodes.append( [primal_graph.vid2nid[vid] for vid in block] )
    ordering, iw = iterative_greedy_variable_order(primal_graph, iter_limit, pvo_nodes, pool_size, cutoff)
    ordering = [primal_graph.nid2vid[nid] for nid in ordering]
    return ordering, iw
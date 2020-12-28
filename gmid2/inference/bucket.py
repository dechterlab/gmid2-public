"""
A bucket data structure is a data structure
that uses the key values as the indices of the buckets,
and store items of the same key value in the corresponding bucket.

A double bucket data structure,
in most cases, refers to the bucket data structure where each bucket contains a bucket data structure.

Mini-bucket is a double bucket data structure but treat each mini-bucket as a bucket with double keys
to form a join graph as a graph of buckets
Label = NamedTuple("Label", ["vid", "ind"])
bucket -> key [varid][0]
mini-bucket -> key [varid]
"""
# from __future__ import annotations
from typing import Iterable, List, Optional
import itertools
from copy import copy, deepcopy
from sortedcontainers import SortedSet, SortedDict
import networkx as nx

from gmid2.basics.factor import Factor
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.message import Message, Separator, ConsistencyConstraint, ScopedMessage
from gmid2.basics.directed_network import topo_sort, rev_topo_sort


class Bucket:
    def __init__(self, var_label: int, ind: int,
                 vids_fids: Optional[Iterable[int]], fids: Optional[Iterable[int]],
                 vids_mids: Optional[Iterable[int]], mids: Optional[Iterable[int]],
                 *args, **kwargs):
        self.bid = None
        self.var_label = var_label
        self.ind = ind
        self.vids_fids = SortedSet(vids_fids) if vids_fids else SortedSet()
        self.vids_mids = SortedSet(vids_mids) if vids_mids else SortedSet()
        self.fids = SortedSet(fids) if fids else SortedSet()        # empty set or None?
        self.mids = SortedSet(mids) if mids else SortedSet()        # bucket is destination of mids

        self.scope_vids = self.vids_fids | self.vids_mids
        super().__init__(*args, **kwargs)

    def __str__(self):
        output = type(self).__name__ + "_{}:({}, {})[v:[{}],f:[{}],m:[{}]]"
        return output.format(self.bid, self.var_label, self.ind,
                             ",".join(str(el) for el in self.scope_vids),
                             ",".join(str(el) for el in self.fids),
                             ",".join(str(el) for el in self.mids)
                             )
    __repr__ = __str__

    # @property
    # def scope_vids(self):
    #     # self._update_vids()
    #     return self.scope_vids
    #
    # def update_scope_vids(self, vids: SortedSet):
    #     self.scope_vids.update(vids)     # not exactly setter

    def update_vids(self):
        self.scope_vids = self.vids_fids | self.vids_mids

    def merge(self, other):
        """
        merging two buckets is simply merging the scope and function ids in it
        """
        self.vids_fids.update(other.vids_fids)
        self.vids_mids.update(other.vids_mids)
        self.fids.update(other.fids)
        self.mids.update(other.mids)
        self.scope_vids = self.vids_fids | self.vids_mids

    # @property
    # def is_empty(self):
    #     if len(self.scope_vids) == 0:
    #         assert len(self.fids) == 0 and len(self.mids) == 0, "bucket.is_empty: if vids is empty fids and mids too"
    #         return True
    #     else:
    #         return False

    def add_factor(self, f:Factor):
        self.fids.add(f.fid)
        self.vids_fids.update(f.scope_vids)
        self.update_vids()

    def add_msg(self, msg:ScopedMessage):
        self.mids.add(msg.mid)
        self.vids_mids.update(msg.scope_vids)
        self.update_vids()


class BucketGraph(nx.DiGraph):
    def __init__(self):
        self.bid2bucket = SortedDict()
        self.src2mids = SortedDict()
        self.mid2message = SortedDict()     # useful to know messages in the graph as it's awkward to modify gm
        self.edge2mid = SortedDict()
        self.next_bid = 0
        super().__init__()

    def register_message(self, m: Message) -> None:
        self.mid2message[m.mid] = m

    def add_bucket_node(self, bucket:Bucket) -> int:
        self.add_node(self.next_bid)
        bucket.bid = self.next_bid
        self.bid2bucket[self.next_bid] = bucket
        if self.next_bid not in self.src2mids:
            self.src2mids[self.next_bid] = SortedSet()      # avoid KeyError when bucket node is a leaf
        self.next_bid += 1
        return self.next_bid - 1

    def remove_node_by_bid(self, bid:int):
        while self.bid2bucket[bid].mids:
            mid = self.bid2bucket[bid].mids.pop()
            self.remove_edge_by_mid(mid)        # bug this in_mids changes while process!

        if bid in self.src2mids:
            while self.src2mids[bid]:
                mid = self.src2mids[bid].pop()
                self.remove_edge_by_mid(mid)

        del self.bid2bucket[bid]
        self.remove_node(bid)           # removing from nx graph

    def _add_src2mids(self, src, mid):
        if src not in self.src2mids:
            self.src2mids[src] = SortedSet()
        self.src2mids[src].add(mid)

    def _remove_src2mids(self, src, mid):
        self.src2mids[src].discard(mid)

    def _add_mid_to_bucket(self, bid, mid):
        self.bid2bucket[bid].mids.add(mid)

    def _remove_mid_from_bucket(self, bid, mid):
        self.bid2bucket[bid].mids.discard(mid)

    def _add_vid_to_bucket_from_mid(self, bid, mid):
        self.bid2bucket[bid].vids_mids.update(self.mid2message[mid].scope_vids)
        self.bid2bucket[bid].update_vids() #= self.bid2bucket[bid].vids_fids | self.bid2bucket[bid].vids_mids

    def _remove_vid_in_bucket_from_mid(self, bid, mid):
        removable = copy(self.mid2message[mid].scope_vids)
        for other_mid in self.bid2bucket[bid].mids:
            if other_mid == mid: continue
            removable.difference_update(self.mid2message[other_mid].scope_vids)     # remove vars in others cannot remove them
        if removable:
            self.bid2bucket[bid].vids_mids -= removable
            self.bid2bucket[bid].update_vids()

    def reset_dest_mid(self, mid: int, dest: int):
        m = self.mid2message[mid]
        m.dest = dest

    def reset_src_mid(self, mid:int, src:int):
        m = self.mid2message[mid]
        m.src = src

    def add_edge_by_mid(self, mid: int):
        """
        add edge in bucket graph after registering message by mid
        only add edge if it's fresh new
        if there's one already subsume the scope to the existing one
        """
        m = self.mid2message[mid]
        for dup_mid in self.src2mids[m.src]:
            if self.mid2message[dup_mid].dest == m.dest:    # if (src,dest) same
                if len(m.scope_vids - self.mid2message[dup_mid].scope_vids) > 0:    # not subset
                    self.mid2message[dup_mid].merge_scope(m)    # merge m to duplicated one
                    self._add_vid_to_bucket_from_mid(m.dest, mid)       # add vid to the destination due to dupicating msg
                self._remove_src2mids(m.dest, mid)  # remove message object
                del self.mid2message[mid]
                return

        self._add_src2mids(m.src, mid)
        self._add_mid_to_bucket(m.dest, mid)
        self._add_vid_to_bucket_from_mid(m.dest, mid)
        self.edge2mid[m.src, m.dest] = mid
        self.add_edge(m.src, m.dest, msg_type=type(m).__name__)

    def remove_edge_by_mid(self, mid: int):
        src, dest = self.mid2message[mid].src, self.mid2message[mid].dest
        self._remove_src2mids(src, mid)
        self._remove_mid_from_bucket(dest, mid)
        self._remove_vid_in_bucket_from_mid(dest, mid)
        del self.edge2mid[(src,dest)]           # this is separate table
        self.remove_edge(src, dest)      # remove edge from graph
        del self.mid2message[mid]           # this is removing actual mid and message connectionrd

    def __str__(self):
        output  = type(self).__name__ + "\n"
        for bids in topo_sort(self):
            for bid in sorted(bids):
                bucket = self.bid2bucket[bid]
                output += str(bucket) + "\n"
                if bid in self.src2mids:
                    for mid in self.src2mids[bid]:
                        output += str(self.mid2message[mid]) + "\n"
                output += "\n"
        return output

    def find_buckets_with_label(self, vid:int)->List[int]:
        ret = []
        for bid in self:
            bucket = self.bid2bucket[bid]
            if bucket.var_label == vid:
                ret.append(bid)
        return ret

    def find_bucket_with_fid(self, fid:int)->int:
        for bid in self:
            bucket = self.bid2bucket[bid]
            if fid in bucket.fids:
                return bid

    def remove_fids(self, fids:Iterable[int]):
        for bid in self:
            self.bid2bucket[bid].fids.difference_update(fids)   # remove any fids intersect with fids


def bucket_tree_decomposition(gm: GraphicalModel, vid_elim_order: List)->BucketGraph:
    bt = BucketGraph()
    vid2incoming_mids = SortedDict({vid:SortedSet() for vid in vid_elim_order})     # store messages until it assigned to a bucket
    fids_assigned = SortedSet()
    fids_unassigned = SortedSet(gm.fid2f.keys())
    mids_unassigend = SortedSet()
    for i, vid in enumerate(vid_elim_order):
        bucket_scope, bucket_fid_scope, bucket_mid_scope = SortedSet(), SortedSet(), SortedSet()
        fids_with_vid = gm.vid2fids[vid] - fids_assigned
        for fid in fids_with_vid:
            bucket_fid_scope.update(gm.fid2f[fid].scope_vids)
        mids_with_vid = vid2incoming_mids[vid]
        for mid in mids_with_vid:
            bucket_mid_scope.update(bt.mid2message[mid].scope_vids)
        bucket_scope = bucket_fid_scope | bucket_mid_scope
        # create a bucket object
        bucket = Bucket(vid, 0, bucket_fid_scope, fids_with_vid, bucket_mid_scope, mids_with_vid)
        fids_assigned.update(fids_with_vid)
        fids_unassigned.difference_update(fids_assigned)
        mids_unassigend.difference_update(mids_with_vid)

        bid = bt.add_bucket_node(bucket)       # only added node
        # set destinations for the incoming message and add edges to the bucket tree
        for mid in mids_with_vid:
            bt.reset_dest_mid(mid, bid)
            bt.add_edge_by_mid(mid)

        if i == len(vid_elim_order) - 1:  # last bucket, no outgoing msg
            for fid in fids_unassigned:
                bucket.add_factor(gm.fid2f[fid])
            for mid in mids_unassigend:
                bucket.add_msg(bt.mid2message[mid])
                bt.reset_dest_mid(mid, bucket.bid)
                bt.add_edge_by_mid(mid)
        else:
            # create outgoing msg from bucket
            bucket_scope.remove(vid)
            if bucket_scope:        # if scope is empty, bucket is the root; no outgoing msg
                msg = Separator(bucket_scope, src=bid)
                bt.register_message(msg)
                mids_unassigend.add(msg.mid)
                for j in range(i+1, len(vid_elim_order)):   # place msg to the earliest variable in the elim order
                    if vid_elim_order[j] in bucket_scope:
                        vid2incoming_mids[vid_elim_order[j]].add(msg.mid)
                        break

    return bt


# regular mini bucket tree eliminating all variables
def mini_bucket_tree_decomposition(gm: GraphicalModel, vid_elim_order: List[int], ibound:int)->BucketGraph:
    mbt = BucketGraph()
    vid2incoming_mids = SortedDict({vid:SortedSet() for vid in vid_elim_order})     # store messages until it assigned to a bucket
    fids_assigned = SortedSet()
    for vid_ind, vid in enumerate(vid_elim_order):
        factors_in_bucket = []      # let the order be fixed
        mids_with_vid = vid2incoming_mids[vid]
        factors_in_bucket.extend(mbt.mid2message[mid] for mid in mids_with_vid)
        fids_with_vid = gm.vid2fids[vid] - fids_assigned
        factors_in_bucket.extend(gm.fid2f[fid] for fid in fids_with_vid)

        # place all initial mini_buckets
        mini_buckets = []
        for mb_ind, f in enumerate(factors_in_bucket):
            if isinstance(f, Separator):
                mini_buckets.append(Bucket(vid, mb_ind, vids_fids=None, fids=None, vids_mids=f.scope_vids, mids=[f.mid]))
            elif isinstance(f, Factor):
                mini_buckets.append(Bucket(vid, mb_ind, vids_fids=f.scope_vids, fids=[f.fid], vids_mids=None, mids=None))

        merge_mini_buckets(mini_buckets, ibound)

        # add mini_buckets to mini_bucket_tree
        for mb_ind, bucket in enumerate(mini_buckets):
            bucket.ind = mb_ind
            bid = mbt.add_bucket_node(bucket)
            fids_assigned.update(bucket.fids)
            for mid in bucket.mids:                     # set destinations for the msg incoming to the mini_bucket
                mbt.reset_dest_mid(mid, bid)
                mbt.add_edge_by_mid(mid)
            if vid_ind < len(vid_elim_order)-1:         # send message if it is not the last layer
                msg_scope = bucket.scope_vids - {vid}       # create outgoing msg from mini_bucket
                if msg_scope:
                    msg = Separator(msg_scope, src=bid)
                    mbt.register_message(msg)
                    # mids_unassigend.add(msg.mid)
                    for j in range(vid_ind + 1, len(vid_elim_order)):   # place to the earliest variable in elim order
                        if vid_elim_order[j] in msg_scope:
                            vid2incoming_mids[vid_elim_order[j]].add(msg.mid)
                            break

        # connect mini-buckets by chain from left to right (don't need to add this for mini-bucket tree)
        for b_ind in range(len(mini_buckets)-1):
            b1, b2 = mini_buckets[b_ind], mini_buckets[b_ind+1]
            msg = ConsistencyConstraint([vid], src=b1.bid, dest=b2.bid)
            mbt.register_message(msg)
            b2.mids.add(msg.mid)
            mbt.add_edge_by_mid(msg.mid)
    return mbt


def merge_mini_buckets(mini_buckets:List[Bucket], ibound:int):
    mini_buckets.sort(key=lambda x: (len(x.scope_vids), x.ind))  # smaller one and created earlier one comes first
    j = 0
    while j < len(mini_buckets):
        if j == len(mini_buckets) - 1:
            break
        for k in range(j + 1, len(mini_buckets)):
            merged_scope_size = len(mini_buckets[j].scope_vids | mini_buckets[k].scope_vids)
            if merged_scope_size <= ibound + 1 or mini_buckets[j].scope_vids <= mini_buckets[k].scope_vids:
                b1, b2 = mini_buckets[j], mini_buckets[k]
                b2.merge(b1)
                mini_buckets[k], mini_buckets[-1] = mini_buckets[-1], mini_buckets[k]
                mini_buckets.pop()
                mini_buckets[j], mini_buckets[-1] = mini_buckets[-1], mini_buckets[j]
                mini_buckets.pop()  # swap & pop or pop(j)
                mini_buckets.append(b2)  # add merged one back (also OK with merging to k-th and bring it back)
                mini_buckets.sort(key=lambda x: (len(x.scope_vids), x.ind))  # one created earlier comes first x.ind
                j = 0
                break
        else:
            j += 1  # didn't break while checking (j, j+1...n-1)


def connect_mini_buckets(mini_buckets:List[Bucket])->List[int]:
    """
    connect mini buckets with possibly different variable labels by maximal separators
    """
    mini_bucket_inds = []
    g_score = dict()
    for src in range(len(mini_buckets)):
        g_score[src] = dict()
        for dest in range(len(mini_buckets)):
            if src != dest:
                sep_size = len(mini_buckets[src].scope_vids & mini_buckets[dest].scope_vids)
                g_score[src][dest] = sep_size

    nodes = SortedSet(range(len(mini_buckets)), key=lambda x: len(mini_buckets[x].scope_vids))
    current = nodes.pop()  # start with the largest scope policy
    mini_bucket_inds.append(current)
    while nodes:
        right = max(g_score[current], key=g_score[current].get)     # current -> right
        if right in mini_bucket_inds:
            del g_score[current][right]
        else:
            del g_score[current]    # current is not in score dict anymore
            nodes.remove(right)
            current = right
            mini_bucket_inds.append(current)
    return mini_bucket_inds


# works well with gdd after fixing mbte
def join_graph_decomposition(mbt: BucketGraph, vid_elim_order:List[int])->BucketGraph:
    """
    construct join graph by merging a strict subset of mini-buckets
    bottom-up process select a leaf and merged it into one of its parent if possible
    """
    pop_next, iter = True, itertools.chain.from_iterable(rev_topo_sort(mbt))
    while True:
        if not pop_next:
            iter = itertools.chain.from_iterable(rev_topo_sort(mbt))
        try:
            current_bid = next(iter)
            pop_next = True
        except StopIteration:
            break
        current_bucket = mbt.bid2bucket[current_bid]
        pa_bids = mbt.predecessors(current_bid)
        pa_bids = sorted(pa_bids, key= lambda x: vid_elim_order.index(mbt.bid2bucket[x].var_label) )
        for pa_bid in reversed(pa_bids):
            if current_bucket.scope_vids <= mbt.bid2bucket[pa_bid].scope_vids:
                # merge into parent bucket
                for src, _ in mbt.in_edges(current_bid):        # for all parents
                    if src != pa_bid:   # this is not the place to be merged, create a message to pa_bid by copying src to current
                        msg = mbt.mid2message[mbt.edge2mid[src, current_bid]]
                        # (change) earlier elim bucket becomes parent
                        if vid_elim_order.index(mbt.bid2bucket[src].var_label) < vid_elim_order.index(mbt.bid2bucket[pa_bid].var_label):
                            msg_copy = Separator(scope_vids=msg.scope_vids, src=src, dest=pa_bid)
                        else:
                            msg_copy = Separator(scope_vids=msg.scope_vids, src=pa_bid, dest=src)
                        mbt.register_message(msg_copy)
                        mbt.add_edge_by_mid(msg_copy.mid)       # add more edge (still src-> parent exists)
                for _, dest in mbt.out_edges(current_bid):
                    # current -> dest to pa_bid -> dest
                    msg = mbt.mid2message[mbt.edge2mid[current_bid, dest]]
                    msg_copy = Separator(scope_vids=msg.scope_vids, src=pa_bid, dest=dest)
                    mbt.register_message(msg_copy)
                    mbt.add_edge_by_mid(msg_copy.mid)
                # remove the current node & all incoming & outgoing arcs
                mbt.bid2bucket[pa_bid].fids.update(current_bucket.fids)        # vids is already subsumed
                mbt.remove_node_by_bid(current_bid)     # transfer fids
                pop_next = False
                break
    return mbt


# mini bucket tree for submodel eliminating subset of variables leaving interface buckets
def mini_bucket_tree_decomposition_st(gm: GraphicalModel, vid_elim_order: List[int], ibound:int,
                                      interface_vids: SortedSet)->BucketGraph:
    """
    do mini-bucket tree decomposition but stop sending branch if var label is in interface variable
    the root nodes (nodes without outgoing edges) are interface nodes
    """
    mbt = BucketGraph()
    vid2incoming_mids = SortedDict({vid:SortedSet() for vid in vid_elim_order})
    fids_assigned = SortedSet()
    for vid_ind, vid in enumerate(vid_elim_order):
        factors_in_bucket = []      # let the order be fixed
        mids_with_vid = vid2incoming_mids[vid]
        factors_in_bucket.extend(mbt.mid2message[mid] for mid in mids_with_vid)
        fids_with_vid = gm.vid2fids[vid] - fids_assigned
        factors_in_bucket.extend(gm.fid2f[fid] for fid in fids_with_vid)

        # place all initial mini_buckets
        mini_buckets = []
        for mb_ind, f in enumerate(factors_in_bucket):
            if isinstance(f, Separator):
                mini_buckets.append(Bucket(vid, mb_ind, vids_fids=None, fids=None, vids_mids=f.scope_vids, mids=[f.mid]))
            elif isinstance(f, Factor):
                mini_buckets.append(Bucket(vid, mb_ind, vids_fids=f.scope_vids, fids=[f.fid], vids_mids=None, mids=None))

        merge_mini_buckets(mini_buckets, ibound)

        # add mini_buckets to mini_bucket_tree
        for mb_ind, bucket in enumerate(mini_buckets):
            bucket.ind = mb_ind
            bid = mbt.add_bucket_node(bucket)
            fids_assigned.update(bucket.fids)
            for mid in bucket.mids:                     # set destinations for the msg incoming to the mini_bucket
                mbt.reset_dest_mid(mid, bid)
                mbt.add_edge_by_mid(mid)
            # send message if it is not an interface bucket of the last layer
            if vid not in interface_vids and vid_ind < len(vid_elim_order)-1:
                msg_scope = bucket.scope_vids - {vid}       # create outgoing msg from mini_bucket
                if msg_scope:
                    msg = Separator(msg_scope, src=bid)
                    mbt.register_message(msg)
                    for j in range(vid_ind + 1, len(vid_elim_order)):   # place to the earliest variable in elim order
                        if vid_elim_order[j] in msg_scope:
                            vid2incoming_mids[vid_elim_order[j]].add(msg.mid)
                            break

        for b_ind in range(len(mini_buckets)-1):
            b1, b2 = mini_buckets[b_ind], mini_buckets[b_ind+1]
            msg = ConsistencyConstraint([vid], src=b1.bid, dest=b2.bid)
            mbt.register_message(msg)
            b2.mids.add(msg.mid)
            mbt.add_edge_by_mid(msg.mid)
    return mbt
# from __future__ import annotations
from typing import Iterable
from sortedcontainers import SortedSet, SortedDict
from gmid2.basics.factor import Variable, Factor

class Message:
    """
    Message is a composite of factors created from one cluster and send to another cluster
    """
    mid = 0

    def __init__(self, src: int=None, dest: int=None):
        self.mid = Message.mid
        Message.mid += 1
        self.src = src
        self.dest = dest

    @classmethod
    def reset_mid(cls):
        cls.mid = 0

    def __hash__(self):
        return hash(self.mid)

    def __str__(self):
        return type(self).__name__ + self.str_post_fix()

    def str_post_fix(self):
        return "_{}:[{}, {}]".format(self.mid, self.src, self.dest)


class PolicyConstraint(Message):        # used for submodel graph decomposition
    def __init__(self, *args, **kwargs):
    # def __init__(self, src:int=None, dest: int=None):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return type(self).__name__ + self.str_post_fix()


class ScopedMessage(Message):
    def __init__(self, scope_vids: Iterable[int], *args, **kwargs):
        self.scope_vids = SortedSet(scope_vids)
        super().__init__(*args, **kwargs)

    def __str__(self):
        return type(self).__name__ +  self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))

    def merge_scope(self, other):
        self.scope_vids.update(other.scope_vids)


class TransitionConstraint(ScopedMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return type(self).__name__ +  self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))

# class TransitionConstraint(Message):
#     def __init__(self, scope_vids: Iterable[int], *args, **kwargs):
#     # def __init__(self, scope_vids: Iterable[int], src: int = None, dest: int = None):
#         self.scope_vids = SortedSet(scope_vids)
#         super().__init__(*args, **kwargs)
#
#     def __str__(self):
#         return type(self).__name__ +  self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))


class Separator(ScopedMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return type(self).__name__ + self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))


# class Separator(Message):
#     def __init__(self, scope_vids: Iterable[int], *args, **kwargs):
#         self.scope_vids = SortedSet(scope_vids)
#         super().__init__(*args, **kwargs)
#
#     def __str__(self):
#         return type(self).__name__ + self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))


class ConsistencyConstraint(ScopedMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return type(self).__name__ + self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))

# class ConsistencyConstraint(Message):
#     def __init__(self, scope_vids: Iterable[int], *args, **kwargs):
#     # def __init__(self, scope_vids: Iterable[int], src: int=None, dest: int = None):
#         self.scope_vids = SortedSet(scope_vids)
#         super().__init__(*args, **kwargs)
#
#     def __str__(self):
#         return type(self).__name__ + self.str_post_fix() + "[v:[{}]]".format(",".join(str(el) for el in self.scope_vids))



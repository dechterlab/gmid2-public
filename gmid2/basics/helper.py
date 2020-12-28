from typing import Generator, Iterable, Iterator
from sortedcontainers import SortedSet


def reduce_tuples(input_list: Iterable, pos: int=0) -> Generator[int, None, None]:
    return (el[pos] for el in input_list)


def filter_subsets(subsets: Iterable[SortedSet], filter_set: SortedSet) -> Iterator[SortedSet]:
    for t in subsets:
        ss = filter_set & t
        if ss:
            yield ss

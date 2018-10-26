#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import ray

# TODO parallelize


def partition(xs) -> List[slice]:
    def split(xs) -> List[int]:
        # `bin` gives str '0b----', so we drop the first 2 chars, '0b'. We can
        # ignore the negative case since length is always nonnegative.
        base2_decomp = reversed([int(x) for x in f"{len(xs):b}"])

        pows = sorted([i for i, a in enumerate(base2_decomp) if a != 0])
        return pows

    parts = split(xs)

    slices, start = [], 0

    for n in parts:
        stop = start + 2 ** n
        slices.append(slice(start, stop))
        start = stop

    return slices


@ray.remote()
def foldr(f, xs):
    L = len(xs)
    if L == 1:
        return xs[0]
    elif L == 0:
        raise ValueError("Sequence must have length greater than 0.")
    else:
        slices = partition(xs)

        @ray.remote
        def _foldr(chunk):
            while len(chunk) > 1:
                chunk = [
                    f(chunk[2 * i], chunk[2 * i + 1]) for i, _ in enumerate(chunk[::2])
                ]
            return chunk[0]

        return foldr(f, [_foldr.remote(xs[slice]) for slice in slices])


if __name__ == "__main__":
    from operator import add
    import hypothesis
    import hypothesis.strategies as st
    from hypothesis import assume, example, given, infer
    from functools import reduce

    @given(xs=infer)
    def test_foldr(f, xs: List[int]):
        assume(xs != [])
        assert foldr(f, xs) == reduce(f, xs)

    f = add
    test_foldr(f)

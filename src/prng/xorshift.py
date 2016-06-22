"""
This file contains routines that implement xorshift random number
generators. These algorithms are chosen for their speed and periods
that are long enough. Benefitted greatly from others who had put
their code online (with licenses I'm comfortable with) and without
that guidance this would have been significantly harder -- In
particular, I found work by David Blackman and Sebastiano Vigna to
be of particular use and ultimately this code is nearly an identical
copy to what they have published (see 2nd link) under public domain.

Algorithms currently implemented:
    * xoroshiro128+

References:
    * Wikipedia (https://en.wikipedia.org/wiki/Xorshift)
    * http://xoroshiro.di.unimi.it/
author: Chase Coleman
date: 21 June 2016
"""
import numpy as np

from numba import jit, f8, u8


_PRNGSTATE = np.array([u8(125), u8(234523)])


@jit("u8(u8, u8)", nopython=True, locals={"sf": u8})
def rotl(x, k):
    sf = 64
    return (x<<k) | (x >> (sf - k))


@jit("u8(u8[:])", nopython=True, locals={"result": u8, "s0": u8, "s1": u8, "ff": u8, "ft": u8, "ts": u8})
def next(prngstate):
    # Pull out the states
    s0 = prngstate[0]
    s1 = prngstate[1]

    # Return their sum
    result = s0 + s1

    # Store values we care about
    ff = 55
    ft = 14
    ts = 36

    # Update their values
    s1 ^= s0
    prngstate[0] = rotl(s0, ff) ^ s1 ^ (s1 << ft)
    prngstate[1] = rotl(s1, ts)

    return result


@jit("f8[:](u8, u8[:])", nopython=True, locals={"bn": u8})
def rand(n, prngstate):
    """
    Draws random numbers U[0, 1]
    """
    x = np.empty(n)
    bn = 2**64 - 1

    for i in range(n):
        x[i] = next(prngstate) / bn
    return x


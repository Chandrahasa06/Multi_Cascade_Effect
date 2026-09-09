"""A minimal HyperLogLog cardinality sketch.

Used by :mod:`dataplane.src_table` to approximate distinct destination
ports/IPs per source without keeping an exact set — exact sets are not
P4-implementable at line rate; a fixed-size register array with a
max-update rule is. ``add`` and ``merge`` are both O(1) (merge is
O(m) in the number of registers, a small constant, not O(n) in items
seen).
"""
from __future__ import annotations

import math
import zlib


def _alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1 + 1.079 / m)


def _hash32(item) -> int:
    """CRC32 is linear (CRC(x XOR y) relates linearly to CRC(x), CRC(y)),
    so it correlates on structured/sequential input — exactly the shape a
    port scan produces (dest port N, N+1, N+2, ...). Running its output
    through Murmur3's fmix32 avalanche step breaks that correlation so
    each bit of the final hash depends on all input bits roughly equally,
    which is what the rank/index split below assumes."""
    h = zlib.crc32(repr(item).encode("utf-8")) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def _rho(w: int, max_bits: int) -> int:
    """1 + position of the lowest set bit in ``w`` (a max_bits-wide value);
    ``max_bits + 1`` if ``w`` is all zero."""
    if w == 0:
        return max_bits + 1
    rank = 1
    while not (w & 1):
        w >>= 1
        rank += 1
    return rank


class HyperLogLog:
    """Fixed-precision HLL sketch. ``precision`` bits select the register
    (``2**precision`` registers total); the rest of the hash estimates the
    leading-zero run length stored in that register."""

    __slots__ = ("precision", "m", "registers")

    def __init__(self, precision: int = 6):
        if not (4 <= precision <= 16):
            raise ValueError("precision must be between 4 and 16")
        self.precision = precision
        self.m = 1 << precision
        self.registers = bytearray(self.m)

    def add(self, item) -> None:
        h = _hash32(item)
        idx = h & (self.m - 1)
        remaining = h >> self.precision
        rank = _rho(remaining, 32 - self.precision)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        if self.precision != other.precision:
            raise ValueError("cannot merge sketches of different precision")
        merged = HyperLogLog(self.precision)
        merged.registers = bytearray(
            a if a > b else b for a, b in zip(self.registers, other.registers)
        )
        return merged

    def cardinality(self) -> float:
        m = self.m
        z = sum(2.0**-r for r in self.registers)
        raw = _alpha(m) * m * m / z
        if raw <= 2.5 * m:
            zeros = self.registers.count(0)
            if zeros != 0:
                return m * math.log(m / zeros)
        return raw

    @property
    def standard_error(self) -> float:
        """Theoretical relative standard error, ``1.04 / sqrt(m)``."""
        return 1.04 / math.sqrt(self.m)

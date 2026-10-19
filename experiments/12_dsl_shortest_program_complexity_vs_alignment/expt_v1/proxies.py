"""Computable stand-ins for complexity on a truth table (PROTOCOL Section 3.2): CTM/BDM via pybdm and compressors.

pybdm (https://github.com/sztal/pybdm, version 0.1.0) implements the Block Decomposition Method of Zenil et al.
(Entropy 2018, arXiv 1609.00110) on top of the Coding Theorem Method tables of Soler-Toscano et al. (PLoS ONE 2014,
arXiv 1211.1302). BDM(ndim=1) uses the CTM table for binary strings of length 12 with the default PartitionIgnore,
so the last 10,240 mod 12 = 4 bits are dropped.
"""
import lzma
import zlib

import numpy as np

WORD_BITS = 10

try:
    from pybdm import BDM
    _BDM = BDM(ndim=1)
    PYBDM_ERROR = None
except Exception as e:  # pragma: no cover - fallback path, recorded in the manifest
    _BDM = None
    PYBDM_ERROR = repr(e)


def rowmajor_bits(table):
    """10,240 bits: inputs x = 0..1023 in order, each output written bits 0..9."""
    t = np.asarray(table).astype(np.int64)
    return ((t[:, None] >> np.arange(WORD_BITS)) & 1).astype(np.uint8).reshape(-1)


def bitplane_bits(table):
    """10,240 bits: all 1,024 values of output bit 0, then bit 1, and so on."""
    t = np.asarray(table).astype(np.int64)
    return ((t[None, :] >> np.arange(WORD_BITS)[:, None]) & 1).astype(np.uint8).reshape(-1)


def bdm_value(bits):
    return float(_BDM.bdm(np.asarray(bits, int)))


def block_entropy_fallback(bits, block=12):
    """Labelled substitute used only if pybdm is unavailable: Shannon entropy (bits) of non-overlapping 12-bit blocks
    times the number of blocks."""
    b = np.asarray(bits, np.int64)
    n = len(b) // block
    v = (b[: n * block].reshape(n, block) << np.arange(block)).sum(1)
    _, cnt = np.unique(v, return_counts=True)
    p = cnt / n
    return float(-np.sum(p * np.log2(p)) * n)


def proxies(table, c=None):
    zl = (c or {}).get('zlib_level', 9)
    lp = (c or {}).get('lzma_preset', 9)
    rm, bp = rowmajor_bits(table), bitplane_bits(table)
    packed = np.packbits(rm).tobytes()
    out = dict(zlib_bytes=len(zlib.compress(packed, zl)), lzma_bytes=len(lzma.compress(packed, preset=lp)),
               zlib_u16_bytes=len(zlib.compress(np.asarray(table, '<u2').tobytes(), zl)))
    if _BDM is not None:
        out.update(bdm_rowmajor=bdm_value(rm), bdm_bitplane=bdm_value(bp), bdm_source='pybdm')
    else:
        out.update(bdm_rowmajor=block_entropy_fallback(rm), bdm_bitplane=block_entropy_fallback(bp),
                   bdm_source='substitute: 12-bit block Shannon entropy')
    return out

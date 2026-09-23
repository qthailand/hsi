# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange, set_num_threads


@njit(parallel=True, fastmath=True, cache=True)
def _bin_average_uint16_numba(X, iL, iR, w):
    b, h, v = X.shape
    n_bins = iL.shape[0]
    Xf = X.reshape(b * h, v)
    out = np.zeros((b * h, n_bins), dtype=np.uint16)

    for r in prange(b * h):
        for k in range(n_bins):
            il = iL[k]; ir = iR[k]; ww = w[k]
            s = 0
            for j in range(il, ir):
                s += Xf[r, j]
            val = (s + (ww // 2)) // ww
            if val > 65535:
                val = 65535
            out[r, k] = val
    return out.reshape(b, h, n_bins)


class LambdaBinner:
    def __init__(self, m: float, c: float, step: float = 2.0,
                 threads: Optional[int] = None):
        self.m = float(m)
        self.c = float(c)
        self.step = float(step)
        self.threads = threads

        self.v_: Optional[int] = None
        self.iL: Optional[np.ndarray] = None
        self.iR: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self._lambda_out: Optional[np.ndarray] = None

    # ---------- precompute ----------
    def _precompute_bins(self, v: int):
        m, c, step = self.m, self.c, self.step
        lam_first = c
        lam_last = m * (v - 1) + c

        n_edges = int(np.round((lam_last - lam_first) / step)) + 1
        edges = np.linspace(lam_first, lam_last, n_edges, dtype=np.float32)

        if edges.size < 2:
            edges = np.array([lam_first, lam_last], dtype=np.float32)

        L = edges[:-1]
        R = edges[1:]

        iL = np.ceil((L - c) / m).astype(np.int32)
        iR = np.floor((R - c) / m).astype(np.int32) + 1

        iL = np.clip(iL, 0, v)
        iR = np.clip(iR, 0, v)

        w = (iR - iL).astype(np.int32)
        ok = w > 0

        return iL[ok], iR[ok], w[ok], ((L[ok] + R[ok]) * 0.5).astype(np.float32)

    # ---------- public API ----------
    def fit(self, v: int) -> "LambdaBinner":
        self.v_ = int(v)
        self.iL, self.iR, self.w, self._lambda_out = self._precompute_bins(self.v_)
        return self

    def fit_from(self, spectrogram: np.ndarray) -> "LambdaBinner":
        return self.fit(int(spectrogram.shape[-1]))

    @property
    def lambda_out(self) -> np.ndarray:
        assert self._lambda_out is not None
        return self._lambda_out

    @property
    def n_bins(self) -> int:
        assert self.iL is not None
        return int(self.iL.size)

    def warmup(self) -> None:
        if self.v_ is None:
            return
        dummy = np.zeros((1, 8, self.v_), dtype=np.uint16)
        _ = _bin_average_uint16_numba(dummy, self.iL, self.iR, self.w)

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        assert self.v_ is not None
        assert spectrogram.shape[-1] == self.v_
        X = np.ascontiguousarray(spectrogram, dtype=np.uint16)
        if self.threads is not None and self.threads > 0:
            set_num_threads(int(self.threads))
        return _bin_average_uint16_numba(X, self.iL, self.iR, self.w)

    def transform_to_cube(self, spectrogram: np.ndarray):
        out = self.transform(spectrogram)
        return out.transpose(2, 1, 0), self.lambda_out
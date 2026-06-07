"""Parallel multi-view PyAV decoder with per-timestep barrier synchronization.

Used by notebooks/n3dv_decode_timing.ipynb. Lives as a real module (not a
notebook cell) so multiprocessing workers can import the worker target under
both the 'fork' and 'spawn' start methods.

Model: a DataLoader-style setup. Each *worker* owns one or more camera streams.
Per timestep `t` the main thread asks every worker to produce frame `t` of its
assigned camera(s); a barrier blocks the bundle until all workers report, then
advances to `t+1`. Workers never run ahead of the barrier, so the measured
per-bundle latency is the true synchronized multi-view cost.

Three phases are timed identically to the sync path:
  T_open   : open all containers + init decoders (workers spun up + ready)
  T_first  : first synchronized multi-view bundle (cold)
  T_steady : median per-bundle latency after T_first + warmup bundles

Demux/decode split: within each bundle the demux step
(`container.demux(stream)`) and the decode step (`packet.decode()`) are timed
separately via DemuxDecodeStream and reported alongside the fused total. For
`sync` the per-bundle demux/decode is the SUM over cameras (work is serialized
in one thread, so the wall-clock contribution is the sum). For
`thread`/`process` it is the MAX over workers (workers run in parallel and the
bundle is gated by the slowest worker, matching the wall-clock semantics of the
fused per-bundle total). `t_steady`/`t_first`/`t_open` remain the fused
demux+decode total and are unchanged.
"""

from __future__ import annotations

import threading
import time
import queue
import multiprocessing as mp

import numpy as np


# --------------------------------------------------------------------------- #
# Shared: assign cameras to workers (round-robin when workers < cameras).
# --------------------------------------------------------------------------- #
def assign(paths, n_workers):
    """Return list[list[(cam_idx, path)]] — one bucket per worker."""
    buckets = [[] for _ in range(n_workers)]
    for i, p in enumerate(paths):
        buckets[i % n_workers].append((i, str(p)))
    return [b for b in buckets if b]  # drop empty buckets if workers > cameras


def _summarize(per_bundle, warmup):
    per_bundle = list(per_bundle)
    if not per_bundle:
        raise RuntimeError("no bundles decoded")
    t_first = per_bundle[0]
    steady = per_bundle[1 + warmup:]
    if steady:
        a = np.asarray(steady)
        return t_first, float(np.median(a)), float(a.mean()), float(a.std())
    return t_first, float(per_bundle[-1]), float(per_bundle[-1]), 0.0


def _summarize_split(per_bundle_vals, warmup):
    """Same windowing as _summarize (skip bundle 0 + `warmup` bundles) but for
    an auxiliary per-bundle array (demux or decode). Returns
    (first, steady_median, steady_std)."""
    vals = list(per_bundle_vals)
    if not vals:
        return 0.0, 0.0, 0.0
    first = vals[0]
    steady = vals[1 + warmup:]
    if steady:
        a = np.asarray(steady)
        return float(first), float(np.median(a)), float(a.std())
    return float(first), float(vals[-1]), 0.0


class DemuxDecodeStream:
    """Wraps one container/stream and yields exactly one frame per call to
    next_frame(), draining as many packets as needed. Demux-call wall time and
    decode-call wall time are accumulated separately so callers can read the
    per-frame split by snapshotting demux_s/decode_s around a bundle.

    A single demux packet can produce 0, 1, or many frames. Extra frames from
    one packet are buffered and served on subsequent calls without touching the
    demuxer (those calls cost ~0 demux, ~0 decode — correct: no libav work
    happened that timestep for that frame).
    """

    __slots__ = ("container", "stream", "_demux", "_pending",
                 "demux_s", "decode_s")

    def __init__(self, container, stream):
        self.container = container
        self.stream = stream
        self._demux = container.demux(stream)
        self._pending = []          # decoded-but-not-yet-served frames
        self.demux_s = 0.0          # cumulative demux wall time (this stream)
        self.decode_s = 0.0         # cumulative decode wall time (this stream)

    def next_frame(self):
        """Return the next decoded video frame. Raises StopIteration at EOF.
        Updates self.demux_s / self.decode_s in place."""
        if self._pending:
            return self._pending.pop(0)
        while True:
            t = time.perf_counter()
            try:
                packet = next(self._demux)
            except StopIteration:
                self.demux_s += time.perf_counter() - t
                raise
            self.demux_s += time.perf_counter() - t

            t = time.perf_counter()
            frames = packet.decode()           # may be empty
            self.decode_s += time.perf_counter() - t

            if frames:
                first = frames[0]
                if len(frames) > 1:
                    self._pending.extend(frames[1:])
                return first
            # 0-frame packet (e.g. header/parameter): keep draining.

    def close(self):
        self.container.close()


# --------------------------------------------------------------------------- #
# Sequential baseline (mode="sync"): one thread, cameras decoded in order.
# --------------------------------------------------------------------------- #
def decode_sync(paths, num_frames, warmup):
    import av

    t0 = time.perf_counter()
    containers, streams = [], []
    for p in paths:
        c = av.open(str(p))
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        containers.append(c)
        streams.append(DemuxDecodeStream(c, vs))
    t_open = time.perf_counter() - t0

    per_bundle, per_demux, per_decode, bundles = [], [], [], 0
    try:
        for _ in range(num_frames):
            b0 = time.perf_counter()
            d0 = sum(s.demux_s for s in streams)
            e0 = sum(s.decode_s for s in streams)
            ok = True
            for s in streams:
                try:
                    fr = s.next_frame()
                except StopIteration:
                    ok = False
                    break
                _ = fr.width
            if not ok:
                break
            per_bundle.append(time.perf_counter() - b0)
            # sync: work serialized in one thread -> sum over cameras.
            per_demux.append(sum(s.demux_s for s in streams) - d0)
            per_decode.append(sum(s.decode_s for s in streams) - e0)
            bundles += 1
    finally:
        for c in containers:
            c.close()

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    dx_first, dx_steady, dx_std = _summarize_split(per_demux, warmup)
    de_first, de_steady, de_std = _summarize_split(per_decode, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle,
                t_demux_first=dx_first, t_demux_steady=dx_steady,
                demux_steady_std=dx_std,
                t_decode_first=de_first, t_decode_steady=de_steady,
                decode_steady_std=de_std,
                per_demux=per_demux, per_decode=per_decode)


# --------------------------------------------------------------------------- #
# Thread workers (mode="thread"): barrier-synced per timestep.
# --------------------------------------------------------------------------- #
def decode_threads(paths, num_frames, warmup, n_workers):
    import av

    buckets = assign(paths, n_workers)
    nW = len(buckets)
    start_evt = [threading.Event() for _ in range(nW)]
    done_evt = [threading.Event() for _ in range(nW)]
    stop = threading.Event()
    errors = []
    # Per-worker per-timestep deltas. Each worker writes only its own index;
    # the main thread reads them only after done_evt[w] (happens-before
    # barrier), so no lock is needed.
    worker_demux_dt = [0.0] * nW
    worker_decode_dt = [0.0] * nW

    def worker(wid, bucket):
        try:
            streams = []
            for _, p in bucket:
                c = av.open(p)
                vs = c.streams.video[0]
                vs.thread_type = "NONE"
                streams.append(DemuxDecodeStream(c, vs))
        except Exception as e:  # noqa: BLE001
            errors.append((wid, repr(e)))
            done_evt[wid].set()
            return
        ready_barrier.wait()
        try:
            while not stop.is_set():
                start_evt[wid].wait()
                start_evt[wid].clear()
                if stop.is_set():
                    break
                d0 = sum(s.demux_s for s in streams)
                e0 = sum(s.decode_s for s in streams)
                for s in streams:
                    fr = s.next_frame()
                    _ = fr.width
                worker_demux_dt[wid] = sum(s.demux_s for s in streams) - d0
                worker_decode_dt[wid] = sum(s.decode_s for s in streams) - e0
                done_evt[wid].set()
        except StopIteration:
            stop.set()
            done_evt[wid].set()
        except Exception as e:  # noqa: BLE001
            errors.append((wid, repr(e)))
            stop.set()
            done_evt[wid].set()
        finally:
            for s in streams:
                s.close()

    ready_barrier = threading.Barrier(nW + 1)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w, b), daemon=True)
               for w, b in enumerate(buckets)]
    for th in threads:
        th.start()
    ready_barrier.wait()  # all workers opened their containers
    t_open = time.perf_counter() - t0
    if errors:
        raise RuntimeError(f"thread worker open failed: {errors}")

    per_bundle, per_demux, per_decode, bundles = [], [], [], 0
    for _ in range(num_frames):
        if stop.is_set():
            break
        b0 = time.perf_counter()
        for w in range(nW):
            done_evt[w].clear()
            start_evt[w].set()
        for w in range(nW):
            done_evt[w].wait()
        if stop.is_set() and not all(done_evt[w].is_set() for w in range(nW)):
            break
        per_bundle.append(time.perf_counter() - b0)
        # parallel: bundle gated by slowest worker -> max over workers.
        per_demux.append(max(worker_demux_dt[w] for w in range(nW)))
        per_decode.append(max(worker_decode_dt[w] for w in range(nW)))
        bundles += 1

    stop.set()
    for w in range(nW):
        start_evt[w].set()
    for th in threads:
        th.join(timeout=5)
    if errors:
        raise RuntimeError(f"thread worker error: {errors}")

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    dx_first, dx_steady, dx_std = _summarize_split(per_demux, warmup)
    de_first, de_steady, de_std = _summarize_split(per_decode, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle,
                t_demux_first=dx_first, t_demux_steady=dx_steady,
                demux_steady_std=dx_std,
                t_decode_first=de_first, t_decode_steady=de_steady,
                decode_steady_std=de_std,
                per_demux=per_demux, per_decode=per_decode,
                n_workers=nW)


# --------------------------------------------------------------------------- #
# Process workers (mode="process"): barrier-synced via per-worker queues.
# --------------------------------------------------------------------------- #
def _proc_worker(bucket, cmd_q, res_q, return_pixels):
    """Runs in a child process. Opens its cameras, then per 'go' decodes one
    frame from each assigned camera and reports back. 'ready' is sent once
    containers are open so the parent can measure T_open."""
    import av

    try:
        stream_recs = []
        for cam_idx, p in bucket:
            c = av.open(p)
            vs = c.streams.video[0]
            vs.thread_type = "NONE"
            stream_recs.append((cam_idx, DemuxDecodeStream(c, vs)))
        res_q.put(("ready", None))
    except Exception as e:  # noqa: BLE001
        res_q.put(("error", repr(e)))
        return

    try:
        while True:
            msg = cmd_q.get()
            if msg == "stop":
                break
            d0 = sum(s.demux_s for _ci, s in stream_recs)
            e0 = sum(s.decode_s for _ci, s in stream_recs)
            payload = []
            for cam_idx, s in stream_recs:
                fr = s.next_frame()
                if return_pixels:
                    payload.append((cam_idx, fr.to_ndarray(format="rgb24")))
                else:
                    payload.append((cam_idx, (fr.height, fr.width)))
            w_demux = sum(s.demux_s for _ci, s in stream_recs) - d0
            w_decode = sum(s.decode_s for _ci, s in stream_recs) - e0
            res_q.put(("frame", (payload, w_demux, w_decode)))
    except StopIteration:
        res_q.put(("eof", None))
    except Exception as e:  # noqa: BLE001
        res_q.put(("error", repr(e)))
    finally:
        for _ci, s in stream_recs:
            s.close()


def decode_processes(paths, num_frames, warmup, n_workers,
                      return_pixels=False, mp_ctx=None):
    ctx = mp_ctx or mp.get_context("fork")
    buckets = assign(paths, n_workers)
    nW = len(buckets)

    cmd_qs = [ctx.Queue() for _ in range(nW)]
    res_qs = [ctx.Queue() for _ in range(nW)]
    procs = []

    t0 = time.perf_counter()
    for w, b in enumerate(buckets):
        pr = ctx.Process(target=_proc_worker,
                         args=(b, cmd_qs[w], res_qs[w], return_pixels),
                         daemon=True)
        pr.start()
        procs.append(pr)

    # Wait for every worker to signal it has opened its containers.
    for w in range(nW):
        tag, info = res_qs[w].get()
        if tag == "error":
            for pr in procs:
                pr.terminate()
            raise RuntimeError(f"process worker open failed: {info}")
        assert tag == "ready", tag
    t_open = time.perf_counter() - t0

    per_bundle, per_demux, per_decode, bundles = [], [], [], 0
    eof = False
    for _ in range(num_frames):
        b0 = time.perf_counter()
        for w in range(nW):
            cmd_qs[w].put("go")
        w_dx = [0.0] * nW
        w_de = [0.0] * nW
        for w in range(nW):
            tag, payload = res_qs[w].get()
            if tag != "frame":
                eof = True
            else:
                _frames, wd, we = payload
                w_dx[w] = wd
                w_de[w] = we
        if eof:
            break
        per_bundle.append(time.perf_counter() - b0)
        # parallel: bundle gated by slowest worker -> max over workers.
        per_demux.append(max(w_dx))
        per_decode.append(max(w_de))
        bundles += 1

    for w in range(nW):
        cmd_qs[w].put("stop")
    for pr in procs:
        pr.join(timeout=5)
        if pr.is_alive():
            pr.terminate()

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    dx_first, dx_steady, dx_std = _summarize_split(per_demux, warmup)
    de_first, de_steady, de_std = _summarize_split(per_decode, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle,
                t_demux_first=dx_first, t_demux_steady=dx_steady,
                demux_steady_std=dx_std,
                t_decode_first=de_first, t_decode_steady=de_steady,
                decode_steady_std=de_std,
                per_demux=per_demux, per_decode=per_decode,
                n_workers=nW)


# --------------------------------------------------------------------------- #
# Unified entry point.
# --------------------------------------------------------------------------- #
def resolve_workers(num_workers, n_cams):
    if num_workers == "auto":
        return n_cams
    return max(1, min(int(num_workers), n_cams))


def run_decode(paths, num_frames, warmup, mode, num_workers="auto",
               return_pixels=False, mp_ctx=None):
    """mode in {sync, thread, process}. Returns the phase-split timing dict
    plus the resolved worker count actually used."""
    n_cams = len(paths)
    if mode == "sync":
        out = decode_sync(paths, num_frames, warmup)
        out["n_workers"] = 1
        return out
    nW = resolve_workers(num_workers, n_cams)
    if mode == "thread":
        return decode_threads(paths, num_frames, warmup, nW)
    if mode == "process":
        return decode_processes(paths, num_frames, warmup, nW,
                                 return_pixels=return_pixels, mp_ctx=mp_ctx)
    raise ValueError(f"unknown mode: {mode}")

"""Parallel multi-view decoder using OpenCV or ffmpeg-python (NO demux/decode
split — these backends fuse demux+decode and expose only a frame iterator).

Sibling of mv_parallel_decode.py (PyAV). Same structure, same per-timestep
barrier synchronization, same three phases (T_open / T_first / T_steady), same
return-dict keys MINUS the demux/decode-split keys, so the benchmark notebook
can drive both modules with identical downstream code.

Backends (select via `backend=`):
  "opencv"  -> cv2.VideoCapture; one cap.read() per frame.
  "ffmpeg"  -> ffmpeg-python: one ffmpeg subprocess per camera streaming
               rawvideo (rgb24) to a pipe; read W*H*3 bytes per frame.

Both fuse container demux + frame decode internally; only the combined
per-bundle wall time is measured. Phases:
  T_open   : open all streams / spawn ffmpeg subprocesses + first-byte readiness
  T_first  : first synchronized multi-view bundle (cold)
  T_steady : median per-bundle latency after T_first + warmup bundles

Lives as a real module so multiprocessing workers can import the worker target
under both 'fork' and 'spawn' start methods.
"""

from __future__ import annotations

import threading
import time
import subprocess
import multiprocessing as mp

import numpy as np


# --------------------------------------------------------------------------- #
# Shared helpers (identical semantics to mv_parallel_decode.py).
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


# --------------------------------------------------------------------------- #
# Per-camera frame sources. Each exposes next_frame() -> object (raises
# StopIteration at EOF) and close(). next_frame() fuses demux+decode.
# --------------------------------------------------------------------------- #
class OpenCVStream:
    """cv2.VideoCapture wrapper. One cap.read() per frame."""

    __slots__ = ("cap",)

    def __init__(self, path):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open {path}")

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def close(self):
        self.cap.release()


class FFmpegStream:
    """ffmpeg-python rawvideo pipe. One ffmpeg subprocess per camera; read
    W*H*3 bytes per frame from stdout."""

    __slots__ = ("proc", "w", "h", "_fsize")

    def __init__(self, path):
        import ffmpeg

        info = ffmpeg.probe(str(path))
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.w, self.h = int(v["width"]), int(v["height"])
        self._fsize = self.w * self.h * 3
        self.proc = (
            ffmpeg
            .input(str(path))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .global_args("-loglevel", "error", "-nostdin")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    def next_frame(self):
        buf = self.proc.stdout.read(self._fsize)
        if buf is None or len(buf) < self._fsize:
            raise StopIteration
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _make_stream(backend, path):
    if backend == "opencv":
        return OpenCVStream(path)
    if backend == "ffmpeg":
        return FFmpegStream(path)
    raise ValueError(f"unknown backend: {backend!r} (use 'opencv' or 'ffmpeg')")


def _touch(frame):
    """Force frame materialization so a backend can't lazily skip work."""
    return frame.shape  # both backends return numpy arrays


# --------------------------------------------------------------------------- #
# Sequential baseline (mode="sync"): one thread, cameras decoded in order.
# --------------------------------------------------------------------------- #
def decode_sync(paths, num_frames, warmup, backend):
    t0 = time.perf_counter()
    streams = [_make_stream(backend, p) for p in paths]
    t_open = time.perf_counter() - t0

    per_bundle, bundles = [], 0
    try:
        for _ in range(num_frames):
            b0 = time.perf_counter()
            ok = True
            for s in streams:
                try:
                    fr = s.next_frame()
                except StopIteration:
                    ok = False
                    break
                _touch(fr)
            if not ok:
                break
            per_bundle.append(time.perf_counter() - b0)
            bundles += 1
    finally:
        for s in streams:
            s.close()

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle)


# --------------------------------------------------------------------------- #
# Thread workers (mode="thread"): barrier-synced per timestep.
# --------------------------------------------------------------------------- #
def decode_threads(paths, num_frames, warmup, n_workers, backend):
    buckets = assign(paths, n_workers)
    nW = len(buckets)
    start_evt = [threading.Event() for _ in range(nW)]
    done_evt = [threading.Event() for _ in range(nW)]
    stop = threading.Event()
    errors = []

    def worker(wid, bucket):
        try:
            streams = [_make_stream(backend, p) for _, p in bucket]
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
                for s in streams:
                    fr = s.next_frame()
                    _touch(fr)
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
    ready_barrier.wait()  # all workers opened their streams
    t_open = time.perf_counter() - t0
    if errors:
        raise RuntimeError(f"thread worker open failed: {errors}")

    per_bundle, bundles = [], 0
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
        bundles += 1

    stop.set()
    for w in range(nW):
        start_evt[w].set()
    for th in threads:
        th.join(timeout=5)
    if errors:
        raise RuntimeError(f"thread worker error: {errors}")

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle,
                n_workers=nW)


# --------------------------------------------------------------------------- #
# Process workers (mode="process"): barrier-synced via per-worker queues.
# --------------------------------------------------------------------------- #
def _proc_worker(bucket, cmd_q, res_q, return_pixels, backend):
    try:
        stream_recs = []
        for cam_idx, p in bucket:
            stream_recs.append((cam_idx, _make_stream(backend, p)))
        res_q.put(("ready", None))
    except Exception as e:  # noqa: BLE001
        res_q.put(("error", repr(e)))
        return

    try:
        while True:
            msg = cmd_q.get()
            if msg == "stop":
                break
            payload = []
            for cam_idx, s in stream_recs:
                fr = s.next_frame()
                if return_pixels:
                    payload.append((cam_idx, np.ascontiguousarray(fr)))
                else:
                    payload.append((cam_idx, fr.shape))
            res_q.put(("frame", payload))
    except StopIteration:
        res_q.put(("eof", None))
    except Exception as e:  # noqa: BLE001
        res_q.put(("error", repr(e)))
    finally:
        for _ci, s in stream_recs:
            s.close()


def decode_processes(paths, num_frames, warmup, n_workers, backend,
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
                         args=(b, cmd_qs[w], res_qs[w], return_pixels, backend),
                         daemon=True)
        pr.start()
        procs.append(pr)

    for w in range(nW):
        tag, info = res_qs[w].get()
        if tag == "error":
            for pr in procs:
                pr.terminate()
            raise RuntimeError(f"process worker open failed: {info}")
        assert tag == "ready", tag
    t_open = time.perf_counter() - t0

    per_bundle, bundles = [], 0
    eof = False
    for _ in range(num_frames):
        b0 = time.perf_counter()
        for w in range(nW):
            cmd_qs[w].put("go")
        for w in range(nW):
            tag, payload = res_qs[w].get()
            if tag != "frame":
                eof = True
        if eof:
            break
        per_bundle.append(time.perf_counter() - b0)
        bundles += 1

    for w in range(nW):
        cmd_qs[w].put("stop")
    for pr in procs:
        pr.join(timeout=5)
        if pr.is_alive():
            pr.terminate()

    t_first, t_steady, s_mean, s_std = _summarize(per_bundle, warmup)
    return dict(t_open=t_open, t_first=t_first, t_steady=t_steady,
                steady_mean=s_mean, steady_std=s_std,
                bundles=bundles, n_cams=len(paths), per_bundle=per_bundle,
                n_workers=nW)


# --------------------------------------------------------------------------- #
# Unified entry point.
# --------------------------------------------------------------------------- #
def resolve_workers(num_workers, n_cams):
    if num_workers == "auto":
        return n_cams
    return max(1, min(int(num_workers), n_cams))


def run_decode(paths, num_frames, warmup, mode, backend="opencv",
               num_workers="auto", return_pixels=False, mp_ctx=None):
    """mode in {sync, thread, process}; backend in {opencv, ffmpeg}.
    Returns the phase-split timing dict + resolved worker count."""
    n_cams = len(paths)
    if mode == "sync":
        out = decode_sync(paths, num_frames, warmup, backend)
        out["n_workers"] = 1
        return out
    nW = resolve_workers(num_workers, n_cams)
    if mode == "thread":
        return decode_threads(paths, num_frames, warmup, nW, backend)
    if mode == "process":
        return decode_processes(paths, num_frames, warmup, nW, backend,
                                return_pixels=return_pixels, mp_ctx=mp_ctx)
    raise ValueError(f"unknown mode: {mode}")

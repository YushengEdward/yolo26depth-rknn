#!/usr/bin/env python3
"""Pipeline benchmark: decouple pre/infer/post into separate stages."""
from __future__ import annotations
import argparse, itertools, os, sys, threading, time
from collections import OrderedDict
from queue import Queue
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or '.')
from yolo26depth_rknn.utils import prepare_input_rect
from rknnlite.api.rknn_lite import RKNNLite

_tls = threading.local()
_counter = itertools.count()

def preprocess(image: np.ndarray, imgsz: tuple):
    return prepare_input_rect(image, imgsz, normalize=False)

def postprocess(outputs: list, orig_size: tuple):
    depth = np.squeeze(outputs[0]).astype(np.float32)
    depth = cv2.resize(depth, (orig_size[1], orig_size[0]),
                       interpolation=cv2.INTER_LINEAR)
    return depth

class InferenceWorker:
    """Dedicated inference worker: takes preprocessed input, returns raw output."""
    def __init__(self, model_path: str, core: str, imgsz: tuple):
        self.core = core
        self.imgsz = imgsz
        core_map = {"0": RKNNLite.NPU_CORE_0,
                    "1": RKNNLite.NPU_CORE_1,
                    "2": RKNNLite.NPU_CORE_2,
                    "auto": RKNNLite.NPU_CORE_AUTO}
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(core_mask=core_map[core])

    def infer(self, inp: np.ndarray) -> list:
        return self.rknn.inference(inputs=[inp])

    def release(self):
        self.rknn.release()

def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--cores", nargs="+", default=["0", "1", "2"],
                        choices=["auto", "0", "1", "2"])
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    orig_h, orig_w = img.shape[:2]

    # Load model once for metadata
    rknn = RKNNLite()
    rknn.load_rknn(args.model)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    imgsz = (rknn.rknn_runtime.get_inputs()[0].dims[1],
             rknn.rknn_runtime.get_inputs()[0].dims[2])
    rknn.release()

    # Preprocess all inputs upfront
    print(f"Preprocessing {args.frames} frames...", file=sys.stderr)
    t0 = time.perf_counter()
    preproc_inputs = [preprocess(img, imgsz) for _ in range(args.frames)]
    t1 = time.perf_counter()
    pre_time = (t1 - t0) / args.frames * 1000
    print(f"  Avg preprocess: {pre_time:.2f} ms", file=sys.stderr)

    n_workers = len(args.cores)
    workers = [InferenceWorker(args.model, c, imgsz) for c in args.cores]

    # Warmup
    print(f"Warmup ({args.warmup} per worker)...", file=sys.stderr)
    for _ in range(args.warmup):
        for w in workers:
            w.infer(preproc_inputs[0])

    # Count-based dispatch (fair round-robin)
    dispatch_counter = itertools.count()
    results = [None] * args.frames
    lock = threading.Lock()
    done_event = threading.Event()
    latencies = []

    def worker_fn(w: InferenceWorker):
        while True:
            idx = next(dispatch_counter)
            if idx >= args.frames:
                break
            t0 = time.perf_counter()
            out = w.infer(preproc_inputs[idx])
            t1 = time.perf_counter()
            with lock:
                results[idx] = out
                latencies.append((t1 - t0) * 1000)
        done_event.set()

    threads = []
    t_start = time.perf_counter()
    for w in workers:
        t = threading.Thread(target=worker_fn, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_wall = time.perf_counter() - t_start

    # Postprocess all
    t0 = time.perf_counter()
    for out in results:
        postprocess(out, (orig_h, orig_w))
    t1 = time.perf_counter()
    post_time = (t1 - t0) / args.frames * 1000

    for w in workers:
        w.release()

    latencies.sort()
    median = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    print(f"\nPipeline benchmark: {args.frames} frames, {n_workers} workers on {args.cores}")
    print(f"  Preprocess:  {pre_time:.1f} ms avg")
    print(f"  Postprocess: {post_time:.1f} ms avg")
    print(f"  Wall time:   {t_wall:.2f} s")
    print(f"  Throughput:  {args.frames/t_wall:.2f} FPS (aggregate)")
    print(f"  NPU latency: median {median:.1f} ms, p95 {p95:.1f} ms")

if __name__ == "__main__":
    main()

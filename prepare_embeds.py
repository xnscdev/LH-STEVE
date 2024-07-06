import argparse
import random
import os
import re
import math
import bisect
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm, trange
from itertools import islice
from contextlib import suppress
from collections import defaultdict, deque

from lh_steve.models import load_mineclip


def get_runs(subsets, data_dir):
    pattern = re.compile(r'^(.*-[0-9a-f]{12})-.*\.mp4$')
    runs = defaultdict(list)
    for subset in subsets:
        for fname in os.listdir(f'{data_dir}/{subset}'):
            match = pattern.match(fname)
            if match:
                session = match.group(1)
                bisect.insort(runs[session], f'{subset}/{fname}')
    return runs.items()


def batch_runs(runs, batches):
    batch_size = math.ceil(len(runs) / batches)
    it = iter(runs)
    while True:
        batch = tuple(islice(it, batch_size))
        if not batch:
            break
        yield batch


def gen_frames(data_dir, segments, lut):
    for fname in segments:
        video = cv2.VideoCapture(f'{data_dir}/{fname}')
        if not video.isOpened():
            tqdm.write(f'Error opening {fname}')
            continue
        while True:
            ret, frame = video.read()
            if not ret:
                video.release()
                break
            frame = cv2.resize(frame, (160, 256), cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))
            frame = cv2.LUT(frame, lut)
            yield frame


def gen_clips(data_dir, segments, lut, n_frames=16):
    frames = deque(maxlen=n_frames)
    it = gen_frames(data_dir, segments, lut)
    for _ in range(n_frames):
        frames.append(next(it))
    i = 0
    with suppress(StopIteration):
        while True:
            yield torch.from_numpy(np.array(frames, dtype=np.float32))
            frames.append(next(it))
            if i > 100:
                break
            i += 1


@torch.no_grad()
def embed_runs(mineclip, runs, args, queue):
    device = torch.device(args.device)
    lut = np.array([(i / 255.0) ** args.gamma for i in range(256)], dtype=np.float32)
    for session, segments in runs:
        tqdm.write(session)
        it = gen_clips(args.data_dir, segments, lut)
        embeddings = []
        while True:
            k = random.randint(args.k_min, args.k_max)
            clips = tuple(islice(it, k))
            if not clips:
                break
            clips = torch.stack(clips, dim=0).to(device).split(args.batch_size)
            clips = torch.cat(tuple(mineclip.encode_video(c) for c in clips), dim=0)
            embeddings.append(clips)
        torch.save(embeddings, f'{args.output_dir}/{session}.pt')
        tqdm.write(f'Saved {len(embeddings)} embeddings to {args.output_dir}/{session}.pt')
        queue.put(None)


def start_processes(args):
    device = torch.device(args.device)
    mineclip = load_mineclip(args.ckpt_path, device)
    mineclip.share_memory()

    subsets = list(filter(lambda x: os.path.isdir(f'{args.data_dir}/{x}'), os.listdir(args.data_dir)))
    for subset in subsets:
        os.makedirs(f'{args.output_dir}/{subset}', exist_ok=True)
    runs = get_runs(subsets, args.data_dir)
    total_runs = len(runs)
    runs = list(batch_runs(runs, args.n_processes))
    n_processes = len(runs)

    processes = []
    queue = mp.Queue()
    for i in range(n_processes):
        p = mp.Process(target=embed_runs, args=(mineclip, runs[i], args, queue))
        p.start()
        processes.append(p)
    for i in trange(total_runs, unit='run'):
        queue.get()
    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-c', '--ckpt-path', type=str, required=True, help='MineCLIP checkpoint path')
    parser.add_argument('-D', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-p', '--n-processes', type=int, default=4)
    parser.add_argument('--k-min', type=int, default=40, help='Minimum number of frames for goal')
    parser.add_argument('--k-max', type=int, default=200, help='Maximum number of frames for goal')
    parser.add_argument('-g', '--gamma', type=float, default=0.5, help='Gamma correction value')
    args = parser.parse_args()
    start_processes(args)


if __name__ == '__main__':
    main()

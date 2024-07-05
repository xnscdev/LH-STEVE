import argparse
import random
import os
import re
import bisect
import cv2
import numpy as np
import torch
from tqdm import tqdm
from mineclip import MineCLIP
from itertools import islice
from collections import defaultdict, deque
from contextlib import suppress


def load_mineclip(ckpt_path, device):
    config = {
        'arch': 'vit_base_p16_fz.v2.t2',
        'hidden_dim': 512,
        'image_feature_dim': 512, 
        'mlp_adapter_spec': 'v0-2.t0',
        'pool_type': 'attn.d2.nh8.glusw',
        'resolution': [160, 256]
    }
    model = MineCLIP(**config).to(device)
    model.load_ckpt(ckpt_path, strict=True)
    model.eval()
    print('Loaded MineCLIP')
    return model


def get_runs(subsets, data_dir):
    pattern = re.compile(r'^(.*-[0-9a-f]{12})-.*\.mp4$')
    runs = {}
    for subset in subsets:
        sessions = defaultdict(list)
        for fname in os.listdir(f'{data_dir}/{subset}'):
            match = pattern.match(fname)
            if match:
                session = match.group(1)
                bisect.insort(sessions[session], fname)
        runs[subset] = list(sessions.items())
    return runs


def gen_clips(subset, run, data_dir, lut, n_frames=16):
    frames = deque(maxlen=n_frames)
    it = gen_frames(subset, run, data_dir, lut)
    for _ in range(n_frames):
        frames.append(next(it))
    with suppress(StopIteration):
        while True:
            yield torch.from_numpy(np.array(frames, dtype=np.float32))
            frames.append(next(it))


def gen_frames(subset, run, data_dir, lut):
    for fname in run:
        video = cv2.VideoCapture(f'{data_dir}/{subset}/{fname}')
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


@torch.no_grad()
def embed_run(subset, session, run, device, mineclip, lut, data_dir, output_dir, batch_size, k_min, k_max):
    it = gen_clips(subset, run, data_dir, lut)
    embeddings = []
    while True:
        k = random.randint(k_min, k_max)
        clips = tuple(islice(it, k))
        if not clips:
            break
        clips = torch.stack(clips, dim=0).to(device).split(batch_size)
        clips = torch.cat(tuple(mineclip.encode_video(c) for c in clips), dim=0)
        embeddings.append(clips)
    torch.save(embeddings, f'{output_dir}/{subset}/{session}.pt')
    tqdm.write(f'Saved {len(embeddings)} embeddings to {output_dir}/{subset}/{session}.pt')


def main(args):
    device = torch.device(args.device)
    mineclip = load_mineclip(args.ckpt_path, device)
    lut = np.array([(i / 255.0) ** args.gamma for i in range(256)], dtype=np.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    subsets = list(filter(lambda x: os.path.isdir(f'{args.data_dir}/{x}'), os.listdir(args.data_dir)))
    runs = get_runs(subsets, args.data_dir)
    for subset in subsets:
        os.makedirs(f'{args.output_dir}/{subset}', exist_ok=True)
        for session, run in tqdm(runs[subset], unit='run', desc=subset):
            embed_run(subset, session, run, device, mineclip, lut, args.data_dir, args.output_dir, args.batch_size, args.k_min, args.k_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-c', '--ckpt-path', type=str, required=True, help='MineCLIP checkpoint path')
    parser.add_argument('-D', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for running MineCLIP')
    parser.add_argument('--k-min', type=int, default=40, help='Minimum number of frames for goal')
    parser.add_argument('--k-max', type=int, default=200, help='Maximum number of frames for goal')
    parser.add_argument('-g', '--gamma', type=float, default=0.5, help='Gamma correction value')
    args = parser.parse_args()
    main(args)

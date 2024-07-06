import argparse
import random
import os
import re
import math
import bisect
import cv2
import numpy as np
import lightning as L
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from lh_steve.datasets import VideoClipDataset, get_subsets
from lh_steve.models import MineCLIPWrapper


def get_runs(subsets, data_dir):
    pattern = re.compile(r'^(.*-[0-9a-f]{12})-.*\.mp4$')
    runs = defaultdict(list)
    for subset in subsets:
        for fname in os.listdir(f'{data_dir}/{subset}'):
            match = pattern.match(fname)
            if match:
                session = f'{subset}/{match.group(1)}'
                bisect.insort(runs[session], f'{subset}/{fname}')
    return runs


@torch.no_grad()
def embed_runs(args):
    mineclip = MineCLIPWrapper(args.ckpt_path)
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    subsets = get_subsets(args.data_dir)
    for subset in subsets:
        os.makedirs(f'{args.output_dir}/{subset}', exist_ok=True)
    runs = get_runs(subsets, args.data_dir)
    print(f'Embedding {len(runs)} runs')
    for i, (session, segments) in enumerate(runs.items(), 1):
        dataset = VideoClipDataset(args.data_dir, segments, gamma=args.gamma)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        predictions = trainer.predict(mineclip, dataloader)
        predictions = torch.cat(predictions, dim=0)
        torch.save(predictions, f'{args.output_dir}/{session}.pt')
        print(f'Saved {predictions.size(0)} embeddings to {session}.pt ({i}/{len(runs)})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-c', '--ckpt-path', type=str, required=True, help='MineCLIP checkpoint path')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-g', '--gamma', type=float, default=0.5, help='Gamma correction value')
    args = parser.parse_args()
    embed_runs(args)


if __name__ == '__main__':
    main()

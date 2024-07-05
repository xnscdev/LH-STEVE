import argparse
import random
import os
import torch
from tqdm import tqdm
from mineclip import MineCLIP
from itertools import islice

from lh_steve.datasets import VideoClipManager


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


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    mineclip = load_mineclip(args.ckpt_path, device)
    clip_manager = VideoClipManager(args.data_dir, resolution=(160, 256), gamma=args.gamma)
    os.makedirs(args.output_dir, exist_ok=True)
    for subset in clip_manager.get_subsets():
        print(f'Processing {subset}')
        os.makedirs(f'{args.output_dir}/{subset}', exist_ok=True)
        for session, run in tqdm(clip_manager.runs[subset]):
            with tqdm(leave=False) as pbar:
                it = clip_manager.gen_clips(subset, run)
                embeddings = []
                while True:
                    k = random.randint(args.k_min, args.k_max)
                    clips = tuple(islice(it, k))
                    if not clips:
                        break
                    clips = torch.stack(clips, dim=0).to(device).split(args.batch_size)
                    clips = torch.cat(tuple(mineclip.encode_video(c) for c in clips), dim=0)
                    embeddings.append(clips)
                    pbar.update(clips.size(0))
                torch.save(embeddings, f'{args.output_dir}/{subset}/{session}.pt')


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

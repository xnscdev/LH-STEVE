import argparse
import urllib.request
from urllib.error import HTTPError
from contextlib import suppress
import os
import json
from tqdm import tqdm

from lh_steve.config import vpt_dataset_metadata


class DownloadProgressBar(tqdm):
    def update_to(self, n_blocks, block_size, total_size):
        if total_size is not None:
            self.total = total_size
        self.update(n_blocks * block_size - self.n)


def download_file(url, dir, skip_if_exist=True, suffix=''):
    name = url.split('/')[-1]
    if suffix:
        suffix = ' ' + suffix
    print(f'Downloading {name}{suffix}')
    path = f'{dir}/{name}'
    if skip_if_exist and os.path.isfile(path):
        print(f'Skipping download of existing file: {path}')
        return path
    with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as pbar:
        name, _ = urllib.request.urlretrieve(url, path, reporthook=pbar.update_to)
        pbar.total = pbar.n
        return path


def main(args):
    for subset in args.subsets:
        try:
            subset_info = vpt_dataset_metadata[subset]
        except KeyError as e:
            raise ValueError(f'Invalid subset "{subset}"') from e

    os.makedirs(args.output_dir, exist_ok=True)
    for subset in args.subsets:
        subset_info = vpt_dataset_metadata[subset]
        subset_dir = f'{args.output_dir}/{subset}'
        os.makedirs(subset_dir, exist_ok=True)
        
        idx_file = download_file(subset_info['index_url'], args.output_dir)
        with open(idx_file, 'r') as f:
            idx = json.load(f)
            n_samples = args.n_samples
            max_len = len(idx['relpaths'])
            if n_samples is None:
                n_samples = max_len
            n_samples = min(n_samples, max_len)
            downloaded = 0
            i = 0
            while downloaded < n_samples:
                mp4_url = idx['basedir'] + idx['relpaths'][i]
                jsonl_url = mp4_url.replace('.mp4', '.jsonl')
                mp4_fname = None
                jsonl_fname = None
                try:
                    mp4_fname = download_file(mp4_url, subset_dir, suffix=f'({downloaded + 1}/{n_samples})')
                    jsonl_fname = download_file(jsonl_url, subset_dir, suffix=f'({downloaded + 1}/{n_samples})')
                except HTTPError:
                    with suppress(FileNotFoundError):
                        if mp4_fname:
                            os.remove(mp4_fname)
                        if jsonl_fname:
                            os.remove(jsonl_fname)
                    print('Download failed')
                else:
                    downloaded += 1
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-s', '--subsets', type=str, nargs='+', required=True, help='Which subsets of data to download')
    parser.add_argument('-n', '--n-samples', type=int, help='Number of samples to download per subset (defaults to all)')
    args = parser.parse_args()
    main(args)
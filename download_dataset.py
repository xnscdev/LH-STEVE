import argparse
import urllib.request
from urllib.error import HTTPError
from contextlib import suppress
import os
import json
from concurrent.futures import ThreadPoolExecutor

from lh_steve.config import vpt_dataset_metadata


def download_file(url, dir, skip_if_exist=True):
    name = url.split('/')[-1]
    path = f'{dir}/{name}'
    if skip_if_exist and os.path.isfile(path):
        print(f'Skipping download of existing file: {path}')
        return path
    name, _ = urllib.request.urlretrieve(url, path)
    return name


def download_pair(url_base, dir, i, total):
    name = url_base.split('/')[-1]
    print(f'Downloading {name} ({i + 1}/{total})')
    try:
        download_file(f'{url_base}.mp4', dir)
        download_file(f'{url_base}.jsonl', dir)
    except HTTPError:
        with suppress(FileNotFoundError):
            os.remove(f'{dir}/{name}.mp4')
            os.remove(f'{dir}/{name}.jsonl')
        print(f'Failed to download {name} ({i + 1}/{total})')
    else:
        print(f'Downloaded {name} ({i + 1}/{total})')


def main(args):
    for subset in args.subsets:
        if subset not in vpt_dataset_metadata:
            raise ValueError(f'Invalid subset "{subset}"')

    os.makedirs(args.output_dir, exist_ok=True)
    for subset in args.subsets:
        subset_info = vpt_dataset_metadata[subset]
        subset_dir = f'{args.output_dir}/{subset}'
        os.makedirs(subset_dir, exist_ok=True)

        print(f'Downloading index file for subset {subset}')
        idx_file = download_file(subset_info['index_url'], args.output_dir)
        with open(idx_file, 'r') as f:
            idx = json.load(f)
            n_samples = args.n_samples
            max_len = len(idx['relpaths'])
            if n_samples is None:
                n_samples = max_len
            n_samples = min(n_samples, max_len)
            with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                for i in range(n_samples):
                    url_base = idx['basedir'] + idx['relpaths'][i][:-4]  # trim '.mp4'
                    executor.submit(download_pair, url_base, subset_dir, i, n_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-s', '--subsets', type=str, nargs='+', required=True, help='Which subsets of data to download (see lh_steve/config.py)')
    parser.add_argument('-n', '--n-samples', type=int, help='Number of samples to download per subset (defaults to all)')
    parser.add_argument('-w', '--n-workers', type=int, default=4)
    args = parser.parse_args()
    main(args)

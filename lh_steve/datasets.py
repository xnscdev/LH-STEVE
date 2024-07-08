import os
import re
import sys
import cv2
import math
import bisect
import numpy as np
import lightning as L
import torch
from torch.utils.data import IterableDataset, DataLoader
from collections import defaultdict, deque
from contextlib import suppress


class VideoClipGoalDataset(IterableDataset):
    def __init__(self, data_dir, runs, resolution=(160, 256), n_frames=16, gamma=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.runs = runs
        h, w = resolution
        self.resolution = (w, h)
        self.n_frames = n_frames
        self.lut = np.array([(i / 255.0) ** gamma for i in range(256)], dtype=np.float32)
        self.length = sum(l for _, l in runs) - n_frames + 1

    def __iter__(self):
        frames = deque(maxlen=self.n_frames)
        it = self.gen_frames()
        for _ in range(self.n_frames):
            frames.append(next(it))
        with suppress(StopIteration):
            while True:
                yield torch.from_numpy(np.array(frames, dtype=np.float32))
                frames.append(next(it))

    def __len__(self):
        return self.length

    def gen_frames(self):
        for segments, length in self.runs:
            videos = []
            for segment in segments:
                video = cv2.VideoCapture(f'{self.data_dir}/{segment}')
                if not video.isOpened():
                    print(f'Error opening {self.data_dir}/{segment}', file=sys.stderr)
                    video.release()
                else:
                    videos.append(video)
            for video in videos:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, self.resolution, cv2.INTER_NEAREST)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.transpose(frame, (2, 0, 1))
                    frame = cv2.LUT(frame, self.lut)
                    yield frame
            for video in videos:
                video.release()


class VideoClipGoalDataModule(L.LightningDataModule):
    def __init__(self, data_dir, split=0.1, min_video_len=32, batch_size=64, n_workers=4, gamma=0.5):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        if stage == 'fit':
            pattern = re.compile(r'^(.*-[0-9a-f]{12})-.*\.mp4$')
            runs = defaultdict(list)
            for subset in get_subsets(self.hparams.data_dir):
                for fname in os.listdir(f'{self.hparams.data_dir}/{subset}'):
                    match = pattern.match(fname)
                    if match:
                        session = match.group(1)
                        bisect.insort(runs[session], f'{subset}/{fname}')
            runs_list = []
            for segments in runs.values():
                new_segments = []
                length = 0
                for segment in segments:
                    video = cv2.VideoCapture(f'{self.hparams.data_dir}/{segment}')
                    if video.isOpened():
                        video_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
                        if video_len >= self.hparams.min_video_len:
                            new_segments.append(segment)
                            length += video_len
                        else:
                            print(
                                f'Skipping video {self.hparams.data_dir}/{segment} as it has less than {self.hparams.min_video_len} frames',
                                file=sys.stderr)
                    else:
                        print(f'Error opening {self.hparams.data_dir}/{segment}', file=sys.stderr)
                    video.release()
                if new_segments:
                    runs_list.append((new_segments, int(length)))
            runs_list.sort(key=lambda x: x[1], reverse=True)
            n_test_runs = math.ceil(self.hparams.split * len(runs_list))
            train_runs = self.split_runs(runs_list[n_test_runs:])
            test_runs = self.split_runs(runs_list[:n_test_runs])
            self.train_dataset = VideoClipGoalDataset(self.hparams.data_dir, train_runs[self.trainer.global_rank],
                                                      gamma=self.hparams.gamma)
            self.test_dataset = VideoClipGoalDataset(self.hparams.data_dir, test_runs[self.trainer.global_rank],
                                                     gamma=self.hparams.gamma)
            print(f'Rank {self.trainer.global_rank}: {train_runs}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_workers,
                          pin_memory=True)

    def split_runs(self, runs):
        splits = [[] for _ in range(self.trainer.world_size)]
        sizes = [0 for _ in range(self.trainer.world_size)]
        for run in runs:
            _, length = run
            i = min(range(self.trainer.world_size), key=lambda i: sizes[i])
            splits[i].append(run)
            sizes[i] += length
        print(sizes)
        return splits


def get_subsets(data_dir):
    return list(filter(lambda x: os.path.isdir(f'{data_dir}/{x}'), os.listdir(data_dir)))

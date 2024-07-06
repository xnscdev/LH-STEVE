import os
import sys
import cv2
import numpy as np
import lightning as L
import torch
from torch.utils.data import IterableDataset, DataLoader
from collections import deque
from contextlib import suppress


class VideoClipDataset(IterableDataset):
    def __init__(self, data_dir, segments, resolution=(160, 256), n_frames=16, gamma=0.5):
        super().__init__()
        self.data_dir = data_dir
        h, w = resolution
        self.resolution = (w, h)
        self.n_frames = n_frames
        self.lut = np.array([(i / 255.0) ** gamma for i in range(256)], dtype=np.float32)
        self.videos = []
        for fname in segments:
            video = cv2.VideoCapture(f'{self.data_dir}/{fname}')
            if not video.isOpened():
                print(f'Error opening {fname}', file=sys.stderr)
            else:
                self.videos.append(video)
        self.length = sum(video.get(cv2.CAP_PROP_FRAME_COUNT) for video in self.videos)

    def __del__(self):
        for video in self.videos:
            video.release()

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
        for video in self.videos:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.resize(frame, self.resolution, cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.transpose(frame, (2, 0, 1))
                frame = cv2.LUT(frame, self.lut)
                yield frame


def get_subsets(data_dir):
    return list(filter(lambda x: os.path.isdir(f'{data_dir}/{x}'), os.listdir(data_dir)))

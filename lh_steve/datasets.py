import os
import re
import bisect
import sys
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from contextlib import suppress


class VideoClipManager:
    def __init__(self, data_dir, resolution=(256, 256), n_frames=16, gamma=0.5):
        super().__init__()
        self.data_dir = data_dir
        h, w = resolution
        self.resolution = (w, h)
        self.n_frames = n_frames

        self.runs = {}
        pattern = re.compile(r'^(.*-[0-9a-f]{12})-.*\.mp4$')
        for subset in self.get_subsets():
            sessions = defaultdict(list)
            for fname in os.listdir(f'{data_dir}/{subset}'):
                match = pattern.match(fname)
                if match:
                    session = match.group(1)
                    bisect.insort(sessions[session], fname)
            self.runs[subset] = list(sessions.items())
        self.lut = np.array([(i / 255.0) ** gamma for i in range(256)], dtype=np.float32)

    def get_subsets(self):
        return filter(lambda x: os.path.isdir(f'{self.data_dir}/{x}'), os.listdir(self.data_dir))

    def gen_clips(self, subset, run):
        frames = deque(maxlen=self.n_frames)
        it = self.gen_frames(subset, run)
        for _ in range(self.n_frames):
            frames.append(next(it))
        with suppress(StopIteration):
            while True:
                yield torch.from_numpy(np.array(frames, dtype=np.float32))
                frames.append(next(it))

    def gen_frames(self, subset, run):
        for fname in run:
            video = cv2.VideoCapture(f'{self.data_dir}/{subset}/{fname}')
            if not video.isOpened():
                print(f'Error opening {fname}', file=sys.stderr)
                continue
            while True:
                ret, frame = video.read()
                if not ret:
                    video.release()
                    break
                frame = cv2.resize(frame, self.resolution, cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.transpose(frame, (2, 0, 1))
                frame = cv2.LUT(frame, self.lut)
                yield frame

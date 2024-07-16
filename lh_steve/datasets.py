import os
import re
import sys
import cv2
import math
import random
import bisect
import numpy as np
import lightning as L
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from collections import defaultdict, deque
from contextlib import suppress


class VideoClipGoalDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        runs,
        n_workers,
        resolution=(160, 256),
        goal_bound=128,
        n_frames=16,
        gamma=0.5,
    ):
        super().__init__()
        self.data_dir = data_dir
        h, w = resolution
        self.resolution = (w, h)
        self.goal_bound = goal_bound
        self.n_frames = n_frames
        self.lut = np.array(
            [(i / 255.0) ** gamma for i in range(256)], dtype=np.float32
        )
        if n_workers == 0:
            n_workers = 1
        self.runs = split_runs(runs, n_workers)
        self.length = sum(l - self.n_frames + 1 for _, l in runs)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            idx = 0
        else:
            idx = worker_info.id
        for segments, _ in self.runs[idx]:
            frames = deque(maxlen=self.n_frames)
            final_clip = self.get_final_clip(segments)
            it = self.gen_frames(segments)
            for _ in range(self.n_frames):
                frames.append(next(it))
            with suppress(StopIteration):
                while True:
                    yield np.array(frames, dtype=np.float32), final_clip
                    frames.append(next(it))

    def __len__(self):
        return self.length

    def gen_frames(self, segments):
        videos = []
        for segment in segments:
            video = cv2.VideoCapture(f"{self.data_dir}/{segment}")
            if not video.isOpened():
                print(f"Error opening {self.data_dir}/{segment}", file=sys.stderr)
                video.release()
            else:
                videos.append(video)
        for video in videos:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                yield self.process_frame(frame)
        for video in videos:
            video.release()

    def process_frame(self, frame):
        frame = cv2.resize(frame, self.resolution, cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        frame = cv2.LUT(frame, self.lut)
        return frame

    def get_final_clip(self, segments):
        video = cv2.VideoCapture(f"{self.data_dir}/{segments[-1]}")
        assert video.isOpened(), f"Failed to open {self.data_dir}/{segments[-1]}"
        pos = video.get(cv2.CAP_PROP_FRAME_COUNT) - random.randint(
            self.n_frames, self.goal_bound
        )
        video.set(cv2.CAP_PROP_POS_FRAMES, pos)
        frames = []
        for _ in range(self.n_frames):
            ret, frame = video.read()
            assert ret
            frames.append(self.process_frame(frame))
        video.release()
        return np.array(frames, dtype=np.float32)


class VideoClipGoalDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        n_workers,
        split=0.1,
        goal_bound=128,
        min_video_len=128,
        gamma=0.5,
    ):
        super().__init__()
        assert (
            goal_bound <= min_video_len
        ), "Goal bound must be less than or equal to minimum video length"
        self.save_hyperparameters()
        self.train_runs = None
        self.test_runs = None

    def setup(self, stage):
        if self.train_runs is not None and self.test_runs is not None:
            return
        self.train_runs, self.test_runs = build_runs_list(
            self.hparams.data_dir,
            self.trainer.world_size,
            min_video_len=self.hparams.min_video_len,
            split=self.hparams.split,
        )

        if stage == "fit":
            self.train_dataset = VideoClipGoalDataset(
                self.hparams.data_dir,
                self.train_runs[self.trainer.global_rank],
                self.hparams.n_workers,
                goal_bound=self.hparams.goal_bound,
                gamma=self.hparams.gamma,
            )
        elif stage == "test":
            self.test_dataset = VideoClipGoalDataset(
                self.hparams.data_dir,
                self.test_runs[self.trainer.global_rank],
                self.hparams.n_workers,
                goal_bound=self.hparams.goal_bound,
                gamma=self.hparams.gamma,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            pin_memory=True,
        )


def get_subsets(data_dir):
    return list(
        filter(lambda x: os.path.isdir(f"{data_dir}/{x}"), os.listdir(data_dir))
    )


def split_runs(runs, n_splits):
    splits = [[] for _ in range(n_splits)]
    sizes = [0 for _ in range(n_splits)]
    for run in runs:
        _, length = run
        i = min(range(n_splits), key=lambda i: sizes[i])
        splits[i].append(run)
        sizes[i] += length
    return splits


def build_runs_list(data_dir, world_size, min_video_len=128, split=0.1):
    pattern = re.compile(r"^(.*-[0-9a-f]{12})-.*\.mp4$")
    runs = defaultdict(list)
    for subset in get_subsets(data_dir):
        for fname in os.listdir(f"{data_dir}/{subset}"):
            match = pattern.match(fname)
            if match:
                session = match.group(1)
                bisect.insort(runs[session], f"{subset}/{fname}")
    runs_list = []
    for segments in runs.values():
        new_segments = []
        length = 0
        for segment in segments:
            video = cv2.VideoCapture(f"{data_dir}/{segment}")
            if video.isOpened():
                video_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
                if video_len >= min_video_len:
                    new_segments.append(segment)
                    length += video_len
                else:
                    print(
                        f"Skipping video {data_dir}/{segment} as it has less than {min_video_len} frames",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"Error opening {data_dir}/{segment}",
                    file=sys.stderr,
                )
            video.release()
        if new_segments:
            runs_list.append((new_segments, int(length)))
    runs_list.sort(key=lambda x: x[1], reverse=True)
    n_test_runs = math.ceil(split * len(runs_list))
    train_runs = split_runs(runs_list[n_test_runs:], world_size)
    test_runs = split_runs(runs_list[:n_test_runs], world_size)
    return train_runs, test_runs

import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lh_steve.models import ShortTermGoalCVAE
from lh_steve.datasets import VideoClipGoalDataModule


def train(args):
    datamodule = VideoClipGoalDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        split=args.split,
        goal_bound=args.goal_bound,
        min_video_len=args.min_video_len,
        gamma=args.gamma,
    )
    logger = TensorBoardLogger("train_logs", name="goal_cvae")
    callbacks = None
    if args.ckpt_dir is not None:
        checkpoints = ModelCheckpoint(
            args.ckpt_dir, every_n_train_steps=args.ckpt_interval
        )
        callbacks = [checkpoints]
    trainer = L.Trainer(
        default_root_dir="checkpoints",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
    )
    if args.ckpt_path is None:
        model = ShortTermGoalCVAE(
            args.mineclip_path,
            clip_dim=args.clip_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
    else:
        model = ShortTermGoalCVAE.load_from_checkpoint(args.ckpt_path)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.save_checkpoint("goal_cvae.ckpt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True)
    parser.add_argument(
        "-m",
        "--mineclip-path",
        type=str,
        required=True,
        help="MineCLIP checkpoint path",
    )
    parser.add_argument(
        "-c",
        "--ckpt-path",
        type=str,
        help="Path to model checkpoint for resuming training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-w", "--n-workers", type=int, default=8)
    parser.add_argument("--ckpt-dir", type=str)
    parser.add_argument("--ckpt-interval", type=int, default=100)
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--split",
        type=float,
        default=0.1,
        help="Ratio of samples to reserve for testing",
    )
    parser.add_argument(
        "--goal-bound",
        type=int,
        default=128,
        help="Bound on number of frames from end of run to sample long-term goal",
    )
    parser.add_argument(
        "--min-video-len",
        type=int,
        default=128,
        help="Minimum number of frames in video",
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

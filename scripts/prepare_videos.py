from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PRESETS = {
    "kinetics": {
        "clip_len": 16,
        "stride": 8,
        "size": 224,
        "val_ratio": 0.1,
        "max_videos": 0,
    },
    "something-something": {
        "clip_len": 16,
        "stride": 4,
        "size": 224,
        "val_ratio": 0.1,
        "max_videos": 0,
    },
    "ego4d": {
        "clip_len": 32,
        "stride": 16,
        "size": 224,
        "val_ratio": 0.05,
        "max_videos": 0,
    },
}


def iter_video_paths(root: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def extract_clips(video_path: Path, clip_len: int, stride: int, size: int):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    clips = []
    if len(frames) < clip_len:
        return clips
    for i in range(0, len(frames) - clip_len + 1, stride):
        clip = np.stack(frames[i : i + clip_len], axis=0).astype(np.uint8)
        clips.append(clip)
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare .npy clips from raw videos")
    parser.add_argument("--input-root", required=True, help="Root folder containing raw videos")
    parser.add_argument("--train-out", default="data/clips/train")
    parser.add_argument("--val-out", default="data/clips/val")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="kinetics")
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-videos", type=int, default=None, help="0 means all videos")
    args = parser.parse_args()

    chosen = PRESETS[args.preset].copy()
    if args.clip_len is not None:
        chosen["clip_len"] = args.clip_len
    if args.stride is not None:
        chosen["stride"] = args.stride
    if args.size is not None:
        chosen["size"] = args.size
    if args.val_ratio is not None:
        chosen["val_ratio"] = args.val_ratio
    if args.max_videos is not None:
        chosen["max_videos"] = args.max_videos

    input_root = Path(args.input_root)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    videos = sorted(iter_video_paths(input_root))
    if chosen["max_videos"] > 0:
        videos = videos[: chosen["max_videos"]]

    print(
        f"preset={args.preset} clip_len={chosen['clip_len']} stride={chosen['stride']} "
        f"size={chosen['size']} val_ratio={chosen['val_ratio']} videos={len(videos)}"
    )

    total = 0
    train_count = 0
    val_count = 0

    val_period = max(int(1 / max(chosen["val_ratio"], 1e-6)), 1)

    for v_idx, video in enumerate(videos):
        clips = extract_clips(video, chosen["clip_len"], chosen["stride"], chosen["size"])
        for c in clips:
            use_val = total % val_period == 0
            out_dir = val_out if use_val else train_out
            out_path = out_dir / f"clip_{total:08d}.npy"
            np.save(out_path, c)
            total += 1
            if use_val:
                val_count += 1
            else:
                train_count += 1

        if (v_idx + 1) % 20 == 0:
            print(f"processed_videos={v_idx+1} total_clips={total}")

    print(f"done videos={len(videos)} train_clips={train_count} val_clips={val_count} total={total}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render Motion-JEPA 1D encoder and predictor masks as a PNG timeline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from mask.collators import MaskCollator1D


BACKGROUND = (248, 249, 251)
GRID = (210, 214, 220)
TEXT = (26, 30, 36)
MUTED = (101, 109, 120)
EMPTY = (229, 232, 237)
ENCODER = (45, 112, 214)
HIDDEN = (151, 159, 170)
PRED_COLORS = (
    (225, 72, 72),
    (235, 145, 38),
    (137, 87, 214),
    (34, 164, 122),
    (207, 74, 151),
    (68, 157, 205),
)


def _font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _mask_from_indices(indices: torch.Tensor, total: int, sample_index: int) -> np.ndarray:
    mask = np.zeros(total, dtype=bool)
    mask[indices[sample_index].cpu().numpy()] = True
    return mask


def sample_masks(
    config: dict,
    *,
    seed: int,
    valid_length: int,
    batch_size: int,
    sample_index: int,
    allow_overlap: bool | None,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    data = config["data"]
    mask_config = config["mask"]
    num_frames = int(data["num_frames"])
    overlap = bool(mask_config["allow_overlap"]) if allow_overlap is None else allow_overlap
    collator = MaskCollator1D(
        num_frames=num_frames,
        enc_frame_mask_ratio=tuple(mask_config["enc_frame_mask_ratio"]),
        pred_frame_mask_ratio=tuple(mask_config["pred_frame_mask_ratio"]),
        nenc=int(mask_config["num_enc_masks"]),
        npred=int(mask_config["num_pred_masks"]),
        allow_overlap=overlap,
    )
    state = collator.state_dict()
    state["counter"] = seed - 1
    collator.load_state_dict(state)
    feature_dim = int(data.get("motion_dim", 366))
    batch = [
        (
            torch.zeros(num_frames, feature_dim),
            torch.tensor(float(data.get("fps", 60))),
            valid_length,
        )
        for _ in range(batch_size)
    ]
    _, encoder_blocks, predictor_blocks = collator(batch)
    if len(encoder_blocks) != 1:
        raise ValueError("This renderer currently expects num_enc_masks=1.")
    encoder = _mask_from_indices(encoder_blocks[0], num_frames, sample_index)
    predictors = [
        _mask_from_indices(indices, num_frames, sample_index)
        for indices in predictor_blocks
    ]
    return encoder, predictors, overlap


def render_mask_png(
    output: Path,
    encoder: np.ndarray,
    predictors: list[np.ndarray],
    *,
    valid_length: int,
    seed: int,
    allow_overlap: bool,
) -> dict[str, int | float]:
    total = len(encoder)
    valid = np.arange(total) < valid_length
    predictor_count = np.stack(predictors).sum(axis=0)
    predictor_union = predictor_count > 0
    hidden = valid & ~encoder
    unused = valid & ~encoder & ~predictor_union
    encoder_predictor_overlap = encoder & predictor_union
    predictor_overlap = predictor_count >= 2

    rows: list[tuple[str, np.ndarray, tuple[int, int, int] | None]] = [
        ("ENC visible context", encoder, ENCODER),
        ("Hidden from encoder", hidden, HIDDEN),
    ]
    rows.extend(
        (f"PRED {index + 1}", mask, PRED_COLORS[index % len(PRED_COLORS)])
        for index, mask in enumerate(predictors)
    )
    rows.extend(
        [
            ("PRED union", predictor_union, (196, 55, 72)),
            ("PRED overlap count", predictor_count, None),
            ("Neither ENC nor PRED", unused, (184, 188, 196)),
        ]
    )

    width = 1800
    left, right = 260, 190
    plot_width = width - left - right
    row_height, row_gap = 42, 10
    header_height, summary_height = 155, 155
    height = header_height + len(rows) * (row_height + row_gap) + summary_height
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    label_font = _font(18, bold=True)
    body_font = _font(17)
    small_font = _font(14)

    draw.text((left, 25), "Motion-JEPA 1D mask coverage", fill=TEXT, font=title_font)
    draw.text(
        (left, 70),
        f"frames={total}   valid={valid_length}   seed={seed}   "
        f"allow_overlap={str(allow_overlap).lower()}",
        fill=MUTED,
        font=body_font,
    )
    draw.text(
        (left, 103),
        "ENC is the final context shown to the online encoder; each PRED row is a target block.",
        fill=MUTED,
        font=body_font,
    )

    def frame_bounds(frame: int) -> tuple[int, int]:
        x0 = left + round(frame / total * plot_width)
        x1 = left + round((frame + 1) / total * plot_width)
        return x0, max(x0 + 1, x1)

    for row_index, (label, values, color) in enumerate(rows):
        y0 = header_height + row_index * (row_height + row_gap)
        y1 = y0 + row_height
        draw.text((left - 14, (y0 + y1) // 2), label, fill=TEXT, font=label_font, anchor="rm")
        draw.rectangle((left, y0, left + plot_width, y1), fill=EMPTY, outline=GRID)
        for frame in range(total):
            x0, x1 = frame_bounds(frame)
            if frame >= valid_length:
                fill = (205, 209, 216)
            elif color is not None:
                fill = color if bool(values[frame]) else EMPTY
            else:
                count = int(values[frame])
                fill = (
                    EMPTY
                    if count == 0
                    else (247, 190, 196)
                    if count == 1
                    else (223, 91, 108)
                    if count == 2
                    else (154, 36, 59)
                    if count == 3
                    else (92, 18, 37)
                )
            draw.rectangle((x0, y0 + 1, x1, y1 - 1), fill=fill)
        for frame in range(0, total + 1, 10):
            x = left + round(frame / total * plot_width)
            draw.line((x, y0, x, y1), fill=GRID, width=1)
        active_count = int(np.count_nonzero(values[:valid_length])) if color is not None else int(np.count_nonzero(values[:valid_length]))
        if label == "PRED overlap count":
            annotation = f"{int(predictor_count[:valid_length].max())}× max"
        else:
            annotation = f"{active_count} ({active_count / valid_length:.1%})"
        draw.text((left + plot_width + 12, (y0 + y1) // 2), annotation, fill=MUTED, font=small_font, anchor="lm")

    axis_y = header_height + len(rows) * (row_height + row_gap) - row_gap + 8
    for frame in range(0, total + 1, 10):
        x = left + round(frame / total * plot_width)
        draw.text((x, axis_y), str(frame), fill=MUTED, font=small_font, anchor="ma")

    pred_details = "   ".join(
        f"P{index + 1}: {int(mask[:valid_length].sum())}"
        for index, mask in enumerate(predictors)
    )
    summary_y = axis_y + 48
    summary_lines = (
        f"ENC visible: {int(encoder[:valid_length].sum())}/{valid_length} "
        f"({encoder[:valid_length].mean():.1%})    "
        f"Hidden: {int(hidden[:valid_length].sum())}/{valid_length} "
        f"({hidden[:valid_length].mean():.1%})",
        f"PRED union: {int(predictor_union[:valid_length].sum())}/{valid_length} "
        f"({predictor_union[:valid_length].mean():.1%})    "
        f"ENC∩PRED: {int(encoder_predictor_overlap[:valid_length].sum())}    "
        f"PRED-overlap frames: {int(predictor_overlap[:valid_length].sum())}",
        pred_details,
    )
    for index, line in enumerate(summary_lines):
        draw.text((left, summary_y + index * 30), line, fill=TEXT, font=body_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "frames": total,
        "valid_frames": valid_length,
        "encoder_visible": int(encoder[:valid_length].sum()),
        "encoder_hidden": int(hidden[:valid_length].sum()),
        "predictor_union": int(predictor_union[:valid_length].sum()),
        "encoder_predictor_overlap": int(encoder_predictor_overlap[:valid_length].sum()),
        "predictor_overlap_frames": int(predictor_overlap[:valid_length].sum()),
        "max_predictor_overlap": int(predictor_count[:valid_length].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mjepa_1d.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/mask-1d.png"))
    parser.add_argument("--seed", type=int, default=0, help="Mask collator step/seed.")
    parser.add_argument("--valid-length", type=int, help="Valid frames (default: num_frames).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--allow-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mask.allow_overlap from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    total = int(config["data"]["num_frames"])
    valid_length = total if args.valid_length is None else int(args.valid_length)
    if not 1 <= valid_length <= total:
        raise ValueError(f"--valid-length must be in [1, {total}].")
    if args.batch_size <= 0 or not 0 <= args.sample_index < args.batch_size:
        raise ValueError("--sample-index must select an item in --batch-size.")
    encoder, predictors, overlap = sample_masks(
        config,
        seed=args.seed,
        valid_length=valid_length,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        allow_overlap=args.allow_overlap,
    )
    stats = render_mask_png(
        args.output,
        encoder,
        predictors,
        valid_length=valid_length,
        seed=args.seed,
        allow_overlap=overlap,
    )
    print(f"Saved mask visualization to {args.output.resolve()}")
    print("  " + "  ".join(f"{key}={value}" for key, value in stats.items()))


if __name__ == "__main__":
    main()

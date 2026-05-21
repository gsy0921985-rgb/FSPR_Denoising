#!/usr/bin/env python3
"""
Minimal batch inference for FSPR / AdaReNet (ada_config=4).
Does not include training, datasets, or loss definitions.

Example:
  python demo_test.py --ckpt path/to/weights.pt --input_dir ./sample_data --output_dir ./results
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as tvF

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AdaReNet import AdaReNet  # noqa: E402


def _list_images(folder: Path):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _forward_padded(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[2], x.shape[3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    y = model(x)
    return y[:, :, :h, :w]


def main():
    parser = argparse.ArgumentParser(description='FSPR / AdaReNet inference (lite)')
    parser.add_argument('--ckpt', type=str, required=True, help='state_dict .pt (not shipped in public lite)')
    parser.add_argument('--input_dir', type=str, default='./sample_data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ada-config', type=int, default=4, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not inp.is_dir():
        print(f'错误: 输入目录不存在: {inp}', file=sys.stderr)
        sys.exit(1)
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f'错误: checkpoint 不存在: {ckpt}', file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = AdaReNet(in_channels=3, out_channels=3, ada_config=int(args.ada_config)).to(device)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    n = 0
    with torch.no_grad():
        for path in _list_images(inp):
            pil = Image.open(path).convert('RGB')
            x = tvF.to_tensor(pil).unsqueeze(0).to(device)
            y = _forward_padded(model, x)
            y = y.squeeze(0).cpu().clamp(0, 1)
            save_path = out / f'denoised_{path.stem}.png'
            tvF.to_pil_image(y).save(save_path)
            print(save_path)
            n += 1
    if n == 0:
        print(f'警告: {inp} 下未找到图像', file=sys.stderr)
        sys.exit(1)
    print(f'完成，共 {n} 张。')


if __name__ == '__main__':
    main()

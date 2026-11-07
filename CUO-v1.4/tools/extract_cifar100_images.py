#!/usr/bin/env python3
"""
Extract images from CIFAR-100 python-format files.

Usage:
  python extract_cifar100_images.py --data-dir /path/to/cifar-100-python \
      --out-root /path/to/output --format bmp

This script expects the python-version files named `train` and `test` (the
standard CIFAR-100 python distribution). It will create `train/<label>/...`
and `val/<label>/...` under `out_root` and save images as BMP/PNG.
"""
import os
import sys
import argparse
import pickle
import tarfile
import tempfile
import shutil
from pathlib import Path

try:
    from PIL import Image
except Exception:
    print('Pillow is required. Install with: pip install pillow')
    raise

try:
    import numpy as np
except Exception:
    print('Numpy is required. Install with: pip install numpy')
    raise


def unpickle(fpath):
    with open(fpath, 'rb') as fh:
        return pickle.load(fh, encoding='bytes')


def process_file(file_path, out_root, split_name, out_ext):
    d = unpickle(file_path)
    data = d.get(b'data')
    if data is None:
        raise RuntimeError(f'No b"data" key found in {file_path}')
    fnames = [n.decode() for n in d.get(b'filenames', [])]
    labels = d.get(b'fine_labels', d.get(b'labels'))
    n = len(fnames)
    saved = 0
    for i in range(n):
        arr = np.array(data[i], dtype=np.uint8)
        arr = arr.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(arr)
        label = labels[i]
        label_dir = os.path.join(out_root, split_name, f"{label:02d}")
        os.makedirs(label_dir, exist_ok=True)
        fname = os.path.splitext(fnames[i])[0] + '.' + out_ext
        out_path = os.path.join(label_dir, fname)
        img.save(out_path)
        saved += 1
        if saved % 1000 == 0:
            print(f'{split_name}: saved {saved}/{n}')
    print(f'Done {split_name}: saved {saved} images to {os.path.join(out_root, split_name)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='/data/xiaotong/data/cifar-100-python', help='path to CIFAR-100 python folder or tar.gz')
    p.add_argument('--out-root', default='/data/xiaotong/images/cifar100', help='output root for images')
    p.add_argument('--format', choices=['bmp', 'png'], default='bmp', help='output image format')
    args = p.parse_args()

    data_dir = args.data_dir
    tempdir = None
    try:
        if os.path.isfile(data_dir) and (data_dir.endswith('.tar.gz') or data_dir.endswith('.tgz') or data_dir.endswith('.zip')):
            tempdir = tempfile.mkdtemp(prefix='cifar100_')
            print(f'Extracting {data_dir} to {tempdir}...')
            if data_dir.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(data_dir, 'r') as zf:
                    zf.extractall(tempdir)
            else:
                with tarfile.open(data_dir, 'r:*') as tf:
                    tf.extractall(tempdir)
            # try to find the extracted folder that contains 'train' and 'test'
            cand = None
            for root, dirs, files in os.walk(tempdir):
                if 'train' in files and 'test' in files:
                    cand = root
                    break
            if cand is None:
                # fallback: use tempdir
                cand = tempdir
            data_folder = cand
        else:
            data_folder = data_dir

        train_file = os.path.join(data_folder, 'train')
        test_file = os.path.join(data_folder, 'test')
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print('Could not find `train` and `test` files in', data_folder)
            print('If you have a tar.gz, use that path; otherwise ensure you point --data-dir to the extracted CIFAR-100 python folder.')
            sys.exit(2)

        out_root = args.out_root
        out_ext = args.format
        os.makedirs(out_root, exist_ok=True)

        print('Processing train...')
        process_file(train_file, out_root, 'train', out_ext)
        print('Processing test -> val...')
        process_file(test_file, out_root, 'val', out_ext)

        print('All done. You can now run tools/split_cifar100.py with root_dir set to', out_root)

    finally:
        if tempdir:
            shutil.rmtree(tempdir)


if __name__ == '__main__':
    main()

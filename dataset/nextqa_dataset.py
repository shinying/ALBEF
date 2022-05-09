import csv
import glob
import json
import os.path as op
import random

from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class nextqa_dataset(Dataset):

    def __init__(self, root, split, nframe, transform):
        self.root = root
        assert split in ['train', 'val', 'test']

        self.data = list(csv.DictReader(open(op.join(self.root, split+'.csv'))))
        self.vid2id = json.load(open(op.join(self.root, 'map_vid_vidorID.json')))

        self.transform = transform
        self.rand_sampling = split == 'train'

        self.nframe = nframe
        self.num_choices = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Read video
        vid = self.vid2id[sample["video"]]
        files = read_frames(op.join(self.root, 'frames', vid), self.nframe)
        frames = [self.transform(Image.open(f)) for f in files]

        question = sample["question"]
        choices = [sample[f"a{i}"] for i in range(self.num_choices)]
        label = int(sample["answer"])

        return frames, question, choices, label, len(frames)


# Borrow from https://github.com/m-bain/frozen-in-time/blob/main/base/base_dataset.py
def sample_frames(num_frames, vlen, sample='rand', crop=1., fix_start=-1):
    acc_samples = min(num_frames, vlen)
    rem = (1. - crop) / 2
    start, stop = int(vlen * rem), int(vlen * (1 - rem))
    intv = np.linspace(start=start, stop=stop, num=acc_samples+1).astype(int)
    if sample == 'rand':
        frame_idxs = [random.randrange(intv[i], intv[i+1]) for i in range(len(intv)-1)]
    elif fix_start >= 0:
        fix_start = int(fix_start)
        frame_idxs = [intv[i]+fix_start for i in range(len(intv)-1)]
    elif sample == 'uniform':
        frame_idxs = [(intv[i]+intv[i+1]-1) // 2 for i in range(len(intv)-1)]
    else:
        raise NotImplementedError
    return frame_idxs


def read_frames(video_path, num_frames, sample='rand', crop=1., fix_start=-1):
    frames = glob.glob(op.join(video_path, '*.png'))
    if not len(frames):
        raise FileNotFoundError("no such videos")

    frames.sort(key=lambda n: int(op.basename(n)[:-4]))
    while len(frames) < num_frames / crop: # duplicate frames
        frames = [f for frame in frames for f in (frame, frame)]
    frame_idxs = sample_frames(num_frames, len(frames), sample, crop, fix_start)
    return [frames[i] for i in frame_idxs]

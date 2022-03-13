import glob
import json
import math
import os.path as op
import random

from PIL import Image
from torch.utils.data import Dataset


class vlep_dataset(Dataset):

    def __init__(self, root, split, transform, max_len=512):
        self.root = root
        assert split in ['train', 'dev']

        data_file = op.join(root, f'vlep_{split}_release_fix.jsonl')
        self.data = [json.loads(line) for line in open(data_file)]
        self.subtitles = json.load(open(op.join(root, 'vlep_subtitles.json')))

        self.max_len = max_len
        self.transform = transform
        self.FPS = 3
        self.rand_sampling = split == 'train'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        s, e = sample['ts']

        # Read video
        files = glob.glob(op.join(self.root, 'vlep_frames', sample['vid_name'], '*.jpg'))
        files.sort(key=lambda n: int(op.basename(n).split('.')[0]))
        fs, fe = self.FPS * math.floor(s), min(self.FPS * math.ceil(e) + 1, len(files))
        fm = random.randrange(fs, fe) if self.rand_sampling else (fs + fe) // 2
        frame = self.transform(Image.open(files[fm]))

        # Read subtitle
        subs = [sub['text'].strip() for sub in self.subtitles[sample['vid_name']]
                if s <= sub['ts'][1] and e >= sub['ts'][0]]

        # Make sure the length of the sentence < max length
        event_len = max(len(e.split()) for e in sample['events'])
        lens = [len(sub.split()) for sub in subs] + [event_len]
        lens_sum = sum(lens)
        i = 0
        while lens_sum > self.max_len:
            lens_sum -= lens[i]
            subs.pop(0)
            i += 1
        subs = ' '.join(subs)

        choices = [f'{subs} Next: {event}' for event in sample['events']]
        return frame, choices, sample['answer']



import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from ast import literal_eval

class Fill50kDataset(Dataset):
    def __init__(self, split="train"):
        self.data = []
        self.data_dir = './data/fill50k'
        with open(os.path.join(self.data_dir, f'{split}.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_dir, 'fill50k/' + source_filename))
        target = cv2.imread(os.path.join(self.data_dir, 'fill50k/' + target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class PhotoSketchDataset(Dataset):
    def __init__(self, split="train"):
        self.sketches = []
        self.images = []
        self.prompts = []

        self.data_dir = './data/sketch/'
        self.sketch_dir = os.path.join(self.data_dir, 'sketch-rendered/width-5')
        self.img_dir = os.path.join(self.data_dir, 'image')
        # read in splits
        with open(os.path.join(self.data_dir, f'list/{split}.txt'), 'rt') as f:
            for line in f:
                img_idx = line.strip()
                for idx in range(1, 6):
                    sketch_path = os.path.join(self.sketch_dir, f'{img_idx}_0{idx}.png')
                    self.sketches.append(sketch_path)

                    img_path = os.path.join(self.img_dir, f'{img_idx}.jpg')
                    self.images.append(img_path)

                    prompt_path = os.path.join(f'./lavis/captions/{img_idx}.json')
                    self.prompts.append(prompt_path)
    
        assert len(self.prompts) == len(self.sketches)
        assert len(self.images) == len(self.sketches)

        """
        heights = []
        widths = []
        for i in range(len(self.prompts)):
            img = cv2.imread(self.sketches[idx])
            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)
        print("max height: ", max(heights))
        print("max width: ", max(widths))
        """

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        # assuming that we get the items by looping through sketch 1 of all images, then sketch 2 of all images, etc.

        prompt_file = self.prompts[idx]
        with open(prompt_file) as f:
            prompt = literal_eval(f.read().strip())[0]
        
        print("prompt: ", prompt)
        print("source: ", self.sketches[idx])
        print("target: ", self.images[idx])

        # source is the sketch
        source = cv2.imread(self.sketches[idx])
        target = cv2.imread(self.images[idx])
        
        source_height, source_width, _ = source.shape 
        target_height, target_width, _ = target.shape

        max_height = 240 
        max_width = 360
        
        pad_bot = max_height - source_height 
        pad_right = max_width - source_height 

        assert pad_bot >= 0
        assert pad_right >= 0

        # pad
        source = cv2.copyMakeBorder(source, 0, pad_bot, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        target = cv2.copyMakeBorder(target, 0, pad_bot, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)

        """
        # write padded image
        source_stub = self.sketches[idx].split('/')[-1]
        target_stub = self.sketches[idx].split('/')[-1]
        os.makedirs(os.path.join(self.sketch_dir, 'padded'), exist_ok=True)
        print("writing to: ", os.path.join(self.sketch_dir, 'padded', source_stub))
        cv2.imwrite(os.path.join(self.sketch_dir, 'padded', source_stub), source)
        cv2.imwrite(os.path.join(self.img_dir, 'padded', target_stub), target)
        """

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)


        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


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
        
        self.padded_sketch_dir = os.path.join(self.sketch_dir, 'padded')
        self.padded_img_dir = os.path.join(self.img_dir, 'padded')
        
        os.makedirs(self.padded_sketch_dir, exist_ok=True)
        os.makedirs(self.padded_img_dir, exist_ok=True)

        # read in splits
        with open(os.path.join(self.data_dir, f'list/{split}.txt'), 'rt') as f:
            for line in f:
                img_idx = line.strip()
                for idx in range(1, 6):
                    
                    sketch_stub = f'{img_idx}_0{idx}.png'
                    img_stub = f'{img_idx}.jpg'

                    prompt_path = os.path.join(f'./lavis/captions/{img_idx}.json')
                    self.prompts.append(prompt_path)
        
                    # source is the sketch
                    source = cv2.imread(os.path.join(self.sketch_dir, sketch_stub))
                    target = cv2.imread(os.path.join(self.img_dir, img_stub))

                    source, target = self.process_imgs(source, target)

                    sketch_path = os.path.join(self.padded_sketch_dir, sketch_stub)
                    cv2.imwrite(sketch_path, source)
                    # add the padded sketch path
                    self.sketches.append(sketch_path)
                    
                    img_path = os.path.join(self.padded_img_dir, img_stub)
                    cv2.imwrite(img_path, target)
                    # add the padded img path
                    self.images.append(img_path)
    
        assert len(self.prompts) == len(self.sketches)
        assert len(self.images) == len(self.sketches)

    def __len__(self):
        return len(self.sketches)

    # Crop images to 512 x 512, then pad
    # TODO: instead of cropping, rescale?

    def process_imgs(self, source, target):

        max_height = 512
        max_width = 512
        
        source = source[:max_height, :max_width, :]
        target = target[:max_height, :max_width, :]
        
        source_height, source_width, _ = source.shape 
        target_height, target_width, _ = target.shape

        # pad
        source = cv2.copyMakeBorder(source, 0, max_height-source_height, 0, max_width-source_width, borderType=cv2.BORDER_CONSTANT, value=0)
        target = cv2.copyMakeBorder(target, 0, max_height-target_height, 0, max_width-target_width, borderType=cv2.BORDER_CONSTANT, value=0)

        return source, target 

    def __getitem__(self, idx):
    # assuming that we get the items by looping through sketch 1 of all images, then sketch 2 of all images, etc.

        prompt_file = self.prompts[idx]
        with open(prompt_file) as f:
            prompt = literal_eval(f.read().strip())[0]
        
        # source is the sketch
        source = cv2.imread(self.sketches[idx])
        target = cv2.imread(self.images[idx])
        

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        assert source.shape == target.shape

        return dict(jpg=target, txt=prompt, hint=source)



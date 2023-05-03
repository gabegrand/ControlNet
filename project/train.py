import sys

sys.path.append('..')
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import * 
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 2 
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Misc
print("Creating data...")
train_dataset = PhotoSketchDataset(split='train')
val_dataset = PhotoSketchDataset(split='val')
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq)
#trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = pl.Trainer(gpus=3, accelerator="gpu", strategy="ddp", precision=32, callbacks=[logger])

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
print("Creating models...")
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Train!
trainer.fit(model, train_dataloader, val_dataloader)

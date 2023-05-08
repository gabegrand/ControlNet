import argparse
import sys

sys.path.append('..')
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import * 
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
#import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = args.batch_size 
logger_freq = 300
learning_rate = args.lr 
sd_locked = True
only_mid_control = False

exp_dir = f'./experiments/lr={learning_rate}_bs={batch_size}'

print("Experiment Directory: ", exp_dir)
print("Learning Rate: ", learning_rate)
print("Batch Size: ", batch_size)
if batch_size >= 4:
    per_device_batch_size = 4
    grad_acc_steps = int(batch_size / per_device_batch_size)
    # TODO: this won't work for batch sizes that aren't multiples of 4
    assert per_device_batch_size * grad_acc_steps == batch_size
else:
    per_device_batch_size = batch_size
    grad_acc_steps = 1
print("Per Device Batch Size: ", per_device_batch_size)
print("Grad Accumulation Steps: ", grad_acc_steps)

# Wandb
#wandb_logger = pl.loggers.WandbLogger()
#wandb_logger.experiment.config.update({"batch_size": batch_size, "lr": learning_rate})

# Misc
print("Creating data...")
train_dataset = PhotoSketchDataset(split='train')
val_dataset = PhotoSketchDataset(split='val')
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=per_device_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=per_device_batch_size, shuffle=False)
image_logger = ImageLogger(batch_frequency=logger_freq)
stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val/loss", mode="min", patience=3)
callbacks = [image_logger, stop_callback]
#callbacks = [stop_callback]
#trainer = pl.Trainer(gpus=1, precision=32, callbacks=callbacks, default_root_dir=exp_dir, accumulate_grad_batches=grad_acc_steps)
trainer = pl.Trainer(gpus=3, accelerator="gpu", strategy="ddp", precision=32, callbacks=callbacks, default_root_dir=exp_dir, accumulate_grad_batches=grad_acc_steps)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
print("Creating models...")
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Train!
trainer.fit(model, train_dataloader, val_dataloader)

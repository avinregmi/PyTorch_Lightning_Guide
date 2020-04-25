# PyTorch_Lightning_Guide

### PyTorch Lightning Function
**def __init__(self):**

**def forward(self, x):**

**def training_step(self, batch, batch_idx): (REQUIRED)**

Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

def configure_optimizers(self): (REQUIRED)

def train_dataloader(self): (REQUIRED)

def validation_step(self, batch, batch_idx): (OPTIONAL)

def validation_epoch_end(self, outputs): (OPTIONAL)

def test_step(self, batch, batch_idx): (OPTIONAL)

def test_epoch_end(self, outputs): (OPTIONAL)

def val_dataloader(self): (OPTIONAL)

def test_dataloader(self): (OPTIONAL)

def prepare_data(self): (OPTIONAL)


### *Trainer*
For more info visit: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class
``` python
from pytorch_lightning import Trainer
model = LitMNIST()
trainer = Trainer(gpus=1)
trainer.fit(model)
```

Args:

  **max_epochs:** Stop training once this number of epochs is reached.

  **min_epochs:** Force training for at least these many epochs
 
  **max_steps:** Stop training after this number of steps. Disabled by default (None).
  
  **min_steps:** Force training for at least these number of steps. Disabled by default (None).
  **logger:** Logger (or iterable collection of loggers) for experiment tracking.

  **checkpoint_callback:** Callback for checkpointing.

  **early_stop_callback** (:class:`pytorch_lightning.callbacks.EarlyStopping`):
  **callbacks:** Add a list of callbacks.
  **default_root_dir:** Default path for logs and weights when no logger/ckpt_callback passed

  **gradient_clip_val:** 0 means don't clip.
 
  **process_position:** orders the progress bar when running multiple models on same machine.
  **num_nodes:** number of GPU nodes for distributed training.
  
  **gpus:** Which GPUs to train on.
  **auto_select_gpus:**
      If enabled and `gpus` is an integer, pick available
      gpus automatically. This is especially useful when
      GPUs are configured to be in "exclusive mode", such
      that only one process at a time can access them.
 
  **num_tpu_cores:** How many TPU cores to train on (1 or 8).

  **progress_bar_refresh_rate:** How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
      Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.
 

  **check_val_every_n_epoch:** Check val every n train epochs.
  **fast_dev_run:** runs 1 batch of train, test  and val to find any bugs (ie: a sort of unit test).
  **accumulate_grad_batches:** Accumulates grads every k batches or as set up in the dict.
 
  
  **train_percent_check:** How much of training dataset to check.
  
  **val_percent_check:** How much of validation dataset to check.
  
  **test_percent_check:** How much of test dataset to check.
  
  **val_check_interval:** How often within one training epoch to check the validation set
  
  **log_save_interval:** Writes logs to disk this often
  
  **row_log_interval:** How often to add logging rows (does not write to disk)
  
  **distributed_backend:** The distributed backend to use.
 
  **precision:** Full precision (32), half precision (16).
 
  **weights_summary:** Prints a summary of the weights when training begins.
  **weights_save_path:** Where to save weights if specified. Will override default_root_dir
          for checkpoints only. Use this if for whatever reason you need the checkpoints
          stored in a different place than the logs written in `default_root_dir`.
  **amp_level:** The optimization level to use (O1, O2, etc...).
  **num_sanity_val_steps:** Sanity check runs n batches of val before starting the training routine.

# PyTorch Lightning Guide

The general pattern is that each loop (training, validation, test loop) has 3 methods:

-   `___step`
    
-   `___step_end`
    
-   `___epoch_end`

# Lifecycle[](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#lifecycle)

### The methods in the  [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule "pytorch_lightning.core.LightningModule")  are called in this order:

1.  `__init__()`
    
2.  [`prepare_data()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.prepare_data "pytorch_lightning.core.LightningModule.prepare_data")
    
3.  [`configure_optimizers()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.configure_optimizers "pytorch_lightning.core.LightningModule.configure_optimizers")
    
4.  [`train_dataloader()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.train_dataloader "pytorch_lightning.core.LightningModule.train_dataloader")
    

**If you define a validation loop then**

5.  [`val_dataloader()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.val_dataloader "pytorch_lightning.core.LightningModule.val_dataloader")
    

**And if you define a test loop:**

6.  [`test_dataloader()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_dataloader "pytorch_lightning.core.LightningModule.test_dataloader")

*In every epoch, the loop methods are called in this frequency:*

1.  [`validation_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step "pytorch_lightning.core.LightningModule.validation_step")  called every batch
    
2.  [`validation_epoch_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_epoch_end "pytorch_lightning.core.LightningModule.validation_epoch_end")  called every epoch

# LightningModule
[pl.LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#)

 - **def __init__(self):**
	 > Define Model Architecture 
 - **def forward(self,x)**
	 > Forward pass our data
 - **def training_step(self, batch_idx): (REQUIRED)**
	>  Parameters:
	> - **batch**: The Output of your DataLoader. A tensor, tuple or list.
	> - **batch_idx** (int): Integer displaying index of this batch
	> - **optimizer_idx** (int): When using multiple optimizer, this argument is used.
	> - **hiddens** (Tensor): Passed in if truncated_bptt_steps > 0
    > 
    >Returns:
    > Dict with loss key and optional log or progress bar keys. 
    > - **loss**: tensor scalar (**required**)
    > - **progress_bar**: Dict for progress bar display (Tensor)
    > - **log**: Dict for metrics to add to logger
    > ```python
    >output = {
    >	'loss': loss, # required
    >     'progress_bar': {'training_loss':loss}, # Optional, must be tensor
   >     'log': logger_logs: {'losses': logger_loss}
   > }
   > return output

 - **def training_step_end(**batch_parts_outputs)**: (OPTIONAL)**
Use this when training with dp or ddp2 because [`training_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.training_step "pytorch_lightning.core.LightningModule.training_step") will operate on only part of the batch. However, this is still optional and only needed for things like softmax or NCE loss.
	> Parameters:
	>**batch_parts_outputs**: What you return in training_step for each batch part.
	>
	> Return: 
	>Dict with loss key and optional log or progress bar keys.
    >	-   loss -> tensor scalar  **REQUIRED**    
    >	-   progress_bar -> Dict for progress bar display. Must have only tensors
   >	-   log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)


   
 - **training_epoch_end(outputs)**
 Called at the end of the training epoch with the outputs of all training steps.
   >Parameters:
   > - **outputs**: List of outputs you defined in training_step() or if there are multiple dataloaders, a list containing a list of outputs for each dataloader.
   >
   >Returns:
   > Dict or OrderedDict. May contain the following optional keys:
   >-   log (metrics to be added to the logger; only tensors) 
   >-   any metric used in a callback (e.g. early stopping).

	The outputs here are strictly for logging or progress bar. If you don’t need to display anything, don’t return anything.

 - **def configure_optimizers(self): (REQUIRED)**
 Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you’d need one. But in the case of GANs or similar you might have multiple.
	> Returns:
	> - Single Optimizer
	> List of Tuple - List of Optimizer
	> ```python
	>def configure_optimizers(self): # most cases
	>    opt = Adam(self.parameters(), lr=1e-3)
	>    return opt
	>    
	> def configure_optimizers(self):
	>    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
	>    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
	 >    return generator_opt, disriminator_opt

 - **def validation_step(batch, batch_idx, dataloader_idx): (OPTIONAL)**
Operates on a single batch of data from the validation set. In this step you’d might generate examples or calculate anything of interest like accuracy.
	> Parameters:
	> - **batch**: The output of your DataLoader. 
	>- **batch_idx** (int): The index of this batch
	> - **dataloader_idx** (int): The index of the dataloader that produced this batch (only if multiple val datasets used)
	>
	> Return: Dict or OrderedDict - passed to [`validation_epoch_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_epoch_end "pytorch_lightning.core.LightningModule.validation_epoch_end"). If you defined [`validation_step_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step_end "pytorch_lightning.core.LightningModule.validation_step_end") it will go to that first.
	
 - **def validation_step_end(**batch_parts_outputs)**: (OPTIONAL)**
Operates on a single batch of data from the validation set. In this step you’d might generate examples or calculate anything of interest like accuracy.
	> Parameters:
	>**batch_parts_outputs**[](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step_end.params.batch_parts_outputs) : What you return in [`validation_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step "pytorch_lightning.core.LightningModule.validation_step") for each batch part.
	>
	> Return: 
	>**Dict or OrderedDict** - passed to the [`validation_epoch_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_epoch_end "pytorch_lightning.core.LightningModule.validation_epoch_end") method.
	 
 - **def validation_epoch_end(**outputs**: (OPTIONAL)**
 Called at the end of the validation epoch with the outputs of all validation steps.
	 > Parameters:
	> - **outputs**: List of outputs you defined in [`validation_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step "pytorch_lightning.core.LightningModule.validation_step"), or if there are multiple dataloaders, a list containing a list of outputs for each dataloader.
	>
	>Retruns:
	>Dict or OrderedDict. May have the following optional keys:
	> - **progress_bar** (dict for progress bar display; only tensors)
	> - **log** (dict of metrics to add to logger; only tensors).
	> ``` python
	>def validation_epoch_end(self, outputs):
   >     val_acc_mean = 0
    >     for output in outputs:
    >         val_acc_mean += output['val_acc']
    >     val_acc_mean /= len(outputs)
    >     tqdm_dict = {'val_acc': val_acc_mean.item()}
    >    # show val_acc in progress bar but only log val_loss
    >    results = {
     >       'progress_bar': tqdm_dict,
      >       'log': {'val_acc': val_acc_mean.item()}
    >     }
    >     return results
	
 - **def test_step(batch, batch_idx, dataloader_idx): (OPTIONAL)**
Operates on a single batch of data from the test set. In this step you’d normally generate examples or calculate anything of interest such as accuracy.
	> Parameters:
	> - **batch**: The output of your DataLoader. 
	>- **batch_idx** (int): The index of this batch
	> - **dataloader_idx** (int): he index of the dataloader that produced this batch (only if multiple test datasets used).
	>
	> Return: Dict or OrderedDict - Dict or OrderedDict - passed to the [`test_epoch_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_epoch_end "pytorch_lightning.core.LightningModule.test_epoch_end") method. If you defined [`test_step_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step_end "pytorch_lightning.core.LightningModule.test_step_end") it will go to that first.

 - **def test_step_end(**batch_parts_outputs)**: (OPTIONAL)**
Use this when testing with dp or ddp2 because [`test_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step "pytorch_lightning.core.LightningModule.test_step") will operate on only part of the batch. However, this is still optional and only needed for things like softmax or NCE loss.
	> Parameters:
	>**batch_parts_outputs**: What you return in [`test_step()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step "pytorch_lightning.core.LightningModule.test_step") for each batch part.
	> Return: 
	>**Dict or OrderedDict**: Dict or OrderedDict - passed to the [`test_epoch_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_epoch_end "pytorch_lightning.core.LightningModule.test_epoch_end").

 - **def test_epoch_end(**outputs**: (OPTIONAL)**
Called at the end of a test epoch with the output of all test steps.	 
> Parameters:
	> - **outputs**: List of outputs you defined in [`test_step_end()`](https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step_end "pytorch_lightning.core.LightningModule.test_step_end"), or if there are multiple dataloaders, a list containing a list of outputs for each dataloader
	>
	>Retruns:
	>Dict or OrderedDict. May have the following optional keys:
	> - **progress_bar** (dict for progress bar display; only tensors)
	> - **log** (dict of metrics to add to logger; only tensors).
	> ``` python
	>def test_epoch_end(self, outputs):
   >     test_acc_mean = 0
    >     for output in outputs:
    >         test_acc_mean += output['test_acc']
    >     test_acc_mean /= len(outputs)
    >     tqdm_dict = {'test_acc': test_acc_mean.item()}
    >    # show val_acc in progress bar but only log val_loss
    >    results = {
     >       'progress_bar': tqdm_dict,
      >       'log': {'test_acc': test_acc_mean.item()}
    >     }
    >     return results

 - **def prepare_data(self): (OPTIONAL)**
 Use this to download and prepare data. In distributed (GPU, TPU), this will only be called once. This is called before requesting the dataloaders:
```python
	def prepare_data(self):
	    download_imagenet()
	    clean_imagenet()
	    cache_imagenet()
```
 - **def train_dataloader(self): (Required)**
 Implement a PyTorch DataLoader for training.
	  > Returns:
	  > - Single PyTorch DataLoader

 - **def val_dataloader(self): (Optional)**
 Implement a PyTorch DataLoader for validation.
	  > Returns:
	  > - Single PyTorch DataLoader

 - **def test_dataloader(self): (Optional)**
 Implement a PyTorch DataLoader for Testing.
	  > Returns:
	  > - Single PyTorch DataLoader


# *Trainer*
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
	


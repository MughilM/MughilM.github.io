---
layout: post
title: How I set up my deep learning projects
description: >
  The frameworks I use to cleanly set up training and evaluation experiments for Kaggle competitions.
date: 25 Mar 2024
tags: [PyTorch Lightning]
applause_button: true
---

There have been a few posts where I detail specific deep learning projects I have done, most recently the UNet 
image segmentation task for detecting contrails. However, that post is in a notebook format, and when I run
my own experiments, I do not use a Jupyter notebook and instead run everything in scripts. So how do I set up
everything cleanly so that training is a relatively easy process to iterate on?

Before I answer that question, a huge thanks to [Lukas (ashleve)](https://github.com/ashleve) and his 
[amazing repository](https://github.com/ashleve/lightning-hydra-template) for giving the inspiration. Many ideas
are from his framework.

* toc
{:toc}
* * *

## The 3 Main Components
There are three main parts to my workflow: Hydra, PyTorch Lightning, and a model tracking framework (Weights and 
Biases, MLFlow, etc.). These components work in tandem so that as much of engineering code is taken out of the equation.
The effect is that I can focus on more of the research aspect of the problem e.g. testing different architectures,
different ways of preprocessing data, and do not waste my time on other issues.

I'll provide a quick overview of these parts, and then show how they were used for the contrail project.

## Hydra
The first part is the Hydra package. This places all configuration and training parameters into YAML files, 
which are then loaded up during each training run. The benefit of placing everything in files is that it provides
a paper trail of what has been tried and what hasn't been. You can also override these parameters on the command line.
Each time the program runs, it outputs to a directory with the full YAML configuration. Therefore, you can load up
past experiments when, for example, needing to simply evaluate a model.

One extra benefit is that we can instantiate full classes straight from these configuration files. This provides
immense flexibility in which parts of the codebase we can configure. For example, for PyTorch Lightning to work,
at minimum we need `LightningDataModule`, `LightningModule` and `Trainer` objects. All of these can be configured
and instantiated with Hydra. 

## PyTorch Lightning
PyTorch Lightning provides a convenient wrapper for PyTorch. Normally, to code a training loop using pure PyTorch,
many statements that move the data back and forth between the CPU and GPU need to be written. Optimization, 
back-propagation, and dataset and dataloader creation and iteration need to be done manually. These little 
steps which are consistent across projects can introduce places of error.

Therefore, PyTorch Lightning abstracts all of these common steps, so that the user does not need to worry about them.
It also introduces a `.fit()` function to PyTorch, similar to TensorFlow's functionality.

## Weights and Biases, MLFlow, and others
The last part concerns model tracking. Training and evaluating a model is fine, but ample tracking of metrics 
and other performance is necessary to fully evaluate how a model is performing. It can also lead to ways to improve
the model. MLFlow provides tracking without needing an account, and it logs anything you tell it to, including
model weights and other artifacts.

Weights and Biases needs an account, and it is my preferred method of tracking. The UI is fantastic, and it can
log a wide variety of things, such as images, tables, models, and other artifacts. It can also perform hyperparameter
tuning for you, and provide neat visualizations of each individual run. 

PyTorch Lightning provides direct integration of these two services, plus many more, as you'll see in the example below.
For a completely bare bones logging framework, you can enable to output the metrics to a local CSV file. 

## Example
We have seen how the 3 components can work together in theory. Next, I'll show how I set up my experiments using
the frameworks and packages listed above. In many ways, it is similar to Lukas's approach, but I have provided my own
tweaks. Running experiments this way leads to more files being created, but they reflect the modular approach
the frameworks provide.

### Directory structure
Using Hydra necessitates creating a special file structure that takes advantage of Hydra's features. The main
code will lie in `src`, while YAML configuration files lie in `config`. Finally, we also need the `pyrootutils`
package so we can easily set up the source root for Python. This eliminates pesky module-related errors when 
running scripts and referencing files in sibling directories of the parent. I have omitted some of the contents
of the directories for brevity.

```text
root
|-- config
  |-- callbacks
  |-- datamodule
  |-- experiment
  |-- hydra
  |-- logger
  |-- model
  |-- paths
  |-- trainer
  train.yaml
|-- data
|-- logs
|-- src
  |-- datamodules
  |-- models
    |-- nets
  |-- utils
  custom_callbacks.py
  train.py
```

### `src` Directory
This is where all the core code is located. It is where datasets, models, training code, and other tidbits are
actually defined. The `train.py` file is the script that is actually run by way `python train.py`. Comparatively,
this is not much code written in this file, as it is only grabbing the configurations, and instantiating all
the necessary classes before calling `trainer.fit()`.

To set up a PyTorch Lightning training run correctly, at minimum you need a `LightningDataModule`, `LightningModule`,
and a `Trainer` object. Optionally, you can provide callbacks and loggers as lists of `Callback`'s and `Logger`'s.
Please see below for what a basic training script looks like. 

```python
# file: "src/train.py"
import pyrootutils

# Whichever directory the .git or .idea file is located,
# that will be set as the root.
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
)

from typing import Optional, List
import logging
import os

# Hydra packages
import hydra
from omegaconf import DictConfig, OmegaConf

# PyTorch Lightning packages + torch
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback, RichProgressBar

# Local function that instantiates all callbacks in a list
from utils.instantiators import instantiate_callbacks


torch.set_float32_matmul_precision('medium')
os.environ['HYDRA_FULL_ERROR'] = '1'


def train(cfg: DictConfig):
    log.info(f'Full hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set seed for everything (numpy, random number generator, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
    # Create the data directory in case it's completely missing
    os.makedirs(cfg.paths.data_dir, exist_ok=True)

    if cfg.get('wandb_enabled'):
        wandb_logger = WandbLogger(project=cfg.datamodule.comp_name, save_dir=cfg.paths.log_dir)
    else:
        wandb_logger = WandbLogger(project=cfg.datamodule.comp_name, save_dir=cfg.paths.log_dir, mode='disabled')

    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>...')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f'Instantiating model <{cfg.model._target_}>...')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info('Instantiating callbacks...')
    if cfg.get('callbacks'):
        callbacks = instantiate_callbacks(cfg.get('callbacks'))

    log.info(f'Instantiating Trainer...')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, callbacks=callbacks)

    log.info('Starting training...')
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))

# The train.yaml file in the config directory is set as the entry point.
# That full configuration is sent to this method as a DictConfig object
@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)
    return 1.0

# It might seem strange, but there is actually NO argument for the main()
# method when we call it here
if __name__ == '__main__':
    log = logging.getLogger('train')
    main()
```

The `instantiate_callbacks` method simply instantiates each callback defined in a list, and returns that whole list.

```python
# file: "src/utils/instantiators.py
def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
```

Notice that we also have directories for the datamodules and models. These contain their own files, which define
the requisite classes necessary for PyTorch Lightning. Additionally, the `nets` directory contain sole definitions
for neural nets, which are extended directly from `nn.Module`. 

Datamodules are custom classes that extend PyTorch Lightning's `LightningDataModule`, and models are classes
that extend `LightningModule`. These are defined in their own Python files e.g. for my contrails, the datamodule
is as follows:

```python
# file: "src/datamodules/gr_contrails_data.py"
import pytorch_lightning as pl
# ... other imports ...

class GRContrailDataModule(pl.LightningDataModule):
    def __init__(self, comp_name: str = 'google-research-identify-contrails-reduce-global-warming',
                 data_dir: str = 'data/', frac: float = 1.0, batch_size: int = 128, num_workers: int = 4,
                 pin_memory: bool = True, use_val_as_train: bool = True, validation_split: float = 0.2,
                 train_url: str = None, val_url: str = None, test_url: str = None, binary: bool = False):
        super().__init__()
        # ... more code ...
    # ... other required methods like prepare_data(), train_dataloaders(), etc.

# We also need the actual torch Dataset class, this is instantiated from GRContrailDataModule
class GRContrailsFalseColorDataset(Dataset):
    def __init__(self, image_dir, directory_ids=None, text=False, binary=False):
        # ... code ...
```

As for the model, we have the following:

```python
# file: "src/models/gr_contrails_model.py
import torch.optim as optim
from src.models.nets.unet import UMobileNet
# ... other imports ...

class GRContrailsClassifierModule(pl.LightningModule):
    def __init__(self, image_size: int, in_image_channels: int, optimizer: optim.Optimizer):
        super().__init__()
        self.model = UMobileNet(image_size, in_image_channels, 1)
        # ... more code ...
    # ... more code ...
```

### `hydra` Directory
Once the datamodule and model classes are written, we use YAML files to configure the classes and provide it
the correct variable values. The datamodule and model are the main ones, but there are others that are handy
and necessary. Providing the values are very straightforward, but the main thing is the presence of the 
`_target_` key, as that tells Hydra which class to instantiate. For example, for our datamodule, we have

```yaml
# file: "config/datamodule/gr_contrails.yaml
_target_: src.datamodules.gr_contrails_data.GRContrailDataModule

comp_name: google-research-identify-contrails-reduce-global-warming
data_dir: ${paths.data_dir}
frac: 1.0
batch_size: 32
num_workers: 4
pin_memory: true
use_val_as_train: true
train_url: ...
val_url: ...
test_url: ...
```

I also make use many handy Hydra shorthand syntax e.g. `${paths.data_dir}` takes the value defined in the 
`data_dir` key under the `paths/default.yaml` configuration. Please check my Github repository directly to 
check out all the full list of Hydra files!

## Conclusion
Those are the basics of how I set up all of my data science projects. There are a few weaknesses here, as I can 
only do projects that use a predictive deep learning format (data preprocessing --> model training --> post evaluation).
Non-DL projects will lead their own separate codebase.
Processes such as reinforcement learning which have a completely different training procedure will also need their
own framework. But this is a good starting point for many different projects.





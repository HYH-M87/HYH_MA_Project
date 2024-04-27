# How to train model
##  configuration files
The configuration files for the model, dataset, schedule, and so on are consistent with those used in MMdetection. the original configuration files are located in 4_dependencies/libraries/mmdetection/configs, and their backup copies are kept in 3_configurations/code_configuration for easy access and use.

## train model
After the relevant files are configured, you can use the scripts under 6_scripts/train_model for training, which includes train.sh and dist_train.sh.
### 1.train.sh
The script calls mmdetection/tools/train.py and can accept two parameters. The parameter list is as follows:

```
CONFIG: the path to your configuration file
RESUME(Optional): If specify checkpoint path, resume from it, while if not specify, try to auto resume from the latest checkpoint in the work directory. the default value is "auto"
```
```
usage: bash train.sh cfg.py checkpoint.pth
```

### 2.dist_train.sh
run a distributed PyTorch training session using the torch.distributed.launch module, the parameter list is as follows:

```
CONFIG: the path to your configuration file
GPUS:  specifying the number of GPUs to use per node.
```
```
usage: bash dist_train.sh cfg.py 8
```

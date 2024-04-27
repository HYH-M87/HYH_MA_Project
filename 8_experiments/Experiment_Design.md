# The structure of the directory

```
.
├── Experiment_Design.md
├── Experiment Summary.xlsx
└── MA_Detection
    └── hyh_ma_det_exp001
        ├── cfg
        │   ├── _base_
        │   │   ├── retinanet_r50_fpn.py
        │   │   └── retinanet_tta.py
        │   ├── dataset.py
        │   ├── model.py
        │   ├── runtime.py
        │   └── schedule.py
        ├── Readme.md
        └── run.py
```
**Experiment Summary.xlsx**
Record the main content of the experimental design for quick retrieval of experiments, including the experiment title, brief description, purpose, main parameters, etc.
**MA_Detection**
Define a category of experiments.
**hyh_ma_det_exp001**
The specific name of an experiment.
- **cfg**
The configuration files for the experiment-related model, dataset, and training strategy, where _base_ is the most fundamental configuration file, originating from MMdetection.
- **run.py**
integrates four configuration files from the cfg directory.

# Log file
log files and the result will save to 9_log/MA_Detection/hyh_ma_det_exp001

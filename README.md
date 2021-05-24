# New project template
todo:
need to describe data template and DVC usage
netapp disk: (e.g. maybe)
    ProjectDirectory.csv
    DATA/project1/dataversion1/

## Introduction
This repository contains a skeleton project template for use with new GSTT-CSC projects. The template provides a starting point with pre-configured access to local GPU resources and MLflow tracking.

A complete example of an AI application for 3D spleen segmentation using this template can be found [here](https://github.com/GSTT-CSC/MLOps_test_project.git).

## Structure
At a minimum users should use the `Experiment` class and the provided `run_project.py` script to set up their experiment.

This template suggests using pytorch-lightning and MONAI for network configuration and DataModules. 
However, this is not strictly necessary and provided the Dockerfile GPU libraries are adapted and the `run_project` function is used then tracking can be performed with any [MLflow compatible framework](https://mlflow.org/docs/latest/tracking.html#automatic-logging).

## Getting started
This project template makes use of the `Experiment` class from https://github.com/GSTT-CSC/MLOps.git. To add this to your python environemt:

```shell
git clone https://github.com/GSTT-CSC/MLOps.git
pip install MLOps/
```

The first thing to do after cloning this template is to rename the appropriate files and folders to make the directory project specific. 
The `project` directory should be renamed to make it clear that it contains your project files. 

There are 3 main components that need to be completed after cloning the template:

### 1. `config/config.cfg`
The config file contains all the information that is used for configuring up the project, experiment, and tracking server. 
The config file path is passed as an argument to the `Experiment` class where the experiment and tracking are configured. 

Note: The values present in the template config file are the minimum required, be careful not to remove any but adding new ones to help configure parts of your project is encourages (e.g. entry points).

#### `[entry_points]`
The entry points defined here allow you to enter the project through different routes. It is necessary to define at least one entry_point called `main`, in this case `main` runs the command `python3 main.py`. 
You can sepcifiy any number of custom entry points. For example, if you had a script that performed hyperparamter optimisation you might want to trigger that workflow by defining another entry point `optimise = python3 optimisation.py`
under `main`. Entry points can be selected at runtime by adding the `-e` flag to `run_project` e.g. `python run_project.py -e optimise`.


### 2. `project/Network.py`
This file is used to define the pytorch `LightningModule` class.

### 3. `project/DataModule.py`
This file is used to define the pytorch `LightningDataModule` class.

## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.

[Laurence Jackson](https://github.com/laurencejackson) (GSTT-CSC) 

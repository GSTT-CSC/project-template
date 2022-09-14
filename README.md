<!-- PROJECT HEADING -->
<br />
<p align="center">
<a href="https://github.com/github_username/repo_name">
    <img src="assets/MOps_template_logo.png" alt="Logo" width="50%">
  </a>
<p align="center">
A framework for AI applications for healthcare
<br />
<br />
<a href="https://github.com/GSTT-CSC/Project_template">View repo</a>
·
<a href="https://github.com/GSTT-CSC/Project_template/issues">Report Bug</a>
·
<a href="https://github.com/GSTT-CSC/Project_template/issues">Request Feature</a>
</p>

# New project template

## Introduction
This repository contains a skeleton project template for use with new projects using the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) development platform. The template provides a starting point with helper classes and functions to facilitate rapid development and deployment of applications.

## Structure
At a minimum users should use the `Experiment` class and the provided `run_project.py` script to set up their experiment.
This template suggests using pytorch-lightning and MONAI for network configuration and DataModules. 
However, this is not strictly necessary and provided the Dockerfile GPU libraries are adapted and the `run_project` function is used then tracking can be performed with any [MLflow compatible framework](https://mlflow.org/docs/latest/tracking.html#automatic-logging).

## Getting started
This project template makes use of classes and functions provided by the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) package, installing this to your local environment is easy with pip:

```shell
pip install csc-mlops
```

To run your project once you've set it up following the guidelines below execute the run_project.py script, run the following for usage information.
```shell 
python3 run_project.py --help
```

The first thing to do after cloning this template is to rename the appropriate files and folders to make the directory project specific. 
The `project` directory should be renamed to make it clear that it contains your project files. 

There are 3 main components that need to be completed after cloning the template:

### 1. `config/config.cfg`
The config file contains all the information that is used for configuring up the project, experiment, and tracking server. 
The config file path is passed as an argument to the `Experiment` class where the experiment and tracking are configured. 

Note: The values present in the template config file are the minimum required, be careful not to remove any but adding new ones to help configure parts of your project is encouraged (e.g. entry points).

#### `[entry_points]`
The entry points defined here allow you to enter the project through different routes. It is necessary to define at least one entry_point called `main`, for example including the line:
```
main = python3 train.py
```
will run the command `python3 train.py` when the main entry point is requested.

You can specify any number of custom entry points. For example, if you had a script that performed hyperparameter optimisation you might want to trigger that workflow by defining another entry point `optimise = python3 optimisation.py`
under `main`. Entry points can be selected at runtime by adding the `-e` flag to `run_project` e.g. `python run_project.py -e optimise`.

### 2. `project/Network.py`
This file is used to define the pytorch `LightningModule` class.

### 3. `project/DataModule.py`
This file is used to define the pytorch `LightningDataModule` class.

### 4. Setup GitHub actions
To run your tests using GithHub actions the `.github/workflows/development_test.yml` and `.github/workflows/production_test.yml` files should be modified.

These workflows use environment variables, defined at the top of the workflow to make testing easier.

The production tests also use a GitHub secret to authenticate the writing of a gist to store the test coverage badge `auth: ${{ secrets.PYTEST_COVERAGE_COMMENT }}`. GitHub secrets are hidden string variables stored at a repository level, these can be defined in the repository settings.

More information about how the test coverage badge is defined can be found [here](https://github.com/Schneegans/dynamic-badges-action).

## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


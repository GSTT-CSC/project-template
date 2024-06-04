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

## Getting started
This project template makes use of classes and functions provided by the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) package, installing this to your local environment is easy with pip:

```shell
pip install csc-mlops
```

To begin a training run once you've set up your project following the guidelines below, utilise the csc-mlops package run command.

```shell
mlops run scripts/train.py -c config/config.cfg
```


Run the following for usage information.
```shell 
mlops run -h
```

The first thing to do after cloning this template is to rename the appropriate files and folders to make the directory project specific. 
The `project` directory should be renamed to make it clear that it contains your project files. 

There are 5 main components that need to be completed after cloning the template:

### 1. `config/config.cfg` `config/local_config.cfg`
The config file contains all the information that is used for configuring up the project, experiment, and tracking server. This includes training parameters and XNAT configurations.

The config file path is also passed as an argument to the MLOps `Experiment` class where the experiment and tracking are configured.

As there will be differences between local development and running on DGX (for example XNAT configurations), it is highly encouraged to make use of local_config when devlopiong locally.

Note: The values present in the template config files are examples, you can remove any except those in [server] and [project] which are necessary for MLOps. Outside of these you are encouraged to add and modify the config files as relevant to your project.

### 2. `project/Network.py`
This file is used to define the pytorch `LightningModule` class.

This is where you set the Network architecture and flow that you will use for training, validation, and testing. 

Here you can set up which metrics are calculated and at which stage in the flow these occur, along with the model and optimiser.

This example has numerous metrics and steps that are not necessary, feel free to delete or add as relevant to your project.

### 3. `project/DataModule.py`
This file is used to define the pytorch `LightningDataModule` class.

This is where you define the data that is used for training, validation, and testing.

This example involves retrieving data from XNAT which may not be necessary for your project, and additionally has data validation steps that might not be relevant.


### 4. `scripts/train.py`
This file is used to define the training run.

This is where the datamodule and network are pulled together.

This example also uses callbacks to retrieve the best model parameters.

### 5. `Dockerfile`
This dockerfile sets up the Docker image that the MLOps run will utilise.

In this example this is just a simple environment running python version 3.10.
You will most likely need to adapt this for your project.

Examples of projects utilising these components:

https://github.com/GSTT-CSC/CARNAX-Neonatal-Abdominal-X-Ray

https://github.com/GSTT-CSC/wrist-fracture-x-ray

https://github.com/GSTT-CSC/dental-classifier

For further information on MLOps please refer to the MLOps tutorial repo:

https://github.com/GSTT-CSC/MLOps-tutorial


There are also additional steps that are strongly recommended to be setup for you project:

### 1. Setup GitHub actions
To run your tests using GitHub actions the `.github/workflows/development_test.yml` and `.github/workflows/production_test.yml` files should be modified.

These workflows use environment variables, defined at the top of the workflow to make testing easier.

The production tests also use a GitHub secret to authenticate the writing of a gist to store the test coverage badge `auth: ${{ secrets.PYTEST_COVERAGE_COMMENT }}`. GitHub secrets are hidden string variables stored at a repository level, these can be defined in the repository settings.

More information about how the test coverage badge is defined can be found [here](https://github.com/Schneegans/dynamic-badges-action).

### 2. Setup Git Hooks

This repository contains a pre-commit hook that helps prevent committing sensitive information to the repository by scanning your commits for certain patterns like names, addresses, phone numbers, patient IDs, etc.

#### 2.1. Setting up the Pre-commit Hook
The pre-commit hook script is located in the git_hooks directory. Copy the pre-commit script from this directory to the .git/hooks/ directory in your local repository.

```bash
cp .github/hooks/pre-commit .git/hooks/ 
```

Make the script executable:

```bash
chmod +x .git/hooks/pre-commit
```

The script will now automatically check the files you're about to commit for any sensitive information patterns.

#### 2.2. Setting up exceptions
Sometimes, there may be legitimate cases where these patterns are allowed. In these cases, you can add exceptions to the .sensitive_exceptions and .files_exceptions files. Populating these files is not mandatory for git hooks to work but should be kept in the root of the project directory.

The .sensitive_exceptions file should contain any specific instances of the forbidden patterns that you want to allow. Each exception should be on its own line. You can for instance add specific addresses or dates you wish to push to remote.

The .files_exceptions file should contain any files/directories that you want to exclude from the checks. Each file should be on its own line.

These files are added to .gitignore as they are not advised to be committed. 

### 2.3. Resolving Pre-commit Hook Issues

When the pre-commit hook identifies potential sensitive information in a commit, it will prevent the commit from being completed and output information about the offending files and patterns.

How you view this output will depend on your method of committing:

- **VSCode**: If you're using VSCode UI to commit your changes, you can view the pre-commit hook output by clicking on "Show command output" when the error is thrown. 

- **Terminal**: If you're committing via terminal, the output will be displayed directly in the terminal.


## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


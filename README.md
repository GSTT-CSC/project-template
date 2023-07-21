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
However, this is not strictly necessary and provided the `mlops run` CLI is used then tracking can be performed with any [MLflow compatible framework](https://mlflow.org/docs/latest/tracking.html#automatic-logging).

## Getting started
This project template makes use of classes and functions provided by the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) package, installing this to your local environment is easy with pip:

```shell
pip install csc-mlops
```

To run your project once you've set it up following the guidelines below execute the run_project.py script, run the following for usage information.
```shell 
mlops run -h
```

The first thing to do after cloning this template is to rename the appropriate files and folders to make the directory project specific. 
The `project` directory should be renamed to make it clear that it contains your project files. 

There are 3 main components that need to be completed after cloning the template:

### 1. `config/config.cfg`
The config file contains all the information that is used for configuring up the project, experiment, and tracking server. 

The config file path is passed as an argument to the `Experiment` class where the experiment and tracking are configured. 

Note: The values present in the template config file are the minimum required, be careful not to remove any but adding new ones to help configure parts of your project is encouraged.

### 2. `project/Network.py`
This file is used to define the pytorch `LightningModule` class.

### 3. `project/DataModule.py`
This file is used to define the pytorch `LightningDataModule` class.

### 4. Setup GitHub actions
To run your tests using GithHub actions the `.github/workflows/development_test.yml` and `.github/workflows/production_test.yml` files should be modified.

These workflows use environment variables, defined at the top of the workflow to make testing easier.

The production tests also use a GitHub secret to authenticate the writing of a gist to store the test coverage badge `auth: ${{ secrets.PYTEST_COVERAGE_COMMENT }}`. GitHub secrets are hidden string variables stored at a repository level, these can be defined in the repository settings.

More information about how the test coverage badge is defined can be found [here](https://github.com/Schneegans/dynamic-badges-action).

### 5. Setup Git Hooks

This repository contains a pre-commit hook that helps prevent committing sensitive information to the repository by scanning your commits for certain patterns like names, addresses, phone numbers, patient IDs, etc.

#### 5.1. Setting up the Pre-commit Hook
The pre-commit hook script is located in the git_hooks directory. Copy the pre-commit script from this directory to the .git/hooks/ directory in your local repository.

```bash
cp .github/hooks/pre-commit .git/hooks/ 
```

Make the script executable:

```bash
chmod +x .git/hooks/pre-commit
```

The script will now automatically check the files you're about to commit for any sensitive information patterns.

#### 5.2. Setting up exceptions
Sometimes, there may be legitimate cases where these patterns are allowed. In these cases, you can add exceptions to the .sensitive_exceptions and .files_exceptions files.

The .sensitive_exceptions file should contain any specific instances of the forbidden patterns that you want to allow. Each exception should be on its own line.
The .files_exceptions file should contain any files/directories that you want to exclude from the checks. Each file should be on its own line.

The Team Members names and the documentation folder are set as exceptions by default. 

### 5.3. Resolving Pre-commit Hook Issues

When the pre-commit hook identifies potential sensitive information in a commit, it will prevent the commit from being completed and output information about the offending files and patterns.

How you view this output will depend on your method of committing:

- **VSCode**: If you're using VSCode UI to commit your changes, you can view the pre-commit hook output by clicking on "Show command output" when the error is thrown. 

- **Terminal**: If you're committing via terminal, the output will be displayed directly in the terminal.


## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


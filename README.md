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
This repository contains mutliple folders with useful files depending on the type of project.
The first thing to do after cloning this template is to delete unnecessary folders, and rename the remaining one specific to your project.
For example if creating a fracture classifier on X-Rays, the `2D_Classifier' directory should be kept and then renamed to 'Fracture' to make it clear that it contains your project files. 

#### Examples of projects that have been built using the project template:

https://github.com/GSTT-CSC/AutoSegCT

https://github.com/GSTT-CSC/wrist-fracture-x-ray

For further information on MLOps please refer to the MLOps repo:

https://github.com/GSTT-CSC/MLOps


### Additional steps that are strongly recommended for project setup:

### 1. Set up GitHub Actions
To run your tests using GitHub actions the `.github/workflows/development_test.yml` and `.github/workflows/production_test.yml` files should be modified.

These workflows use environment variables, defined at the top of the workflow to make testing easier.

The production tests also use a GitHub secret to authenticate the writing of a gist to store the test coverage badge `auth: ${{ secrets.PYTEST_COVERAGE_COMMENT }}`. GitHub secrets are hidden string variables stored at a repository level, these can be defined in the repository settings.

More information about how the test coverage badge is defined can be found [here](https://github.com/Schneegans/dynamic-badges-action).

### 2. Set up Git Hooks

This repository contains a pre-commit hook that helps prevent committing sensitive information to the repository by scanning your commits for certain patterns like names, addresses, phone numbers, patient IDs, etc.

#### 2.1. Set up the Pre-commit Hook
The pre-commit hook script is located in the git_hooks directory. Copy the pre-commit script from this directory to the .git/hooks/ directory in your local repository.

```bash
cp .github/hooks/pre-commit .git/hooks/ 
```

Make the script executable:

```bash
chmod +x .git/hooks/pre-commit
```

The script will now automatically check the files you're about to commit for any sensitive information patterns.

#### 2.2. Set up Pre-commit Hook Exceptions
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


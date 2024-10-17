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

# New Image Classification Project Template

## Introduction
This repository contains a skeleton project template for use with new image classification projects using the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) development platform. The template provides a starting point with helper classes and functions to facilitate rapid development and deployment of applications.

## Getting started

1. The first thing to do after cloning this template is to rename the appropriate files and folders to make the directory project specific. 
The `project` directory should be renamed to make it clear that it contains your project files. 

2. This project template makes use of XNAT for image storage and MLOps. Follow instructions in https://github.com/GSTT-CSC/MLOps-tutorial to set up XNAT and MLOps for use. However you don't need to use XNAT for the datastore the train.py can be run with a local dataset.    

3. This project template makes use of classes and functions provided by the [csc-mlops](https://github.com/GSTT-CSC/MLOps.git) package, installing this to your local environment is easy with pip:

```shell
pip install csc-mlops
```
4. Define the config parameters and details in `config/config.cfg` and `config/local_config.cfg`. Further details about the config files can be seen below. 

5. Once, the configuration variables have been defined. To begin a training run once you've set up your project following the guidelines below, utilise the csc-mlops package run command.

```shell
mlops run scripts/train.py -c config/config.cfg
```

Run the following for usage information.
```shell 
mlops run -h
```

## Project Structure 

### There are 5 main components that need to be completed after cloning the template:

### 1. `config/config.cfg` and `config/local_config.cfg`
The config file contains all the information that is used for configuring the project, experiment, and tracking server. This includes training parameters and XNAT configurations.

The config file path is also passed as an argument to the MLOps `Experiment` class where the experiment and tracking are configured.

As there will be differences between local development and running on DGX (for example XNAT configurations), it is highly encouraged to make use of `local_config` when devlopiong locally.

The config files have the sections `[server]`, `[project]`, `[system]`, `[xnat]` and `[params]`.  `[server]` defines the mlflow project,  `[project]` conatines the name and details about the project, `[system]` contains details about the computer systems detailing the CPU and GPU availability and usage, `[xnat]` contains the XNAT project connections information, and `[params]` contains the parameters for the training model. The values present in the template config files are examples, you can remove any except those in `[server]` and `[project]` which are necessary for MLOps. Outside of these you are encouraged to add and modify the config files as relevant to your project.


### 2. `project/Network.py`
This file is used to define the PyTorch `LightningModule` class.

This is where you set the Network architecture and flow that you will use for training, validation, and testing. 

Here you can set up which metrics are calculated and at which stage in the flow these occur, along with the model and optimiser.

The example has numerous metrics and steps that are not always necessary, feel free to delete or add as relevant to your project.

### 3. `project/DataModule.py`
This file is used to define the PyTorch `LightningDataModule` class.

This is where you define the data that is used for training, validation, and testing.

The example involves retrieving data from XNAT (more on this below) which may not be necessary for your project. There are additional data validation steps that might not be relevant, feel free to delete or add as relevant to your project.


### 4. `scripts/train.py`
This file is used to define the training run. 

This is where the `Datamodule` and `Network` are pulled together and the values defines in the config file is used to define the datamodule 
and network used in in the training.  

The example includes callbacks to retrieve the best model parameters, feel free to delete or add as relevant to your project.

### 5. `Dockerfile`
This dockerfile sets up the Docker image that the MLOps run will utilise.

In the example this is just a simple environment running python version 3.10.
You will most likely need to adapt this for your project.

When runnning not with XNAT and local data you will not use the dockerfile.


#### Examples of projects utilising these components:

https://github.com/GSTT-CSC/CARNAX-Neonatal-Abdominal-X-Ray

https://github.com/GSTT-CSC/wrist-fracture-x-ray

https://github.com/GSTT-CSC/dental-classifier

For further information on MLOps please refer to the MLOps tutorial repo:

https://github.com/GSTT-CSC/MLOps-tutorial


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


## Utility functions that may be useful

### XNAT data handler
Accessing data stored in an XNAT archive is performed through two steps - first the XNAT database is queried for project subjects using the DataBuilderXNAT class. This list of results is then loaded using the PyTorch style data loading transform called LoadImageXNATd.

![](assets/xnat-image-import.png)


#### 1. Create list of data samples
A list of subjects is extracted from the XNAT archive for the specified project. This is done automatically by the helper function `xnat_build_dataset`. 
```python
from utils.tools import xnat_build_dataset

PROJECT_ID = 'my_project'
xnat_configuration = {'server': XNAT_LOCATION,
                      'user': XNAT_USER,
                      'password': XNAT_USER_PASSWORD,
                      'project': XNAT_PROJECT_ID}

xnat_data_list = xnat_build_dataset(self.xnat_configuration)
``` 
Each element in the list `xnat_data_list` is a dictionary with two keys, Where these fields indicated unique references to each subject. 
```
{
    'subject_id': <subject_id>,
    'subject_uri': <subject_uri>
}
```
#### 2. Download relevant data using LoadImageXNATd and actions
A MONAI transform `LoadImageXNATd` is used to download the data from XNAT. This transform can be used in place of the conventional `LoadImaged` transform provided by MONAI to access local data.

A worked example is given below to create a valid dataloader containing the sag_t2_tse scans from XNAT where each subject has two experiments
This first thing that is required is an action function. This is a function that operates on an XNAT SubjectData object and returns the desired ImageScanData object from the archive and the key under which is will be stored in the dataset. For example the function below will extract the 'sag_t2_tse' scans from the archive.

```python
def fetch_sag_t2_tse(subject_data: SubjectData = None) -> (ImageScanData, str):
    """
    Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
    along with the 'key' that it will be used to access it.
    """
    for exp in subject_data.experiments:
        if 'MR_2' in subject_data.experiments[exp].label:
            for scan in subject_data.experiments[exp].scans:
                if 'sag_t2_tse' in subject_data.experiments[exp].scans[scan].series_description:
                    return subject_data.experiments[exp].scans[scan], 'sag_t2_tse'
```

In this example, the `fetch_sag_t2_tse` function will loop over all experiments available for the subject, then if one of these experiments has 'MR_2' in the label it will loop over all the scans in this experiment until it finds one with 'sag_t2_tse' in the series_description. The URI to this scan is then extracted and returned along with the key it will be stored under in the data dictionary, in this case 'sag_t2_tse'. 

We can now pass this action function to the `LoadImageXNATd` transform which will perform each action function in the list sequentially. So if multiple datasets are required for each Subject then multiple functions can be used.

```python
from transforms.LoadImageXNATd import LoadImageXNATd
from monai.transforms import Compose, ToTensord
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from xnat.mixin import ImageScanData, SubjectData
from monai.data.utils import list_data_collate

# list of actions to be applied sequentially
actions = [fetch_sag_t2_tse]

train_transforms = Compose(
    [
        LoadImageXNATd(keys=['subject_uri'], actions=actions, xnat_configuration=xnat_configuration),
        ToTensord(keys=['sag_t2_tse'])
    ]
)

dataset = CacheDataset(data=xnat_data_list, transform=train_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=list_data_collate)
```

If further transforms are required they can be added to the `Compose` transform list as usual.


## Contact
For bug reports and feature requests please raise a GitHub issue on this repository.


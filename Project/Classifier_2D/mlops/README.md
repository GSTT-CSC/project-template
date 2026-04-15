## Example Workflow:

#### 1. Adapt XNATDataImport.py for your data
#### 2. Adapt Datamodule.py and Network.py if necessary
#### 3. Perform hyperparameter tuning using tune.py
#### 4. Utilise these hyperparameter values with train.py and threshold tune.
#### 5. Success! Model is stored in mlflow and ready for app development.

More information can be found below.

## There are 7 main components for training 2D Classifiers:

### 1. `config/config.cfg` and `config/local_config.cfg`
The config file contains all the information that is used for configuring the project, experiment, and tracking server. This includes training parameters and XNAT configurations.

The config file path is also passed as an argument to the MLOps `Experiment` class where the experiment and tracking are configured.

As there will be differences between local development and running on DGX (for example XNAT configurations), it is highly encouraged to make use of `local_config` when developing locally.

Note: The values present in the template config files are examples, you can remove any except those in `[server]` and `[project]` which are necessary for MLOps. Outside of these you are encouraged to add and modify the config files as relevant to your project.

### 2. `src/XNATDataImport.py`
This file is used to define and pull the required data from XNAT. It utilises DataBuilderXNAT to do so as shown in the example, if you require additional or different data from XNAT additional actions can be added.

If your data is not stored in XNAT this can be replaced by any method that accesses your data.

### 3. `src/Network.py`
This file is used to define the PyTorch `LightningModule` class.

This is where you set the Network architecture and flow that you will use for training, validation, and testing. 

Here you can set up which metrics are calculated and at which stage in the flow these occur, along with the model and optimiser.

The example has numerous metrics and steps that are not always necessary, feel free to delete or add as relevant to your project.

### 4. `src/DataModule.py`
This file is used to define the PyTorch `LightningDataModule` class.

This is where you define the data that is used for training, validation, and testing.

The example involves additional data validation steps that might not be relevant, feel free to delete or add as relevant to your project.

### 5. `scripts/tune.py`
This file is used to define hyperparameter tuning runs.

This is where the 'Datamodule' and' Network' are pulled together and uses optuna to perform the trials.
Starting hyperparameters are included near the top of the file.

### 6. `scripts/train.py`
This file is used to define the training run.

This is where the `Datamodule` and `Network` are pulled together.

The example includes callbacks to retrieve the best model parameters, feel free to delete or add as relevant to your project.

### 7. `Dockerfile`
This dockerfile sets up the Docker image that the MLOps run will utilise.

In the example this is just a simple environment running python version 3.10.
You will most likely need to adapt this for your project.

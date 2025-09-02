# ML Template for Structured Data

A clean, production-ready template for machine learning on tabular data with automated preprocessing, hyperparameter tuning, and comprehensive evaluation.



##  Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `train/config/local_config.cfg`:
```ini
[data]
data_path = data/your_dataset.csv
target_column = your_target

[model]
model_name = xgboost
task = classification
```

### 3. Run
```bash
cd train

# Optional: Hyperparameter tuning
python scripts/tune.py --config config/local_config.cfg

# Train model
python scripts/train.py --config config/local_config.cfg
```

## Project Structure

```
project-template/
├── data/                    # Your datasets
├── train/
│   ├── config/             # Configuration files
│   ├── scripts/            # Training & tuning scripts
│   └── src/                # Core modules
│       ├── DataLoader.py   # Data I/O operations
│       ├── DataModule.py   # Pipeline orchestration
│       └── utils/          # Utilities
├── models/                 # Saved models
└── plots/                  # Generated visualizations
```

## Supported Models
Random Forest, XGBoost, Logistic Regression, Linear Regression

##  Key Configuration Options

```ini
# Data preprocessing
[preprocessing.numerical]
age.imputer = median
age.scaler = standard

[preprocessing.categorical]
category.encoder = onehot
category.encoder_options = {"sparse_output": false}

# Training
[training]
use_optuna = true          # Enable auto-tuning
n_trials = 50             # Optimization trials
use_kfold_cv = true       # Cross-validation
n_splits = 5              # CV folds
```

##  Output

- **Models**: Saved in `models/` with preprocessors & metadata
- **Metrics**: Logged to MLflow (view at `http://localhost:5000`)
- **Plots**: EDA & evaluation charts in `plots/`

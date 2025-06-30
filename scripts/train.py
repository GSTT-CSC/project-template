print("--- RUNNING TRAIN.PY VERSION 2025-06-26-FIX ---")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import joblib
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from project.utils.parse_config import load_config, get_model_and_hyperparams, get_data_config, get_logging_config, get_training_config
from project.DataModule import DataModule
import numpy as np

# New: Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# New: Import optuna
import optuna

# Modified objective function signature to accept fitted_preprocessor_global
def objective(trial, model_class, model_name, task, data_config, X_raw, y, n_splits, shuffle_cv, random_state_cv, fitted_preprocessor_global):
    """
    Optuna objective function to find optimal hyperparameters.
    It trains and evaluates a model using K-Fold Cross-Validation for each trial.
    Uses a pre-fitted preprocessor for efficiency.
    """
    # Define hyperparameter search space based on model_name
    hyperparams = {}
    if model_name == "random_forest":
        hyperparams['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        hyperparams['max_depth'] = trial.suggest_int('max_depth', 5, 20, log=True)
        hyperparams['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        hyperparams['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
        # Add more RF specific hyperparameters if needed
    elif model_name == "xgboost":
        hyperparams['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        hyperparams['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        hyperparams['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        hyperparams['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        # Add more XGBoost specific hyperparameters if needed
    elif model_name == "linear_regression":
        hyperparams['fit_intercept'] = trial.suggest_categorical('fit_intercept', [True, False])
        pass


    # Perform K-Fold Cross-Validation within the objective
    if task == "classification" and data_config.get("stratify", False):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)

    fold_metrics = []

    # REMOVED: No longer create DataModule instance or load data here
    # dm = DataModule(**{k: v for k, v in data_config.items() if k not in ['visualise_data', 'check_imbalance']})
    # dm.load_and_prepare() # This call sets up dm.preprocessor internally.

    for fold, (train_index, test_index) in enumerate(kf.split(X_raw, y)):
        X_train_fold_raw, X_test_fold_raw = X_raw.iloc[train_index], X_raw.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Use the globally fitted preprocessor for transformation
        # IMPORTANT: Only transform, do NOT call fit_transform here to avoid data leakage
        # REMOVED: preprocessor_fold = dm.get_preprocessor()
        X_train_fold_processed = fitted_preprocessor_global.transform(X_train_fold_raw)
        X_test_fold_processed = fitted_preprocessor_global.transform(X_test_fold_raw)

        try:
            # Use the static method for feature names
            processed_feature_names = DataModule.get_feature_names_after_preprocessing(fitted_preprocessor_global, X_raw.columns.tolist())
            X_train_fold_final = pd.DataFrame(X_train_fold_processed, columns=processed_feature_names, index=X_train_fold_raw.index)
            X_test_fold_final = pd.DataFrame(X_test_fold_processed, columns=processed_feature_names, index=X_test_fold_raw.index)
        except Exception as e:
            print(f"Warning: Could not reconstruct DataFrames for fold {fold+1}: {e}. Proceeding with NumPy arrays.")
            X_train_fold_final = pd.DataFrame(X_train_fold_processed)
            X_test_fold_final = pd.DataFrame(X_test_fold_processed)

        model = model_class(**hyperparams) # Initialize model with trial's hyperparameters
        model.fit(X_train_fold_final, y_train_fold) # y_train_fold must be numerical now
        y_pred_fold = model.predict(X_test_fold_final)

        if task == "classification":
            metric = accuracy_score(y_test_fold, y_pred_fold)
        else: # Regression
            metric = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)) # RMSE for regression

        fold_metrics.append(metric)

        # Report intermediate value to Optuna for pruning
        trial.report(metric, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_metric = np.mean(fold_metrics)
    return avg_metric


def main():
    # --- 1. Load Configuration ---
    config_path = "config/local_config.cfg"
    config = load_config(config_path)

    # Get model and hyperparameters (initial, will be overridden by Optuna if used)
    model_class, model_hyperparams_initial = get_model_and_hyperparams(config)
    model_name = config["model"]["model_name"]
    task = config["model"]["task"]

    # Get data configuration
    data_config = get_data_config(config)

    print("\n--- Debugging Data Configuration ---")
    print(f"Categorical columns settings: {data_config.get('categorical_columns_settings')}")
    print(f"Type of categorical_columns_settings: {type(data_config.get('categorical_columns_settings'))}")
    print(f"Numerical columns settings: {data_config.get('numerical_columns_settings')}")
    print(f"Type of numerical_columns_settings: {type(data_config.get('numerical_columns_settings'))}")
    print("------------------------------------")

    # Get logging configuration
    logging_config = get_logging_config(config)
    save_model = logging_config['save_model']
    model_output_path = logging_config['model_output_path']
    mlflow_tracking_uri = logging_config['mlflow_tracking_uri']

    # Get training configuration (including K-Fold CV and Optuna settings)
    training_config = get_training_config(config)
    use_kfold_cv = training_config['use_kfold_cv']
    n_splits = training_config['n_splits']
    shuffle_cv = training_config['shuffle_cv']
    random_state_cv = training_config['random_state_cv']
    use_optuna = training_config['use_optuna']
    n_trials = training_config['n_trials']
    timeout = training_config['timeout']
    optuna_direction = training_config['optuna_direction']


    # --- 2. MLflow Setup ---
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"{model_name}_experiment")
    with mlflow.start_run(run_name=f"{model_name}_training_run"):
        mlflow.log_params({"model_name": model_name, "task": task})

        # Log initial model hyperparameters if Optuna is not used
        if not use_optuna:
            mlflow.log_params(model_hyperparams_initial)

        # Log data configuration
        visualise_data_flag = config['data'].getboolean('visualise_data', False)
        check_imbalance_flag = config['data'].getboolean('check_imbalance', False)

        mlflow.log_params({
            f"data_{k}": v for k, v in data_config.items()
            if k not in ["categorical_columns_settings", "numerical_columns_settings", "visualise_data", "check_imbalance"]
        })
        mlflow.log_params({"data_categorical_settings": data_config.get("categorical_columns_settings", {})})
        mlflow.log_params({"data_numerical_settings": data_config.get("numerical_columns_settings", {})})
        mlflow.log_param("data_visualise_data_flag", visualise_data_flag)
        mlflow.log_param("data_check_imbalance_flag", check_imbalance_flag)

        # Log K-Fold CV configuration
        mlflow.log_params({
            "use_kfold_cv": use_kfold_cv,
            "n_splits": n_splits,
            "shuffle_cv": shuffle_cv,
            "random_state_cv": random_state_cv
        })

        # New: Log Optuna configuration
        mlflow.log_params({
            "use_optuna": use_optuna,
            "n_trials": n_trials,
            "timeout": timeout,
            "optuna_direction": optuna_direction
        })


        # --- 3. Data Loading and Initial Preparation ---
        print("\n--- Data Module Initialization and Initial Preparation ---")

        # --- START OF MODIFIED BLOCK FOR DM_INIT_PARAMS ---
        # Extract the specific settings for preprocessor_settings
        categorical_settings = data_config.get('categorical_columns_settings', {})
        numerical_settings = data_config.get('numerical_columns_settings', {})

        # Build the preprocessor_settings dictionary for DataModule
        preprocessor_settings_for_dm = {}
        if categorical_settings:
            preprocessor_settings_for_dm['categorical'] = categorical_settings
        if numerical_settings:
            preprocessor_settings_for_dm['numerical'] = numerical_settings
        print(f"DEBUG: Attempting to set random_state. data_config has random_state: {data_config.get('random_state', 'NOT FOUND')}")
        # The next line is the one we are fixing:
        #'random_state': data_config.get('random_state', 42), # CORRECTED: Get random_state from data_config
        # Construct dm_init_params explicitly,
        # mapping config keys to DataModule __init__ parameters
        dm_init_params = {
            'data_path': data_config['data_path'],
            'target_column': data_config['target_column'],
            'columns_to_drop': data_config.get('columns_to_drop', []),
            'visualise': data_config.get('visualise_data', False),
            'check_imbalance': data_config.get('check_imbalance', False),
            'test_size': data_config.get('test_size', 0.2),
            'stratify': data_config.get('stratify', False),
            'random_state': data_config.get('random_state', 42), # CORRECTED: Get random_state from data_config
            'preprocessor_settings': preprocessor_settings_for_dm
        }
        # --- END OF MODIFIED BLOCK FOR DM_INIT_PARAMS ---

        dm = DataModule(**dm_init_params)
        #dm.print_column_names() # Assuming this method exists in DataModule

        X_raw, y = dm.load_and_prepare()
        print(f"X_raw columns: {X_raw.columns.tolist()}") # Add this if you want to see columns
        # FIX: Ensure y is a pandas Series for .iloc indexing
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X_raw.index)

        # NEW: Apply LabelEncoder to target variable if it's a classification task
        label_encoder = None
        if task == "classification":
            print("\n--- Encoding Target Variable ---")
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            print(f"Target variable encoded. Original classes: {label_encoder.classes_}")
            print(f"Encoded classes: {np.unique(y)}")
            mlflow.log_param("target_classes_original", label_encoder.classes_.tolist())

        # Call show_data_table AFTER data has been loaded
        #dm.show_data_table()

        # NEW: Fit the preprocessor once on the entire dataset
        fitted_preprocessor_global = dm.create_and_fit_preprocessor(X_raw)

        # --- 4. Call Visualization Methods (when required) ---
        if visualise_data_flag:
            print("\n--- Generating Visualizations ---")
            dm.visualize_column_distributions()
            dm.plot_correlation_heatmap()
            # The original plot_scatter call checked for dm.df.columns. 'df' is not an attribute on DataModule after load_and_prepare separates X,y.
            # You should pass X_raw for plotting or ensure dm.data holds the full DataFrame.
            # Assuming dm.data is updated correctly in DataModule.load_and_prepare()
            if 'Weight' in X_raw.columns and 'Height' in X_raw.columns: # Changed to X_raw.columns
                print("Attempting scatter plot for Weight vs Height (hue by target)...")
                # Pass original y for visualization if you want categorical labels in plots
                dm.plot_scatter(x_col='Weight', y_col='Height', hue_col=data_config['target_column']) # Assumes plot_scatter can handle hue_col as string
            else:
                print("Skipping example scatter plot: 'Weight' or 'Height' not found in data.")

        if check_imbalance_flag:
            print("\n--- Checking Data Imbalance ---")
            # Pass original y for imbalance check visualization
            dm.check_imbalance_and_report(y) # Changed to check_imbalance_and_report(y)

        # --- 5. Hyperparameter Tuning with Optuna or Standard Training ---
        print("\n--- Starting Model Training ---")
        if use_optuna:
            print(f"\n--- Performing Hyperparameter Tuning with Optuna ({n_trials} trials) ---")
            # Create a study object and optimize the objective function
            # Use 'maximize' for accuracy/f1-score, 'minimize' for RMSE/MSE
            study = optuna.create_study(direction=optuna_direction, sampler=optuna.samplers.TPESampler(seed=random_state_cv))

            # Pass arguments needed by the objective function, including the fitted preprocessor
            study.optimize(
                lambda trial: objective(
                    trial, model_class, model_name, task, data_config, X_raw, y, n_splits, shuffle_cv, random_state_cv, fitted_preprocessor_global # New argument
                ),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True # Shows progress bar during optimization
            )

            print("\n--- Optuna Optimization Complete ---")
            print(f"Number of finished trials: {len(study.trials)}")
            print(f"Best trial parameters: {study.best_params}")
            print(f"Best trial value: {study.best_value:.4f}")

            # Log best Optuna results to MLflow
            mlflow.log_params({"optuna_best_value": study.best_value})
            mlflow.log_params({f"optuna_best_param_{k}": v for k, v in study.best_params.items()})

            # Train the final model with the best hyperparameters found by Optuna on the full dataset
            print("\n--- Training Final Model with Best Optuna Hyperparameters ---")
            final_model_hyperparams = study.best_params
            final_model = model_class(**final_model_hyperparams)

            # Preprocess the full dataset before final training using the global fitted preprocessor
            X_processed_final = fitted_preprocessor_global.transform(X_raw)

            try:
                processed_feature_names = DataModule.get_feature_names_after_preprocessing(fitted_preprocessor_global, X_raw.columns.tolist())
                X_final_df = pd.DataFrame(X_processed_final, columns=processed_feature_names, index=X_raw.index)
            except Exception as e:
                print(f"Warning: Could not reconstruct DataFrame for final training: {e}. Proceeding with NumPy array.")
                X_final_df = pd.DataFrame(X_processed_final)

            final_model.fit(X_final_df, y) # y is now encoded
            print("Final model trained on full dataset with best Optuna hyperparameters.")

            # Log the final model
            if save_model:
                os.makedirs(model_output_path, exist_ok=True)
                model_path = os.path.join(model_output_path, f"{model_name}_{task}_optuna_best.pkl")
                joblib.dump(final_model, model_path)
                mlflow.log_artifact(model_path)
                print(f"Best Optuna model saved to {model_path}")
            else:
                print("Model saving skipped as per configuration.")

            if task == "classification":
                mlflow.log_metric("final_model_avg_accuracy", study.best_value)
            else:
                mlflow.log_metric("final_model_avg_rmse", study.best_value)


        elif use_kfold_cv: # K-Fold CV without Optuna
            print(f"Performing {n_splits}-Fold Cross-Validation...")

            if task == "classification" and data_config.get("stratify", False):
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)
                print("Using Stratified K-Fold Cross-Validation.")
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state_cv)
                print("Using K-Fold Cross-Validation.")

            fold_metrics = []
            for fold, (train_index, test_index) in enumerate(kf.split(X_raw, y)):
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
                X_train_fold_raw, X_test_fold_raw = X_raw.iloc[train_index], X_raw.iloc[test_index]
                y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

                # Use the globally fitted preprocessor for transformation
                X_train_fold_processed = fitted_preprocessor_global.transform(X_train_fold_raw)
                X_test_fold_processed = fitted_preprocessor_global.transform(X_test_fold_raw)

                try:
                    processed_feature_names = DataModule.get_feature_names_after_preprocessing(fitted_preprocessor_global, X_raw.columns.tolist())
                    X_train_fold_final = pd.DataFrame(X_train_fold_processed, columns=processed_feature_names, index=X_train_fold_raw.index)
                    X_test_fold_final = pd.DataFrame(X_test_fold_processed, columns=processed_feature_names, index=test_index)
                except Exception as e:
                    print(f"Warning: Could not reconstruct DataFrames for fold {fold+1}: {e}. Proceeding with NumPy arrays.")
                    X_train_fold_final = pd.DataFrame(X_train_fold_processed)
                    X_test_fold_final = pd.DataFrame(X_test_fold_processed)


                model = model_class(**model_hyperparams_initial) # Use initial hyperparams for K-Fold CV without Optuna
                model.fit(X_train_fold_final, y_train_fold) # y_train_fold is now encoded
                y_pred_fold = model.predict(X_test_fold_final)

                if task == "classification":
                    acc = accuracy_score(y_test_fold, y_pred_fold)
                    mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)
                    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
                    fold_metrics.append(acc)
                else: # Regression
                    mse = mean_squared_error(y_test_fold, y_pred_fold)
                    rmse = np.sqrt(mse)
                    mlflow.log_metric(f"fold_{fold+1}_mse", mse)
                    mlflow.log_metric(f"fold_{fold+1}_rmse", rmse)
                    print(f"Fold {fold+1} MSE: {mse:.4f}")
                    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
                    fold_metrics.append(rmse)

            avg_metric = np.mean(fold_metrics)
            if task == "classification":
                mlflow.log_metric("avg_accuracy", avg_metric)
                print(f"\nAverage Accuracy across {n_splits} folds: {avg_metric:.4f}")
            else:
                mlflow.log_metric("avg_rmse", avg_metric)
                print(f"\nAverage RMSE across {n_splits} folds: {avg_metric:.4f}")

            if save_model:
                print("Model saving skipped for K-Fold Cross-Validation setup without Optuna.")

        else: # Original single train-test split flow (without K-Fold or Optuna)
            print("Performing single Train-Test Split training...")

            print("\n--- Performing Train-Test Split ---")
            X_train_raw, X_test_raw, y_train, y_test = dm.perform_train_test_split(X_raw, y) # y is now encoded
            print(f"X_train shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")

            print("\n--- Applying Preprocessing ---")
            # Use the globally fitted preprocessor
            preprocessor = fitted_preprocessor_global

            X_train_processed = preprocessor.transform(X_train_raw)
            X_test_processed = preprocessor.transform(X_test_raw)

            try:
                processed_feature_names = DataModule.get_feature_names_after_preprocessing(preprocessor, X_raw.columns.tolist())
                X_train_final = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train_raw.index)
                X_test_final = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test_raw.index)
                print("Processed data converted back to DataFrames with feature names.")
            except Exception as e:
                print(f"Warning: Could not reconstruct DataFrames with feature names after preprocessing: {e}")
                print("Proceeding with NumPy arrays for model training, which might affect feature importance interpretation later.")
                X_train_final = pd.DataFrame(X_train_processed)
                X_test_final = pd.DataFrame(X_test_processed)

            print(f"Processed X_train shape: {X_train_final.shape}")
            print(f"Processed X_test shape: {X_test_final.shape}")


            print(f"\n--- Training {model_name} Model ---")
            model = model_class(**model_hyperparams_initial) # Use initial hyperparams for single split
            model.fit(X_train_final, y_train) # y_train is now encoded
            print(f"{model_name} model trained successfully.")

            print("\n--- Evaluating Model Performance ---")
            y_pred = model.predict(X_test_final)

            if task == "classification":
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", acc)
                print("Classification Report:")
                # Use original labels for report
                print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
            else: # Regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                print(f"Mean Squared Error (MSE): {mse:.4f}")
                print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

            if save_model:
                os.makedirs(model_output_path, exist_ok=True)
                model_path = os.path.join(model_output_path, f"{model_name}_{task}.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                print(f"Model saved to {model_path}")
            else:
                print("Model saving skipped as per configuration.")

        mlflow.end_run()

if __name__ == "__main__":
    main()
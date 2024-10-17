import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from project_NLP.NLPDataset import NLPDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import mlflow


class NLPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        batch_size=8,
        max_token_len=256,
        num_workers=0,
        fold_indices=None,
        random_state=1,
    ):
        """Initialize the DataModule. Set batch size, data path, tokenizer,
        maximum token length, label columns, sample status and number of workers.

        Args:
            data_path (str): Path to the data file.
            tokenizer: Tokenizer to be used.
            batch_size (int, optional): Size of the data batches. Default is 8.
            max_token_len (int, optional): Maximum length of tokens. Default is 256.
            num_workers (int, optional): Number of workers. Default is 0.
            sample (bool, optional): If True, use WeightedRandomSampler. Default is False.
        """

        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.fold_indices = fold_indices
        self.random_state = random_state
        self.df = None
        self.train_df = None
        self.val_df = None
        self.label_columns = None
        self.train_dataset = None
        self.val_dataset = None
        self.read_csv_data()

    def setup(self, stage=None):
        """
        Set up the data module. Parse the data and create train and validation datasets.
        """
        self.parse_df_data()
        if self.fold_indices:
            train_indices, val_indices = self.fold_indices
            self.train_df = self.df.iloc[train_indices]
            self.val_df = self.df.iloc[val_indices]
        else:
            self.default_train_val_split()

        self.train_dataset = NLPDataset(
            self.tokenizer,
            self.train_df,
            self.label_columns,
            self.max_token_len,
        )

        self.val_dataset = NLPDataset(
            self.tokenizer,
            self.val_df,
            self.label_columns,
            self.max_token_len,
        )

    def train_dataloader(self):
        """
        Create and return a data loader for the training data.

        Returns:
            DataLoader: Data loader for the training data.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Create and return a data loader for the validation data.

        Returns:
            DataLoader: Data loader for the validation data.
        """

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def read_csv_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()

    def parse_df_data(self):
        """
        Parse the data. Create label dictionary and add 'label' and 'data_type' columns.
        Split the data into train and validation dataframes. Store the label columns in a pickle file.
        """

        self.label_columns = self.df.ClassLabel.unique()
        self.label_dict = {}
        for index, possible_label in enumerate(self.label_columns):
            self.label_dict[possible_label] = index
            self.df[possible_label] = 0
            self.df.loc[self.df["ClassLabel"] == possible_label, [possible_label]] = 1

        self.df["label"] = self.df.ClassLabel.replace(self.label_dict)

        self.length_label_dict = len(self.label_dict)
        self.num_labels = self.length_label_dict
        self.num_classes = len(list(set(self.df.label)))

        with open("label_columns.data", "wb") as filehandle:
            pickle.dump(self.label_columns, filehandle)

        class_counts = self.df["ClassLabel"].value_counts()
        for class_label, count in class_counts.items():
            mlflow.log_param(f"class_{class_label}_count", count)

    def default_train_val_split(self):
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.df["label"],
        )
        self.train_df = train_df
        self.val_df = val_df

    def steps_per_epoch(self):
        """
        Calculate and return the number of steps per epoch based on the batch size.

        Returns:
            int: Number of steps per epoch.
        """
        if self.train_df is None or len(self.train_df) == 0:
            self.parse_df_data()
            self.default_train_val_split()
        return len(self.train_df) // self.batch_size

    def dataset_stats(self, dataset) -> dict:
        """
        Calculate and return a dictionary of dataset statistics including label distribution and number of samples.

        Args:
            dataset (pandas.DataFrame): The dataset to analyze.

        Returns:
            dict: Dictionary of dataset statistics.
        """
        stats = {}
        stats["n_samples"] = len(dataset)
        label_counts = dataset["ClassLabel"].value_counts()
        label_counts_dict = label_counts.to_dict()
        stats["label_counts"] = label_counts_dict
        return stats

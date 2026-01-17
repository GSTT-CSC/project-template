import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional
from transformers import BertTokenizerFast as BertTokenizer

import configparser


class NLPDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        data: Optional[pd.DataFrame] = None,
        label_columns: list = None,
        max_token_len: int = 256,
    ):
        if tokenizer == None:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = label_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Given an index, return a dictionary containing 'diag_final', 'input_ids', 'attention_mask' and 'labels'
        after encoding a row of the data.

        Args:
            index (int): Index of the data row.

        Returns:
            dict: A dictionary containing 'diag_final', 'input_ids', 'attention_mask' and 'labels'
            for the data row at the given index.
        """

        data_row = self.data.iloc[index]
        diag_final = [data_row["TextString"]]
        labels = data_row[self.label_columns]  # nth index is nth class
        encoding = self.encoder(diag_final)
        return dict(
            diag_final=diag_final,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )

    def encoder(self, diag_final):
        """
        Encode a given list of text strings using the tokenizer set during initialization.
        Args:
            diag_final (list): List of text strings to be encoded.

        Returns:
            dict: A dictionary containing the following keys:
                - 'input_ids': Tensor of token ids obtained from the text strings.
                - 'attention_mask': Tensor where positions with original tokens are represented by 1 and positions with
                padding are represented by 0.
        """

        return self.tokenizer.batch_encode_plus(
            diag_final,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

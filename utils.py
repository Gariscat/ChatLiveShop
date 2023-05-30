"""
    Reference: https://colab.research.google.com/drive/1lEyFhEfoc-5Z5xqpEKkZt_iMaojH1MP_?usp=sharing#scrollTo=o1rLCBtBXhPa
"""


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import time
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def convert_json_to_csv(source_file, target_file):
    lines = open(source_file,'r').readlines()
    print("length of lines:", len(lines))
    input_list = []
    output_list = []
    answer_choices_list = []
    type_list = []

    for i, line in enumerate(lines):
        json_string = json.loads(line.strip())
        input_ = json_string["input"].replace("\n", "_")
        output_ = json_string["target"]
        answer_choices_ = json_string.get("answer_choices",[])
        type_ = json_string["type"]
 
        input_list.append(input_)
        output_list.append(output_)
        answer_choices_list.append(answer_choices_)
        type_list.append(type_)

    df = pd.DataFrame({'input': input_list,
                       'target':output_list,
                       'answer_choices': answer_choices_list,
                       'type': type_list,
                       })
    
    df.to_csv(target_file, index=False)


class YourDataSetClass(Dataset):
    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.long(),
            "source_mask": source_mask.long(),
            "target_ids": target_ids.long(),
            "target_mask": target_mask.long(),
            "target_ids_y": target_ids.long(),
        }


if __name__ == '__main__':
    source_file='~/data/train.json'
    target_file='~/data/train.csv'
    convert_json_to_csv(source_file, target_file)
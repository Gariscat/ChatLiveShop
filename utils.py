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
from transformers import AutoTokenizer, AutoModel
from torch import cuda

from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    # console.print(table) # TODO TODO TODO 

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

device = 'cuda' if cuda.is_available() else 'cpu'

def convert_json_to_csv(source_file, target_file, num_items=None):
    if num_items is None:
        lines = open(source_file, 'r').readlines()
    else:
        lines = open(source_file, 'r').readlines()[:num_items]
    ### lines = open(source_file, 'r').readlines()
    print("length of lines:", len(lines))
    input_list = []
    output_list = []
    answer_choices_list = []
    type_list = []

    for i, line in enumerate(lines):
        json_string = json.loads(line.strip())
        input_ = json_string["input"].replace("\n", "_")
        output_ = json_string["target"]
        """answer_choices_ = json_string.get("answer_choices",[])
        type_ = json_string["type"]"""
 
        input_list.append(input_)
        output_list.append(output_)
        """answer_choices_list.append(answer_choices_)
        type_list.append(type_)"""

        """if i < 5:
            print(input_, output_)"""

    df = pd.DataFrame({'input': input_list,
                       'target':output_list,
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


def validate(epoch, tokenizer, model, device, loader, max_length,
    config={
        'num_beams': 2,
        'repetition_penalty': 2.5,
        'length_penalty': 1.0,
        'early_stopping': True,  
    },
):
    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device).long()
            ids = data['source_ids'].to(device).long()
            mask = data['source_mask'].to(device).long()

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask, 
                max_length=max_length, 
                num_beams=config['num_beams'],
                repetition_penalty=config['repetition_penalty'],
                length_penalty=config['length_penalty'], 
                early_stopping=config['early_stopping']
            )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _ % 1000 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


"""
if __name__ == '__main__':
    source_file='/root/data/train.json'
    target_file='/root/data/train.csv'
    convert_json_to_csv(source_file, target_file)
"""

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
import wandb
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

def convert_json_to_csv(source_file, target_file):
    lines = open(source_file, 'r').readlines()
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

        if i < 5:
            print(input_, output_)

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


def train(epoch, tokenizer, model, device, loader, optimizer, online_logger=wandb):
    """
    Function to be called for training with the parameters passed from main function
    """
    model.train()
    time1 = time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device).long()
        y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device).long() # input. e.g. "how are you?"
        mask = data["source_mask"].to(device).long()

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0].item()

        if online_logger:
            online_logger.log({'step': _, 'loss': loss})

        if _ % 100 == 0:
            time2 = time.time()
            print(_,"epoch:"+str(epoch)+"-loss:"+str(loss)+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))
            # console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


def T5Trainer(
    dataframe, source_text, target_text, model_params, train_size=0.95, output_dir="./outputs/"
):
    """
    T5 trainer
    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using ChatYuan model and added a Language model layer on top for generation of prediction.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = AutoModel.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")
    total_train_steps=int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"])/model_params["TRAIN_BATCH_SIZE"])
    console.print(f"Total Train Steps: {total_train_steps}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        # 1) train for one epoch
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        
        # 2) save model for each epoch
        console.log(f"[Saving Model]...\n")
        path = os.path.join(output_dir, "model_files")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        # 3) evaluating test dataset
        console.log(f"[Initiating Validation]...\n")
        with torch.no_grad(): # add 2022.10.4
            #for epoch in range(model_params["VAL_EPOCHS"]):
            predictions, actuals = validate(epoch, tokenizer, model, device, val_loader,model_params["MAX_TARGET_TEXT_LENGTH"])
            final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
            final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

"""
if __name__ == '__main__':
    source_file='/root/data/train.json'
    target_file='/root/data/train.csv'
    convert_json_to_csv(source_file, target_file)
"""
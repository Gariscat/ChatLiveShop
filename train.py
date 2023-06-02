from utils import *
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb
import deepspeed
import argparse

def train_epoch(epoch, tokenizer, device, loader, optimizer, model_engine, online_logger=None):
    """
    Function to be called for training with the parameters passed from main function
    """
    time1 = time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device).long()
        y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach() # target, from second to end.e.g."好吗？<EOS>"
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device).long() # input. e.g. "how are you?"
        mask = data["source_mask"].to(device).long()

        outputs = model_engine(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if online_logger:
            online_logger.log({'step': _, 'loss': loss.item()})

        if _ % 100 == 0:
            time2 = time.time()
            print(_,"epoch:"+str(epoch)+"-loss:"+str(loss.item())+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))
            # console.print(training_logger)

        ### optimizer.zero_grad()
        ### loss.backward()
        ### optimizer.step()
        model_engine.backward(loss)
        model_engine.step()


def train_main(
    dataframe,
    source_label,
    target_label,
    model_params,
    train_size=0.95,
    output_dir="./outputs/",
    cmd_args=None,
    online_logger=None,
):
    """
    T5 trainer
    """
    os.makedirs(output_dir, exist_ok=True)
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using ChatYuan model and added a Language model layer on top for generation of prediction.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    ### model = model.to(device)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=cmd_args,
        model=model,
        model_parameters=model.parameters(),
        dist_init_required=True,
    )

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_label, target_label]]
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
        source_label,
        target_label,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_label,
        target_label,
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
        train_epoch(epoch, tokenizer, device, training_loader, optimizer, model_engine, online_logger=online_logger)
        
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--fp', type=str, default="train.json",)
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    model_params = {
        "MODEL": "/root/ChatYuan-large-v2",  # model_type
        "TRAIN_BATCH_SIZE": 8,  # training batch size, 8
        "VALID_BATCH_SIZE": 8,  # validation batch size,8 
        "TRAIN_EPOCHS": 1,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
        "SEED": 42,  # set seed for reproducibility
    }

    wandb.init(
        project='ChatLiveShop',
        entity='gariscat',
        config=model_params
    )

    prefix = cmd_args.fp[:cmd_args.fp.index('.')]
    source_file=f'./data/{cmd_args.fp}'
    target_file=f'./data/{prefix}'+'.csv'
    convert_json_to_csv(source_file, target_file, num_items=10000)
    df = pd.read_csv('./data/sample.csv', engine='python')
    print("df.head:", df.head(n=5))
    print("df.shape:", df.shape)
    
    train_main(
        dataframe=df,
        source_label="input",
        target_label="target",
        model_params=model_params,
        output_dir="outputs",
        cmd_args=cmd_args,
        online_logger=wandb,
    )

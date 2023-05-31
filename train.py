from utils import *
import wandb


def trainer(
    dataframe, source_text, target_text, model_params, train_size=0.95, output_dir="./outputs/", online_logger=None
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
    print('Using:', model_params["MODEL"])
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
        train(epoch, tokenizer, model, device, training_loader, optimizer, online_logger=online_logger)
        
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
    model_params = {
        "MODEL": "ClueAI/ChatYuan-large-v2",  # model_type
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

    source_file='/root/data/train.json'
    target_file='/root/data/train.csv'
    convert_json_to_csv(source_file, target_file)
    df = pd.read_csv('/root/data/train.csv')
    print("df.head:", df.head(n=5))
    print("df.shape:", df.shape)
    
    trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="outputs",
        online_logger=wandb,
    )
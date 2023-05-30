from utils import *

if __name__ == '__main__':
    model_params = {
        "MODEL": "ClueAI/ChatYuan-large-v1",  # model_type
        "TRAIN_BATCH_SIZE": 8,  # training batch size, 8
        "VALID_BATCH_SIZE": 8,  # validation batch size,8 
        "TRAIN_EPOCHS": 1,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
        "SEED": 42,  # set seed for reproducibility
    }

    source_file='/root/data/train.json'
    target_file='/root/data/train.csv'
    convert_json_to_csv(source_file, target_file)
    df = pd.read_csv('/root/data/train.csv')
    print("df.head:", df.head(n=5))
    print("df.shape:", df.shape)
    
    T5Trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="outputs",
    )
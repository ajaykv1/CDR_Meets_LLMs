import pandas as pd
from datasets import Dataset,DatasetDict


def load_train_val_data(k_shot, source, target, data_info, rating_ranking, injection, prompt_context):
    # train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_{rating_ranking}_{injection}_injection.csv")
    # validation_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_{rating_ranking}_{injection}_injection.csv")
    test_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_{rating_ranking}_{injection}_injection.csv")
    
    fineune_dataset_dict = DatasetDict({
        # "train": Dataset.from_pandas(train_df),
        # "validation": Dataset.from_pandas(validation_df),
        "test": Dataset.from_pandas(test_df)
    })

    return fineune_dataset_dict












    
import random
import pandas as pd

def load_data(dataset, data_info):
    if data_info == 'amazon':
        data = pd.read_json(f'../dataset/reviews_{dataset}_5.json', lines=True)
        meta = pd.read_json(f'../dataset/meta_{dataset}.json', lines=True)

        data_columns = ['reviewerID', 'asin', 'overall']
        meta_columns = ['asin', 'title']

        data = data[data_columns]
        meta = meta[meta_columns]

        merged_df = pd.merge(data, meta, on='asin')

        return merged_df

def overlapping_users_df(source_data, target_data, k_shot):
    overlapping_users = set(source_data['reviewerID']).intersection(target_data['reviewerID'])
    num_users_to_keep = int(len(overlapping_users) * k_shot / 100)
    sampled_users = random.sample(overlapping_users, num_users_to_keep)

    source_df = source_data[source_data['reviewerID'].isin(overlapping_users)]
    target_df = target_data[target_data['reviewerID'].isin(overlapping_users)]

    return source_df, target_df, sampled_users, overlapping_users

def train_test_split(k_shot, source, target, data_info, prompt_context):
    df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}.csv")

    # Sort the data by the 'overall' rating (or another criteria if needed)
    df = df.sort_values(by=['reviewerID', 'overall'], ascending=[True, False])

    # Splitting the data
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    validation_data = pd.DataFrame()

    # For each user, put the last interaction in the test set, the second-to-last in the validation set, and all others in the training set
    for user, group in df.groupby('reviewerID'):
        if len(group) > 1:
            train_data = pd.concat([train_data, group.iloc[:-2]]) # All but last two
            validation_data = pd.concat([validation_data, group.iloc[-2:-1].reset_index(drop=True)]) # Second-to-last
            test_data = pd.concat([test_data, group.iloc[-1:].reset_index(drop=True)]) # Last one
        else:
            # For users with only one interaction, put it in the test set
            test_data = pd.concat([test_data, group])

    # Save the train, validation, and test data
    train_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv", index=False)
    validation_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv", index=False)
    test_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv", index=False)

def partition_data(source, target, k_shot, data_info, prompt_context):
    source_data = load_data(source, data_info)
    target_data = load_data(target, data_info)

    source_df, target_df, sampled_users, overlapping_users = overlapping_users_df(source_data, target_data, int(k_shot))
    
    print("---------- Data Statistics ----------\n")
    print(f"Source Dataframe:\n\n{source_df}\n")
    print(f"Target Dataframe:\n\n{target_df}\n")
    print(f"{k_shot}% of Sampled Users: {len(sampled_users)}")
    print(f"Total number of overlapping Users: {len(overlapping_users)}")

    source_df.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv", index=False)
    target_df.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}.csv", index=False)   

    train_test_split(k_shot, source, target, data_info, prompt_context)

    return sampled_users







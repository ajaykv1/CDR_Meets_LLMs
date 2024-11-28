import pandas as pd
import random

NL = "\n\n"
Q = "\'"

def ranking_with_target_injection_train_dataset(k_shot, source, target, data_info, neg_samples, sampled_users, prompt_context):
    train_df = train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    
    train_df = train_df[train_df['reviewerID'].isin(sampled_users)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users)]

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in train_df.iterrows():
        
        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" + 
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source and target domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_ranking_with_injection.csv", index=False)   

def ranking_no_target_injection_train_dataset(k_shot, source, target, data_info, neg_samples, sampled_users, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    
    train_df = train_df[train_df['reviewerID'].isin(sampled_users)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users)]

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in train_df.iterrows():

        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create no target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source domain. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_ranking_no_injection.csv", index=False)   

def ranking_with_target_injection_validation_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    train_df = train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in validation_df.iterrows():
        
        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source and target domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_ranking_with_injection.csv", index=False)   

def ranking_no_target_injection_validation_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in validation_df.iterrows():

        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create no target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source domain. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_ranking_no_injection.csv", index=False)   

def ranking_with_target_injection_test_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    train_df = train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in test_df.iterrows():
        
        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" + 
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source and target domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_ranking_with_injection.csv", index=False)   

def ranking_no_target_injection_test_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Create an empty dataframe to store prompts and correct rankings
    ranking_data = pd.DataFrame(columns=['prompt', 'correct_ranking'])

    for index, row in test_df.iterrows():

        user = row['reviewerID']
        interacted_item = row['title']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != interacted_item)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Generate a list of candidate items. This includes 1 item the user has interacted with and n other items.
        non_interacted_items = train_df[(train_df['reviewerID'] != user) & (train_df['title'] != interacted_item)]['title'].sample(neg_samples).tolist()
        candidate_items = [interacted_item] + non_interacted_items

        # Shuffle the candidate items
        random.shuffle(candidate_items)

        # Perfect ranking should have the interacted item first
        perfect_ranking = [interacted_item] + non_interacted_items

        items_format = ', '.join([f'Item{i+1}' for i in range(neg_samples+1)])

        # Create no target behavior injection prompt for ranking
        ranking_prompt = ""

        if prompt_context == 'none':

            print("None")

        elif prompt_context == 'medium':

            ranking_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "This is a list of candidate items in the target domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user’s past interactions in the source domain. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        elif prompt_context == 'high':

            ranking_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the list of candidate items in the {target} domain: " +
                f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user’s past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don’t make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
            )

        # Store the prompts and ground truth rankings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [ranking_prompt],
            'correct_ranking': [perfect_ranking]
        })

        # Use concat to combine the data
        ranking_data = pd.concat([ranking_data, new_data], ignore_index=True)

    ranking_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_ranking_no_injection.csv", index=False)   






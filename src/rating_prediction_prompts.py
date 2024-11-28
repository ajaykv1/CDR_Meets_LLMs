import pandas as pd

def transform_data(rating):

    candidate_item_rating = ''

    if rating >= 0 and rating < 1.0:
        candidate_item_rating = 'Very Unlikely'

    if rating >= 1.0 and rating < 2.0:
        candidate_item_rating = 'Unlikely'

    if rating >= 2.0 and rating < 3.0:
        candidate_item_rating = 'Somewhat Unlikely'

    if rating >= 3.0 and rating < 4.0:
        candidate_item_rating = 'Neutral'

    if rating >= 4.0 and rating < 5.0:
        candidate_item_rating = 'Likely'

    if rating >= 5.0:
        candidate_item_rating = 'Highly Likely'

    return candidate_item_rating

def rating_with_target_injection_train_dataset(k_shot, source, target, data_info, sampled_users, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")

    train_df = train_df[train_df['reviewerID'].isin(sampled_users)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users)]

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in train_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = transform_data(int(row['overall'])) # row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create target behavior injection prompt for rating prediction
        target_injection_prompt = ""

        if prompt_context == 'none':

            print("Nothing")

        elif prompt_context == 'medium':

            target_injection_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" + 
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source and target domains? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain::\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_rating_with_injection.csv", index=False)   

def rating_no_target_injection_train_dataset(k_shot, source, target, data_info, sampled_users, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    
    train_df = train_df[train_df['reviewerID'].isin(sampled_users)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users)]

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in train_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create no target behavior injection prompt for rating prediction
        no_target_injection_prompt = ""

        if prompt_context == 'none':
            print("Nothing")

        elif prompt_context == 'medium':

            no_target_injection_prompt = (
                "Here is a user’s source domain ratings \n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" +
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source domain? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            no_target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                f"Here is a user’s rating history:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [no_target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_rating_no_injection.csv", index=False)   

def rating_with_target_injection_validation_dataset(k_shot, source, target, data_info, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in validation_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create target behavior injection prompt for rating prediction
        target_injection_prompt = ""

        if prompt_context == 'none':
            print("Nothing")

        elif prompt_context == 'medium':

            target_injection_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" + 
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source and target domains? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain::\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_rating_with_injection.csv", index=False)   

def rating_no_target_injection_validation_dataset(k_shot, source, target, data_info, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in validation_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create no target behavior injection prompt for rating prediction
        no_target_injection_prompt = ""

        if prompt_context == 'none':
            print("Nothing")

        elif prompt_context == 'medium':

            no_target_injection_prompt = (
                "Here is a user’s source domain ratings \n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" +
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source domain? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            no_target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                f"Here is a user’s rating history:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [no_target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_rating_no_injection.csv", index=False)   

def rating_with_target_injection_test_dataset(k_shot, source, target, data_info, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in test_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create target behavior injection prompt for rating prediction
        target_injection_prompt = ""

        if prompt_context == 'none':
            print("Nothing")

        elif prompt_context == 'medium':

            target_injection_prompt = (
                "Here is a user’s rating history in the source domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                "Here is a user’s rating history in the target domain:\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" + 
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source and target domains? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user’s rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"Here is the same user’s rating history in the {target} domain::\n\n" +
                "\n".join(target_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_rating_with_injection.csv", index=False)   

def rating_no_target_injection_test_dataset(k_shot, source, target, data_info, prompt_context):
    train_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Create an empty dataframe to store prompts and ground truth ratings
    fine_tuning_data = pd.DataFrame(columns=['prompt', 'ground_truth'])

    for index, row in test_df.iterrows():
        
        user = row['reviewerID']
        candidate_item_title = row['title']
        candidate_item_rating = row['overall']

        # Retrieve user's rating history in source domain, limit to 10 items
        user_source_data = source_data[source_data['reviewerID'] == user].head(10)
        source_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_source_data['title'], user_source_data['overall'])]

        # Retrieve user's rating history in target domain, limit to 10 items
        user_target_data = train_df[(train_df['reviewerID'] == user) & (train_df['title'] != candidate_item_title)].head(10)
        target_items_formatted = [f"title: {title}, rating: {rating}" for title, rating in zip(user_target_data['title'], user_target_data['overall'])]

        # Create no target behavior injection prompt for rating prediction
        no_target_injection_prompt = ""

        if prompt_context == 'none':
            print("Nothing")

        elif prompt_context == 'medium':

            no_target_injection_prompt = (
                "Here is a user’s source domain ratings \n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"The item in the target domain is: {candidate_item_title}\n\n" +
                "How likely is the user to interact with the candidate item in the target domain based on their rating behavior in the source domain? Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        elif prompt_context == 'high':

            no_target_injection_prompt = (
                f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source}, which means that this domain consists of items related to {source}. The target domain is {target}.  Below is the user’s rating history in only the {source} domain, where you will see the ratings that the user gave to items. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.  \n\n" + 
                f"Here is a user’s rating history in the {source} domain:\n\n" +
                f"Here is a user’s rating history:\n\n" +
                "\n".join(source_items_formatted) + "\n\n" +
                f"This is the candidate item in the {target} domain: {candidate_item_title}\n\n" + 
                f"You need to infer the user’s preferences in the target domain ({target}) based on their rating information in the {source} domain in order to assess the likelihood of the user interacting with the candidate item in the {target} domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Don’t provide any explanation, and only output one of the options listed. Do not say anything else. "
            )

        # Store the prompts and ground truth ratings in the fine-tuning dataframe
        new_data = pd.DataFrame({
            'prompt': [no_target_injection_prompt],
            'ground_truth': [candidate_item_rating]
        })

        # Use concat to combine the data
        fine_tuning_data = pd.concat([fine_tuning_data, new_data], ignore_index=True)

    fine_tuning_data.to_csv(f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_rating_no_injection.csv", index=False)   




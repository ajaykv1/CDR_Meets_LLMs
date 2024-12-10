import os
import re
import math
import torch
from math import sqrt
from rating_prediction_prompts import transform_data
from fine_tune import load_train_val_data
from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM

def evaluate_llm(k_shot, source, target, data_info, model_name, rating_ranking, injection, prompt_context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fineune_dataset_dict = load_train_val_data(k_shot, source, target, data_info, rating_ranking, injection, prompt_context)
    test_data = fineune_dataset_dict['test'].to_pandas()

    inputs = test_data['prompt'].values
    inputs = inputs[:1000]
    ground_truths = None

    if rating_ranking == 'rating':
        ground_truths = test_data['ground_truth'].values
        ground_truths = ground_truths[:1000]

    elif rating_ranking == 'ranking':
        gt = test_data['correct_ranking'].values
        gt = gt[:1000]
        ground_truths = []
        for i in gt:
            i = i.replace(", nan,", ", 'NA',")
            ranked_list = eval(i)
            ground_truths.append(str(ranked_list[0]))


    count = 0
    base_model = None
    base_tokenizer = None

    if model_name.lower() in ['7b', '13b']:

        # Quantization configuration for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4', 
            bnb_4bit_compute_dtype=torch.float16
        )


        base_model = AutoModelForCausalLM.from_pretrained(
                f"meta-llama/Llama-2-{model_name.lower()}-chat-hf",
                return_dict=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
        base_tokenizer = AutoTokenizer.from_pretrained(
            f"meta-llama/Llama-2-{model_name.lower()}-chat-hf"
        )

    elif model_name.lower() in ['8b']:

        # Quantization configuration for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
                f"meta-llama/Meta-Llama-3-{model_name}-Instruct",
                return_dict=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
        base_tokenizer = AutoTokenizer.from_pretrained(
            f"meta-llama/Meta-Llama-3-{model_name}-Instruct",
        )
    
    best_mae = 1000
    best_rmse = 1000

    worst_mae = 0
    worst_rmse = 0

    best_mrr_5 = float('-inf')
    worst_mrr_5 = float('inf')
    best_ndcg_5 = float('-inf')
    worst_ndcg_5 = float('inf')

    best_mrr_10 = float('-inf')
    worst_mrr_10 = float('inf')
    best_ndcg_10 = float('-inf')
    worst_ndcg_10 = float('inf')

    for i in range(2):

        print(f"\n\n Iteration {i+1}/2\n\n")

        base_preds = []
        fine_tuned_preds = []

        for prompt in inputs:
            if prompt == None:
                continue

            system_prompt = ""
            input_with_label = ""
            input_with_label_base = ""

            if model_name.lower() in ['7b', '13b']:

                if rating_ranking == 'rating':

                    if prompt_context == 'none':
                        system_prompt = "<s> [INST]<</SYS>>"

                    elif prompt_context == 'medium':
                        
                        if injection == 'with':
                            system_prompt = "<s> [INST] <<SYS>>Output an option from the prompt.<</SYS>>"
                            
                        if injection == 'no':
                            system_prompt = "<s> [INST] <<SYS>>Output an option from the prompt.<</SYS>>"

                    elif prompt_context == 'high':
                        
                        if injection == 'with':
                            system_prompt = "<s> [INST] <<SYS>>You are a cross-domain recommender system. Analyze user behavior in a source domain and transfer that knowledge to make a recommendation in a target domain. If both domains are available, prioritize the target domain but consider the source domain as well. Respond with one of the following: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item.<</SYS>>"
                        
                        if injection == 'no':
                            system_prompt = "<s> [INST] <<SYS>>You are a cross-domain recommender system. Analyze user behavior in a source domain and transfer that knowledge to make a recommendation in a target domain. If both domains are available, prioritize the target domain but consider the source domain as well. Respond with one of the following: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item.<</SYS>>"

                elif rating_ranking == 'ranking':

                    if prompt_context == 'none':
                        system_prompt = "<s> [INST] <<SYS>>You will output a list<</SYS>>"

                    elif prompt_context == 'medium':
                        system_prompt = "<s> [INST] <<SYS>>You will output a ranked list, where items will be ranked from most likely to interact with to least likely to interact with. <</SYS>>"

                    elif prompt_context == 'high':
                        system_prompt = "<s> [INST] <<SYS>>You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. You will output a ranked list of candidate items. The first item in the list should be the item that the user will most likeley interact with, and the last item in the list should be the item the user will be least likely to interact with. You should follow the expected format in the prompt. <</SYS>>"


                input_with_label = "[INST] " + str(prompt) + " [/INST] "
                input_with_label_base = str(system_prompt) + str(prompt) + " [/INST] "

            if model_name.lower() in ['8b']:

                if rating_ranking == 'rating':

                    if prompt_context == 'none':
                        system_prompt = "<s> [INST] <<SYS>>You will output a number.<</SYS>>"

                    elif prompt_context == 'medium':
                        
                        if injection == 'with':
                            system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Output an option from the prompt.<|eot_id|><|start_header_id|>user<|end_header_id|>"
                            
                        if injection == 'no':
                            system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Output an option from the prompt.<|eot_id|><|start_header_id|>user<|end_header_id|>"

                    elif prompt_context == 'high':
                        
                        if injection == 'with':
                            system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a cross-domain recommender system. Analyze user behavior in a source domain and transfer that knowledge to make a recommendation in a target domain. If both domains are available, prioritize the target domain but consider the source domain as well. Respond with one of the following: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item.<|eot_id|><|start_header_id|>user<|end_header_id|><"
                        
                        if injection == 'no':
                            system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a cross-domain recommender system. Analyze user behavior in a source domain and transfer that knowledge to make a recommendation in a target domain. If both domains are available, prioritize the target domain but consider the source domain as well. Respond with one of the following: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item.<|eot_id|><|start_header_id|>user<|end_header_id|>"

                elif rating_ranking == 'ranking':

                    if prompt_context == 'none':
                        system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You will output a list.<|eot_id|><|start_header_id|>user<|end_header_id|>"

                    elif prompt_context == 'medium':
                        system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You will output a ranked list, where items will be ranked from most likely to interact with to least likely to interact with.<|eot_id|><|start_header_id|>user<|end_header_id|>"

                    elif prompt_context == 'high':
                        system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. You will output a ranked list of candidate items. The first item in the list should be the item that the user will most likeley interact with, and the last item in the list should be the item the user will be least likely to interact with. You should follow the expected format in the prompt.<|eot_id|><|start_header_id|>user<|end_header_id|>"


                input_with_label = str(prompt) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                input_with_label_base = str(system_prompt) + str(prompt) + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            if rating_ranking == 'rating':

                input_ids = base_tokenizer(input_with_label_base, return_tensors="pt", truncation=True).input_ids.cuda()
                outputs = base_model.generate(input_ids=input_ids, max_new_tokens=1000)

                if model_name.lower() in ['7b', '13b']:
                    output = base_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_with_label_base)-2:]

                    base = output.lower()
                    base_preds.append(base)

                elif model_name.lower() in ['8b']:
                    output = base_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                    base = str(output).lower().split("assistant\n\n")[-1]
                    base = str(base).split("assistant")[-1]
                    base_preds.append(str(base))

            elif rating_ranking == 'ranking':
                input_ids = base_tokenizer(input_with_label_base, return_tensors="pt", truncation=True).input_ids.cuda()
                outputs = base_model.generate(input_ids=input_ids, max_new_tokens=4096)
                output = base_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_with_label_base):]

                base_preds.append(output)

        if rating_ranking == 'rating':

            base_mae = 0.0
            base_rmse = 0.0
            for x, y in zip(base_preds, ground_truths):

                candidate_item_rating = 0.0

                if 'highly likely' in x:
                    candidate_item_rating = 5.0

                elif 'very unlikely' in x:
                    candidate_item_rating = 0.5

                elif 'somewhat unlikely' in x:
                    candidate_item_rating = 2.5

                elif 'unlikely' in x:
                    candidate_item_rating = 1.5

                elif 'likely' in x:
                    candidate_item_rating = 4.5

                elif 'neutral' in x:
                    candidate_item_rating = 3.0

                base_mae += abs(candidate_item_rating - y)
                base_rmse += (candidate_item_rating - y) ** 2
            
            base_mae /= len(ground_truths)
            base_rmse /= len(ground_truths)

            if base_mae < best_mae:
                best_mae = base_mae
            if base_rmse < best_rmse:
                best_rmse = base_rmse

            if base_mae > worst_mae:
                worst_mae = base_mae
            if base_rmse > worst_rmse:
                worst_rmse = base_rmse

        if rating_ranking == 'ranking':


            total_mrr_5 = 0.0
            total_ndcg_5 = 0.0

            total_mrr_10 = 0.0
            total_ndcg_10 = 0.0

            for pred, truth in zip (base_preds, ground_truths):
                pred_string = pred.lower().strip()
                ground_truth = truth.lower().strip()

                predictions = [item.strip() for item in pred_string.split(",") if item.strip()]

                position = -1.0
                count = 1.0
                for pred in predictions:
                    if ground_truth in pred:
                        position = count
                        break
                    count += 1.0

                mrr_5 = 1 / position if (position > 0 and position <= 5) else 0
                mrr_10 = 1 / position if (position > 0 and position <= 10) else 0

                ndcg_5 = 0
                ndcg_10 = 0

                if position > 0 and position <= 5:
                    # Calculate DCG@5
                    dcg_at_5 = 1 / math.log2(position + 1)

                    # Calculate IDCG (ideal situation is having the relevant item at the top)
                    idcg = 1 / math.log2(2) 

                    # Calculate NDCG@5
                    ndcg_at_5 = dcg_at_5 / idcg
                    ndcg_5 = ndcg_at_5

                if position > 0 and position <= 10:
                    # Calculate DCG@10
                    dcg_at_10 = 1 / math.log2(position + 1)

                    # Calculate IDCG (ideal situation is having the relevant item at the top)
                    idcg = 1 / math.log2(2) 

                    # Calculate NDCG@10
                    ndcg_at_10 = dcg_at_10 / idcg
                    ndcg_10 = ndcg_at_10

                total_mrr_5 += mrr_5
                total_ndcg_5 += ndcg_5

                total_mrr_10 += mrr_10
                total_ndcg_10 += ndcg_10

            total_mrr_5 /= len(ground_truths)
            total_ndcg_5 /= len(ground_truths)

            total_mrr_10 /= len(ground_truths)
            total_ndcg_10 /= len(ground_truths)

            # Update best and worst values for MRR@5 and NDCG@5
            if total_mrr_5 > best_mrr_5:
                best_mrr_5 = total_mrr_5
            if total_ndcg_5 > best_ndcg_5:
                best_ndcg_5 = total_ndcg_5

            if total_mrr_5 < worst_mrr_5:
                worst_mrr_5 = total_mrr_5
            if total_ndcg_5 < worst_ndcg_5:
                worst_ndcg_5 = total_ndcg_5

            # Update best and worst values for MRR@10 and NDCG@10
            if total_mrr_10 > best_mrr_10:
                best_mrr_10 = total_mrr_10
            if total_ndcg_10 > best_ndcg_10:
                best_ndcg_10 = total_ndcg_10

            if total_mrr_10 < worst_mrr_10:
                worst_mrr_10 = total_mrr_10
            if total_ndcg_10 < worst_ndcg_10:
                worst_ndcg_10 = total_ndcg_10

    print("\n\n----- Final Statistics -----\n\n")

    if rating_ranking == "rating":

        print("Base Model *BEST* MAE:", best_mae)
        print("Base Model *BEST* RMSE:", sqrt(best_rmse))

        print()

        print("Base Model *WORST* MAE:", worst_mae)
        print("Base Model *WORST* RMSE:", sqrt(worst_rmse))

        print()

    elif rating_ranking == "ranking":

        print("Base Model *BEST* MRR@5: " + str(best_mrr_5))
        print("Base Model *BEST* NDCG@5: " + str(best_ndcg_5))

        print()

        print("Base Model *BEST* MRR@10: " + str(best_mrr_10))
        print("Base Model *BEST* NDCG@10: " + str(best_ndcg_10))

        print()

        print("Base Model *WORST* MRR@5: " + str(worst_mrr_5))
        print("Base Model *WORST* NDCG@5: " + str(worst_ndcg_5))

        print()

        print("Base Model *WORST* MRR@10: " + str(worst_mrr_10))
        print("Base Model *WORST* NDCG@10: " + str(worst_ndcg_10))










import os
import time
import argparse
from evaluate import evaluate_llm
from process_data import partition_data
from gpt_evaluate import gpt_evaluate_llm
from rating_prediction_prompts import rating_with_target_injection_train_dataset, rating_with_target_injection_validation_dataset, rating_with_target_injection_test_dataset, rating_no_target_injection_train_dataset, rating_no_target_injection_validation_dataset, rating_no_target_injection_test_dataset
from ranking_prediction_prompts import ranking_with_target_injection_train_dataset, ranking_with_target_injection_validation_dataset, ranking_with_target_injection_test_dataset, ranking_no_target_injection_train_dataset, ranking_no_target_injection_validation_dataset, ranking_no_target_injection_test_dataset

if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process data for a given dataset and provide results for LLM-CDR.")

    # List of arguments in this project
    parser.add_argument('--data_info', required=True, help="Specify which dataset the domains are from (e.g., 'amazon')")
    parser.add_argument('--source', required=True, help="Name of the dataset to process as the source domain (e.g., 'Movies_and_TV')")
    parser.add_argument('--target', required=True, help="Name of the dataset to process as the target domain (e.g., 'Books')")
    parser.add_argument('--neg_samples', required=True, help="How many negative items to sample for ranking metrics (e.g., '10' or '20')")
    parser.add_argument('--k_shot', required=True, help="Percent of K-Shot training examples (e.g., '25' or '50' or '75'')")
    parser.add_argument('--model_name', required=True, help="Specify the name of the model that you want to fine-tune (e.g. 7B or 13B or 70B)")
    parser.add_argument('--task', required=True, help="Specify the task that you want to fine-tune for (e.g. rating or ranking)")
    parser.add_argument('--injection', required=True, help="Specify if you want to finetune with target injection (e.g. with or no)")
    parser.add_argument('--prompt_context', required=True, help="Specify how much context you want in your prompt (e.g. none, medium, high)")

    # Parse the arguments
    args = parser.parse_args()

    start_time = time.time()

    # # Call the main function to create dataset for the project
    k_sampled_users = partition_data(args.source, args.target, args.k_shot, args.data_info)

    if args.task == "rating":

        # Generate target injection rating prediction prompts
        if args.injection == "with":

            # Un-comment the other two if you want to generate training and validation datasets for your task

            # rating_with_target_injection_train_dataset(args.k_shot, args.source, args.target, args.data_info, k_sampled_users)
            # rating_with_target_injection_validation_dataset(args.k_shot, args.source, args.target, args.data_info)
            rating_with_target_injection_test_dataset(args.k_shot, args.source, args.target, args.data_info)

        # Generate no target injection rating prediction prompts
        if args.injection == "no":

            # Un-comment the other two if you want to generate training and validation datasets for your task
            
            # rating_no_target_injection_train_dataset(args.k_shot, args.source, args.target, args.data_info, k_sampled_users)
            # rating_no_target_injection_validation_dataset(args.k_shot, args.source, args.target, args.data_info)
            rating_no_target_injection_test_dataset(args.k_shot, args.source, args.target, args.data_info)

    if args.task == "ranking":

        # Generate target injection ranking task prompts
        if args.injection == "with":

            # Un-comment the other two if you want to generate training and validation datasets for your task

            # ranking_with_target_injection_train_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples), k_sampled_users)
            # ranking_with_target_injection_validation_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples))
            ranking_with_target_injection_test_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples))

        # Generate no target injection ranking task prompts for train and test sets
        if args.injection == "no":

            # Un-comment the other two if you want to generate training and validation datasets for your task

            # ranking_no_target_injection_train_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples), k_sampled_users)
            # ranking_no_target_injection_validation_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples))
            ranking_no_target_injection_test_dataset(args.k_shot, args.source, args.target, args.data_info, int(args.neg_samples))

    # Evaluation GPT model on the test dataset
    if "GPT" in args.model_name:

        gpt_evaluate_llm(args.k_shot, args.source, args.target, args.data_info, args.model_name, args.task, args.injection, args.prompt_context)

    # Evaluation LLama model on the test dataset
    else:
        
        evaluate_llm(args.k_shot, args.source, args.target, args.data_info, args.model_name, args.task, args.injection, args.prompt_context)

    end_time = time.time()

    elapsed_time_seconds = end_time - start_time
    minutes = int(elapsed_time_seconds // 60)
    seconds = int(elapsed_time_seconds % 60)

    print(f"\nTotal execution time: {minutes} minutes and {seconds} seconds")



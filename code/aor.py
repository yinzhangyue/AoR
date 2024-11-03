# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/3/10
import jsonlines
from collections import Counter
from prompt import general_generation_prompt, local_scoring_prompt, global_evaluation_prompt, math_generation_prompt, math_local_scoring_prompt, math_global_evaluation_prompt
from inference import Inference_Model
from metric import GSM8K_Metric, MultiArith_Metric, SingleEq_Metric, AddSub_Metric, AQuA_Metric, SVAMP_Metric, CSQA_Metric, StrategyQA_Metric
import argparse
import os
import re
import json
from ipdb import set_trace

parser = argparse.ArgumentParser()

# Basic Setting
parser.add_argument("--task", type=str, required=True, help="Reasoning Task")
parser.add_argument("--data-path", type=str, required=True, help="path to the data file")
parser.add_argument("--record-path", type=str, required=True, help="path to save the record file")
parser.add_argument("--inference-model", default="gpt-35-turbo-0301", choices=["gpt-35-turbo-0301", "gpt-4"], type=str, help="inference model for AoR")
# Hyperparameters
parser.add_argument("--initial-sample-size", type=int, default=20, help="Initial sample size for reasoning chains")
parser.add_argument("--max-sample-size", type=int, default=40, help="Upper limit for dynamic sampling of reasoning chains")
parser.add_argument("--batch-size", type=int, default=5, help="Batch size for sampling reasoning chains")
parser.add_argument("--representative-count", type=int, default=3, help="Representative count for local scoring phase")
parser.add_argument("--scoring-threshold", type=int, default=6, help="Threshold for scoring during local scoring phase")
parser.add_argument("--termination-threshold", type=int, default=2, help="Threshold for dynamic sampling termination")
parser.add_argument("--additional-sample-size", type=int, default=5, help="Number of additional reasoning chains sampled in each iteration")

args = parser.parse_args()


def read_jsonl_file(file_path: str):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


class AoR:
    def __init__(self, args, metric, inference_model, generation_prompt="", local_scoring_prompt="", global_evaluation_prompt=""):
        # Basic Parameter
        self.args = args
        self.metric = metric
        self.inference_model = inference_model
        self.data_path = args.data_path
        self.record_path = args.record_path
        # Hyperparameters
        self.initial_sample_size = args.initial_sample_size
        self.max_sample_size = args.max_sample_size
        self.batch_size = args.batch_size
        self.representative_count = args.representative_count
        self.scoring_threshold = args.scoring_threshold
        self.termination_threshold = args.termination_threshold
        self.additional_sample_size = args.additional_sample_size
        # Prompt
        self.generation_prompt = generation_prompt
        self.local_scoring_prompt = local_scoring_prompt
        self.global_evaluation_prompt = global_evaluation_prompt

    def sample_reasoning_chains(self, question: str, n: int):
        prompt = f"Question: {question}\nAnswer the question step by step.\n"
        reasoning_chains = self.inference_model.get_info(query=prompt, System_Prompt=self.generation_prompt, n=n)
        return reasoning_chains

    def group_chains_by_answer(self, reasoning_chains: list):
        buckets = {}
        for chain in reasoning_chains:
            answer = self.metric.process_pred(chain)
            if answer not in buckets:
                buckets[answer] = []
            buckets[answer].append(chain)
        return buckets

    def parse_local_scoring_response(self, score_response: str, batch_size: int):
        # Use regex to find all occurrences of the pattern "Chain N Total Score: X/10"
        scores = re.findall(r"Chain (\d+) Total Score: (\d+)/10", score_response)

        # Convert the matches into a dictionary with chain index as key and score as value
        scores_dict = {f"Chain {chain}": int(score) for chain, score in scores}

        # Check if the length of the scores matches the batch size
        if len(scores_dict) != batch_size:
            raise ValueError(f"Expected {batch_size} scores, but got {len(scores_dict)}.")

        return scores_dict

    def automated_parse_local_scoring_response(self, score_response: str, batch_size: int):
        # Prepare the prompt
        prompt = f"""
    You are an assistant that extracts data from text.

    Given the following text:

    {score_response}

    Please extract the total scores for each chain. For each chain, identify the chain index (e.g., "Chain 1") and its total score (an integer from 0 to 10).

    Provide the output as a JSON object representing a dictionary where the keys are the chain indices (e.g., "Chain 1") and the values are the scores.

    For example:

    {{"Chain 1": 8, "Chain 2": 7, "Chain 3": 9}}

    If any chain's score is missing or the required information is not present for all chains, output `null`.

    Output only the JSON object and no additional text.
    """

        # Query GPT
        response = self.inference_model.get_info(query=prompt)
        response = response[0]
        # set_trace()
        # Extract JSON content from the response
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # Try without 'json' label
            match = re.search(r"```.*?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                json_text = match.group(1)
            else:
                # Fallback to attempt parsing the entire response
                json_text = response.strip()

        # Process the response
        try:
            if json_text.lower() == "null":
                return None

            # Parse the JSON output
            scores_dict = json.loads(json_text)

            # Check if the length of the scores matches the batch size
            if len(scores_dict) != batch_size:
                print(f"Expected {batch_size} scores, but got {len(scores_dict)}.")
                return None

            # Ensure that the keys are in the correct format
            for key in scores_dict.keys():
                if not key.startswith("Chain "):
                    print(f"Invalid chain index format: {key}")
                    return None

            # Ensure that all scores are integers between 0 and 10
            for score in scores_dict.values():
                if not isinstance(score, int) or not (0 <= score <= 10):
                    print(f"Invalid score value: {score}")
                    return None
            return scores_dict

        except Exception as e:
            print(f"Failed to parse scores: {e}")
            print(f"JSON text to parse:\n{json_text}")
            return None

    def local_scoring(self, buckets):
        topk_buckets = {}
        for answer, chains in buckets.items():
            scores = []
            batch_size = self.batch_size  # Define an appropriate batch size
            total_chains = len(chains)

            # Process chains in batches
            for i in range(0, total_chains, batch_size):
                batch_chains = chains[i : i + batch_size]
                # Prepare prompt for batch
                System_Prompt = self.local_scoring_prompt
                prompt = ""
                for idx, chain in enumerate(batch_chains):
                    prompt += f"Chain {idx + 1}:\n{chain}\n\n"
                prompt += "For each reasoning chain, analyze it in detail.\n" "At the end of each chain, calculate the total score and provide the final output in the format: " "'Chain N Total Score: X/10', where N is the index of the chain and X is the total score.\n"

                # Get response from LLM and attempt to parse scores with up to 5 retries
                attempts = 0
                batch_scores = None
                while attempts < 5:
                    try:
                        score_response = self.inference_model.get_info(query=prompt, System_Prompt=System_Prompt)
                        batch_scores = self.automated_parse_local_scoring_response(score_response[0], len(batch_chains))
                        if batch_scores is not None:
                            break  # Exit loop if parsing is successful
                        else:
                            attempts += 1
                            print(f"Parsing attempt {attempts} failed. Retrying...")
                            # print(f"Score response: {score_response[0]}")
                    except Exception as e:
                        attempts += 1
                        print(f"Exception during local scoring parsing: {e}")
                        print(f"Score response: {score_response[0]}")

                # If parsing fails after 5 attempts, log an error and skip to next batch
                if batch_scores is None:
                    print("Local scoring parsing failed after 5 attempts.")
                    print(f"Failed score response:\n{score_response[0]}")
                    return None
                # Add the parsed scores to the scores list, ensuring each chain matches its score
                scores.extend(zip(batch_chains, batch_scores.values()))
            # Filter chains based on the scoring threshold
            filtered_chains = [item for item in scores if item[1] >= self.scoring_threshold]

            # Sort and select the top k chains
            topk_chains = sorted(filtered_chains, key=lambda x: x[1], reverse=True)[: self.representative_count]
            if topk_chains:
                topk_buckets[answer] = topk_chains

        return topk_buckets

    def local_scoring_new_chains(self, new_buckets, buckets, topk_buckets):
        new_topk_buckets = {}
        for answer, chains in new_buckets.items():
            existing_chains = buckets[answer]  # All chains for this answer
            if answer in topk_buckets:
                # Get highest and lowest scoring chains from topk_buckets
                existing_scores = topk_buckets[answer]
                highest_chain, highest_score = max(existing_scores, key=lambda x: x[1])
                lowest_chain, lowest_score = min(existing_scores, key=lambda x: x[1])
            else:
                # No existing scores, cannot get highest and lowest
                highest_chain = None
                lowest_chain = None
                highest_score = None
                lowest_score = None
            # Prepare prompt for local scoring
            batch_size = self.batch_size
            total_chains = len(chains)
            scores = []
            for i in range(0, total_chains, batch_size):
                batch_chains = chains[i : i + batch_size]
                System_Prompt = self.local_scoring_prompt
                prompt = ""

                # Include highest and lowest chain for demonstration if available
                if highest_chain:
                    prompt += f"Below are examples of high-scoring and low-scoring reasoning chains for your reference:\n\n"
                    prompt += f"High-Scoring Chain (Total Score: {highest_score}/10):\n{highest_chain}\n\n"
                    if lowest_chain and lowest_chain != highest_chain:
                        prompt += f"Low-Scoring Chain (Total Score: {lowest_score}/10):\n{lowest_chain}\n\n"
                    prompt += "Please use these examples as references when evaluating the following reasoning chains.\n\n"
                else:
                    # If no references are available
                    prompt += "Please evaluate the following reasoning chains.\n\n"

                idx = 1
                # Add batch chains to the prompt
                for j, chain in enumerate(batch_chains, start=idx):
                    prompt += f"Chain {j}:\n{chain}\n\n"
                prompt += "For each reasoning chain, analyze it in detail.\n" "At the end of each analysis, calculate the total score and provide the final output in the format: " "'Chain N Total Score: X/10', where N is the index of the chain and X is the total score.\n"
                # Get response from LLM and parse scores
                attempts = 0
                scores_dict = None
                while attempts < 5:
                    try:
                        score_response = self.inference_model.get_info(query=prompt, System_Prompt=System_Prompt)
                        # Number of chains is len(batch_chains)
                        scores_dict = self.automated_parse_local_scoring_response(score_response[0], len(batch_chains))
                        if scores_dict is not None:
                            # Extract scores for the batch_chains
                            for j, chain in enumerate(batch_chains, start=1):
                                score = scores_dict.get(f"Chain {j}", None)
                                if score is not None:
                                    scores.append((chain, score))
                            break
                        else:
                            attempts += 1
                            print(f"Parsing attempt {attempts} failed. Retrying...")
                    except Exception as e:
                        attempts += 1
                        print(f"Exception during local scoring parsing: {e}")
                        print(f"Score response: {score_response[0]}")
                else:
                    print("Local scoring parsing failed for new chains after 5 attempts.")
                    print(f"Failed score response:\n{score_response[0]}")
                    continue  # Skip this batch
            # Filter chains based on scoring threshold
            filtered_chains = [item for item in scores if item[1] >= self.scoring_threshold]
            # Sort and select top k chains
            topk_chains = sorted(filtered_chains, key=lambda x: x[1], reverse=True)[: self.representative_count]
            if topk_chains:
                new_topk_buckets[answer] = topk_chains
        return new_topk_buckets

    def parse_global_evaluation_response(self, score_response, answer_options):
        # Use regex to extract the total score for each answer option
        scores = re.findall(r"Answer Option: ([^\s]+)\s+Total Score: (\d+)/10", score_response)

        # Convert the extracted scores into a dictionary
        scores_dict = {answer: int(score) for answer, score in scores}

        # Check if all answer_options are in the parsed results
        missing_options = [option for option in answer_options if option not in scores_dict]
        if missing_options:
            print(f"Missing scores for the following answer options: {', '.join(missing_options)}")
            return None  # Indicate parsing failure

        return scores_dict

    def automated_parse_global_evaluation_response(self, score_response: str, answer_options):
        # Prepare the prompt
        prompt = f"""
        You are an assistant that extracts data from text.

        Given the following text:

        {score_response}

        Please extract the total scores for each answer option. For each answer option, identify the **answer** (e.g., "4", "45") and its total score (an integer from 0 to 10).

        **Important:** Do not include any prefixes like "Answer Option: " or "Option: " in the keys of the JSON object.

        Provide the output as a JSON object representing a dictionary where the keys are the answer options **without any prefixes** and the values are the scores.

        For example:

        {{"4": 8, "45": 7, "90": 9}}

        If any answer option's score is missing or the required information is not present for all answer options, output `null`.

        Output only the JSON object and no additional text.
        """

        # Query GPT
        response = self.inference_model.get_info(query=prompt)
        response = response[0]
        # set_trace()

        # Extract JSON content from the response
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # Try without 'json' label
            match = re.search(r"```.*?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                json_text = match.group(1)
            else:
                # Fallback to attempt parsing the entire response
                json_text = response.strip()

        # Process the response
        try:
            if json_text.lower() == "null":
                return None

            # Parse the JSON output
            scores_dict = json.loads(json_text)

            # Check if all answer_options are in the parsed results
            missing_options = [option for option in answer_options if option.strip() not in scores_dict]
            if missing_options:
                print(f"Missing scores for the following answer options: {', '.join(missing_options)}")
                return None

            # Ensure that all scores are integers between 0 and 10
            for score in scores_dict.values():
                if not isinstance(score, int) or not (0 <= score <= 10):
                    print(f"Invalid score value: {score}")
                    return None

            return scores_dict

        except Exception as e:
            print(f"Failed to parse scores: {e}")
            print(f"JSON text to parse:\n{json_text}")
            return None

    def global_evaluation(self, topk_buckets):
        total_scores = {answer: 0 for answer in topk_buckets}
        k = self.representative_count  # Number of representatives to evaluate
        for t in range(k):
            # Select one representative per bucket for this round
            selected_chains = {}
            for answer, chains in topk_buckets.items():
                chain = chains[t % len(chains)][0]  # Select chain, wrap around if fewer than k
                selected_chains[answer] = chain

            # Prepare the prompt including all selected chains
            System_prompt = self.global_evaluation_prompt
            prompt = ""
            for answer, chain in selected_chains.items():
                prompt += f"Answer Option: {answer}\nReasoning Chain:\n{chain}\n\n"
            prompt += "For each answer option, analyze it in detail and provide a final score. " "At the end of each evaluation, output the result in the format: 'Answer Option: Z Total Score: S/10', " "where Z is the answer option, and S is the total score.\n"

            # Get response from the LLM and attempt to parse scores with up to 5 retries
            attempts = 0
            scores_dict = None
            while attempts < 5:
                try:
                    score_response = self.inference_model.get_info(query=prompt, System_Prompt=System_prompt)
                    # Use the automated parsing function
                    scores_dict = self.automated_parse_global_evaluation_response(score_response[0], selected_chains.keys())
                    if scores_dict is not None:
                        break  # Exit loop if parsing is successful
                    else:
                        attempts += 1
                        print(f"Parsing attempt {attempts} failed. Retrying...")
                except Exception as e:
                    attempts += 1
                    print(f"Exception during global evaluation parsing: {e}")
                    print(f"Score response: {score_response[0]}")
            else:
                print("Global scoring parsing failed after 5 attempts.")
                print(f"Failed score response:\n{score_response[0]}")
                return None, {}

            # Accumulate the scores for each answer
            for answer, score in scores_dict.items():
                total_scores[answer] += score

        # Calculate average scores
        average_scores = {answer: total_scores[answer] / k for answer in total_scores}

        # Determine the final answer based on the highest average score
        if average_scores:
            final_answer = max(average_scores, key=average_scores.get)
            return final_answer, average_scores
        else:
            return None, average_scores

    def dynamic_sampling(self, current_size):
        return min(self.additional_sample_size, self.max_sample_size - current_size)

    def run(self):
        data_list = read_jsonl_file(self.data_path)
        total_acc = 0
        total_num = len(data_list)
        for data in data_list:
            question = data["question"]
            answer = data["answer"]
            reasoning_chains = []

            if "response" in data.keys():
                if len(data["response"]) >= self.initial_sample_size:
                    reasoning_chains = data["response"][: self.initial_sample_size]
                else:
                    reasoning_chains = data["response"]
                    additional_n = self.initial_sample_size - len(reasoning_chains)
                    if additional_n > 0:
                        new_chains = self.sample_reasoning_chains(question, additional_n)
                        reasoning_chains.extend(new_chains)
            else:
                reasoning_chains = self.sample_reasoning_chains(question, self.initial_sample_size)
            current_sample_size = len(reasoning_chains)
            margin = 0
            final_answer = None
            # Process initial reasoning chains
            buckets = self.group_chains_by_answer(reasoning_chains)
            if not buckets:
                continue  # No valid answers
            if len(buckets) == 1:
                final_answer = list(buckets.keys())[0]
                margin = float("inf")  # Since only one answer
            else:
                topk_buckets = self.local_scoring(buckets)
                if not topk_buckets:
                    continue  # No chains passed local scoring
                final_answer, average_scores = self.global_evaluation(topk_buckets)
                if not final_answer:
                    continue  # No valid final answer
                sorted_scores = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_scores) > 1:
                    margin = sorted_scores[0][1] - sorted_scores[1][1]
                else:
                    margin = sorted_scores[0][1]
            # Now enter the loop for dynamic sampling
            while margin < self.termination_threshold and current_sample_size < self.max_sample_size:
                additional_n = self.dynamic_sampling(current_sample_size)
                if additional_n <= 0:
                    break
                additional_chains = self.sample_reasoning_chains(question, additional_n)
                current_sample_size += additional_n
                reasoning_chains.extend(additional_chains)
                # Process new chains
                new_buckets = self.group_chains_by_answer(additional_chains)
                if not new_buckets:
                    continue  # No valid answers in new chains
                # Merge new buckets with existing buckets
                for ans, chains in new_buckets.items():
                    if ans in buckets:
                        buckets[ans].extend(chains)
                    else:
                        buckets[ans] = chains
                # Perform local scoring on new chains per answer group
                updated = False
                new_topk_buckets = self.local_scoring_new_chains(new_buckets, buckets, topk_buckets)
                if new_topk_buckets:
                    # Update topk_buckets with new_topk_buckets
                    for ans, chains_scores in new_topk_buckets.items():
                        if ans in topk_buckets:
                            # Combine and keep top-k
                            combined_chains = topk_buckets[ans] + chains_scores
                            # Sort and select top k
                            top_chains = sorted(combined_chains, key=lambda x: x[1], reverse=True)[: self.representative_count]
                            # Check if top_chains has changed
                            if top_chains != topk_buckets[ans]:
                                topk_buckets[ans] = top_chains
                                updated = True
                        else:
                            # New answer, add to topk_buckets
                            topk_buckets[ans] = chains_scores
                            updated = True  # Since we have a new answer, we need to update
                if updated:
                    # Re-run global evaluation
                    final_answer, average_scores = self.global_evaluation(topk_buckets)
                    if not final_answer:
                        break  # No valid final answer
                    sorted_scores = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_scores) > 1:
                        margin = sorted_scores[0][1] - sorted_scores[1][1]
                    else:
                        margin = sorted_scores[0][1]
                else:
                    # No change in topk_buckets, margin remains the same
                    pass
            # After loop, check accuracy
            if final_answer:
                is_correct = self.metric.get_acc([final_answer], answer)
                total_acc += is_correct
                tag = "Executed Successfully"
            else:
                is_correct = 0
                tag = "Executed Failed"
            # Save record
            data["final_answer"] = final_answer
            data["is_correct"] = is_correct
            data["reasoning_chains"] = reasoning_chains
            data["tag"] = tag
            self.save_record(data)
        print("Total Accuracy: ", total_acc / total_num)

    def save_record(self, data):
        with jsonlines.open(self.record_path, "a") as writer:
            writer.write(data)


if __name__ == "__main__":
    format_hint_dict = {"GSM8K": "", "MultiArith": "", "AddSub": "", "SingleEq": "", "AQuA": " (A/B/C/D/E)", "SVAMP": "", "CSQA": " (A/B/C/D/E)", "StrategyQA": " (yes/no)"}
    metric_dict = {"GSM8K": GSM8K_Metric, "MultiArith": MultiArith_Metric, "AddSub": AddSub_Metric, "SingleEq": SingleEq_Metric, "AQuA": AQuA_Metric, "SVAMP": SVAMP_Metric, "CSQA": CSQA_Metric, "StrategyQA": StrategyQA_Metric}
    # Set the most appropriate evaluation prompt based on the task type.
    general_generation_prompt_dict = {"GSM8K": math_generation_prompt, "MultiArith": math_generation_prompt, "AddSub": math_generation_prompt, "SingleEq": math_generation_prompt, "AQuA": math_generation_prompt, "SVAMP": math_generation_prompt, "CSQA": general_generation_prompt, "StrategyQA": general_generation_prompt}
    local_scoring_prompt_dict = {"GSM8K": math_local_scoring_prompt, "MultiArith": math_local_scoring_prompt, "AddSub": math_local_scoring_prompt, "SingleEq": math_local_scoring_prompt, "AQuA": math_local_scoring_prompt, "SVAMP": math_local_scoring_prompt, "CSQA": local_scoring_prompt, "StrategyQA": local_scoring_prompt}
    global_evaluation_prompt_dict = {"GSM8K": math_global_evaluation_prompt, "MultiArith": math_global_evaluation_prompt, "AddSub": math_global_evaluation_prompt, "SingleEq": math_global_evaluation_prompt, "AQuA": math_global_evaluation_prompt, "SVAMP": math_global_evaluation_prompt, "CSQA": global_evaluation_prompt, "StrategyQA": global_evaluation_prompt}

    task = args.task
    data_path = args.data_path
    record_path = args.record_path
    model = args.inference_model
    inference_model = Inference_Model(model)

    format_hint = format_hint_dict[task]
    metric = metric_dict[task]()
    generation_prompt = general_generation_prompt_dict[task]
    local_scoring_prompt = local_scoring_prompt_dict[task]
    global_evaluation_prompt = global_evaluation_prompt_dict[task]

    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    print("Task:{} Model:{}".format(task, model))
    aor = AoR(args, metric, inference_model, generation_prompt, local_scoring_prompt, global_evaluation_prompt)
    aor.run()

import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
import os
import torch
import json
import codecs
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Optional, Union, Dict, Tuple
import math
from tqdm import tqdm
import numpy as np
import itertools
import argparse
import gc

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--GPU', type=int, default=0, help='The device ID of GPU')
parser.add_argument('--begin', type=int, default=0, help='In this experiment, which option order to start from')
parser.add_argument('--num', type=int, default=1, help='In this experiment, how many option orders are involved')
parser.add_argument('--dataset', type=str, default='SST-2', help='In this experiment, which dataset is considered')
parser.add_argument('--subset', type=str, default='train', help='train/test, i.e. the held-out set or validation set')
parser.add_argument('--restart', type=str, default='n', help='If the program is interrupted, continue to run based on the saved results (y) or not (n)')
# 解析参数，即从命令行获取这些待定参数
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9'
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
device_ids = list(range(args.GPU,10))
torch.cuda.set_device(9)


tokenizer = AutoTokenizer.from_pretrained("/Data/transformers/chatglm4", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "/Data/transformers/chatglm4",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


torch.set_grad_enabled(False)

raw_file_formats = {'SST-2': 'jsonl', 'IMDB': 'parquet', 'Subj': 'jsonl', 'RTE': 'jsonl',
                   'MNLI': 'parquet', 'AG News': 'parquet', 'BBC News': 'jsonl', 'Emergency_TL': 'jsonl', 
                    'MedNLI': 'parquet'}
file_format = raw_file_formats[args.dataset]
raw_file_path = f'/Data/datasets/Option order results/{args.dataset}/raw data/{args.subset}.' + file_format
labels_dict = {'SST-2': ['negative', 'positive'],
          'IMDB': ['negative', 'positive'],
          'RTE': ['entailment', 'not entailment'],
          'Subj': ['objective', 'subjective'],
          'MNLI': ['entailment', 'neutral', 'contradiction'],
          'AG News': ["Wordl news", "Sports news", "Business news", "Science/Technology news"],
          'BBC News': ['Science/Technology news', 'Business news', 'Sport news', 'Entertainment news', 'Politics news'],
          'Emergency_TL': ["这名患者的分诊级别为1级，属于急危患者", "这名患者的分诊级别低于1级，不属于急危患者"],
          'MedNLI': ['entailment', 'neutral', 'contradiction']}

prompt_templates = {'SST-2': 'The following is a movie review:\n%s\nDetermine the sentiment of this movie review:%s\nThe correct option is ',
                    'IMDB': 'Review: %s\nDetermine the sentiment of the review:%s\nThe correct option is ',
                    'RTE': 'Sentence 1: %s\nSentence 2: %s\nDetermine the existence of entailment relationship between Sentence 1 and 2:%s\nThe correct option is ',
                    'Subj': 'Sentence: %s\nDetermine the narrative perspective of the sentence:%s\nThe correct option is ',
                    'MNLI': 'Premise: %s\nHypothesis: %s\nDetermine the relationship between the premise and hypothesis:%s\nThe correct option is ',
                    'AG News': 'Determine the category of the following news:\nNews: %s%s\nThe correct option is ',
                    'BBC News': 'News: %s\nClassify the above news into one of the following categories:%s\nThe correct option is ',
                    'Emergency_TL': '以下描述涉及急诊患者的症状及病史：\n%s\n根据以上信息, 判断这名患者的分诊级别是否为1级。其中1级是指患者正在或即将面临生命威胁或病情恶化。%s\n正确选项是：',
                    'MedNLI': 'Premise: %s\nHypothesis: %s\nDetermine the relationship between the premise and hypothesis:%s\nThe correct option is '}

fixed_indices_dict = {'SST-2': [1, 0],
                      'IMDB': [0, 12500],
                      'RTE': [1, 0],
                      'Subj': [0, 3],
                      'MNLI': [2, 0, 1],
                      'AG News': [492, 448, 0, 78],
                      'BBC News': [4, 1, 0, 2, 5],
                      'Emergency_TL': [0, 11],
                      'MedNLI': [0, 2, 1]}


labels = labels_dict[args.dataset]
letters = [chr(i) for i in range(65,70)][:len(labels)]

knowledge_base_file = f'/Data/datasets/Option order results/{args.dataset}/raw data/train.' + file_format
if file_format == 'jsonl':
    knowledge_base = pd.read_json(knowledge_base_file, orient='records', lines=True)
else:
    knowledge_base = pd.read_parquet(knowledge_base_file)

knowledge_base = knowledge_base.to_dict(orient='records')

shot_indices = fixed_indices_dict[args.dataset]

# The text in BBC News is too long, cut the text in context example
if args.dataset == 'BBC News':
    for i in shot_indices:
        knowledge_base[i]['text'] = knowledge_base[i]['text'][:knowledge_base[i]['text'].find('.')]


# Implement in-context learning according to build the dialogue history
def build_history(knowledge_base:list,indices, datasets, letters, labels, prompt_templates):
    history: List[Tuple[str,str]] = []
    for indice in indices:
        query = construct_input_query(datasets, letters, labels, prompt_templates, knowledge_base[indice])
        answer = f"{letters[exchange[knowledge_base[indice]['label']]]}. {labels[exchange[knowledge_base[indice]['label']]]}"
        history.append({'role': 'user', 'content': query})
        history.append({'role': 'assistant', 'content': answer})
    return history

def format_choice(letters, labels):
    results = ""
    for i in range(len(letters)):
        results += f"\n{letters[i]}. {labels[i]}"
    return results

# Construct input based on prompt templates
def construct_input_query(datasets, letters, labels, prompt_templates, item):
    options_text = format_choice(letters, labels)
    prompt = prompt_templates[datasets]
    if datasets == 'SST-2':
        input_query = prompt % (item['text'], options_text)
    elif datasets == 'IMDB':
        input_query = prompt % (item['text'], options_text)
    elif datasets == 'RTE':
        input_query = prompt % (item['text1'], item['text2'], options_text)
    elif datasets == 'Subj':
        input_query = prompt % (item['text'], options_text)
    elif datasets == 'MNLI':
        input_query = prompt % (item['premise'], item['hypothesis'], options_text)
    elif datasets == 'AG News':
        input_query = prompt % (item['text'], options_text)
    elif datasets == 'BBC News':
        input_query = prompt % (item['text'], options_text)
    elif datasets == 'Emergency_TL':
        input_query = prompt % (item['护理记录'], options_text)
    elif datasets == 'MedNLI':
        input_query = prompt % (item['sentence1'], item['sentence2'], options_text)

    return input_query

# Using tokenizer to construct the actual input
def build_inputs(model, tokenizer, history: List[Dict[str, str]] = None, add_generation_prompt=True, tokenize=True,
                 return_tensors="pt", return_dict=True):
    inputs = tokenizer.apply_chat_template(history, add_generation_prompt=add_generation_prompt, tokenize=tokenize,
                                           return_tensors=return_tensors, return_dict=return_dict)
    inputs = inputs.to(model.device)
    return inputs

# When LLM generated unstructured response, find the position of the option ID in the unstructured output
def find_unstructured_answer(output,aim_token_ids_1,aim_token_ids_2):
    candidate_position_1 = torch.where(torch.isin(torch.tensor(output).to(device), aim_token_ids_1))[0]
    candidate_position_2 = torch.where(torch.isin(torch.tensor(output).to(device), aim_token_ids_2))[0]
    if len(candidate_position_1) >= 1 and len(candidate_position_2) >= 1:
        answer_position = torch.max(candidate_position_1[-1], candidate_position_2[-1]).item()
    elif len(candidate_position_1) >= 1:
        answer_position = candidate_position_1[-1]
    elif len(candidate_position_2) >= 1:
        answer_position = candidate_position_2[-1]
    else:
        answer_position = 0
    return answer_position

# Given the unstructured output, concatenate the input and output, find the indice/position of the aim logit vector (corresponding to the option ID)
def find_position(input_ids,answer, aim_token_ids_1, aim_token_ids_2, letters):
    target_id_1 = aim_token_ids_1[letters.index(answer)]
    target_id_2 = aim_token_ids_2[letters.index(answer)]
    candidate_position_1 = torch.where(torch.isin(input_ids, target_id_1))[0]
    candidate_position_2 = torch.where(torch.isin(input_ids, target_id_2))[0]
    if len(candidate_position_1) >= 1 and len(candidate_position_2) >= 1:
        answer_position = torch.max(candidate_position_1[-1], candidate_position_2[-1]).item()
    elif len(candidate_position_1) >= 1:
        answer_position = candidate_position_1[-1]
    elif len(candidate_position_2) >= 1:
        answer_position = candidate_position_2[-1]
    else:
        answer_position = 0
    return answer_position-1

# If the number of options is less than 5, we consider all possible option orders; otherwise only cyclic permutations are considered
if len(letters) != 5:
    orders_list = list(itertools.permutations(list(range(len(letters)))))[args.begin:args.begin+args.num]
else:
    orders_list = [[0, 1, 2, 3, 4],
                   [4, 0, 1, 2, 3],
                   [3, 4, 0, 1, 2],
                   [2, 3, 4, 0, 1],
                   [1, 2, 3, 4, 0]]
    orders_list = orders_list[args.begin:args.begin+args.num]
print(f'此次测试的选项顺序为：{orders_list}')


#循环过程中待统计的量
for exchange in orders_list:

    if file_format == 'jsonl':
        data = pd.read_json(raw_file_path, orient='records', lines=True)
    else:
        data = pd.read_parquet(raw_file_path)

    if args.subset == 'train':
        # Due to the large sample size, for the held-out set in AG News, only 5% samples (randomly selected) are considered
        if args.dataset == 'AG News':
            data = data.sample(frac=0.05, random_state=4396)
        # Those context examples are not included
        else:
            data = data.drop(index = shot_indices)
            data = data.reset_index(drop=True)

    data = data.to_dict(orient='records')
    
    labels = labels_dict[args.dataset]
    labels = [labels[exchange.index(i)] for i in range(len(labels))]

    aim_token_ids_1 = torch.tensor([32, 33, 34, 35, 36]).to(device)
    aim_token_ids_1 = aim_token_ids_1[:len(labels)]
    aim_token_ids_2 = torch.tensor([362, 425, 356, 422, 468]).to(device)
    aim_token_ids_2 = aim_token_ids_2[:len(labels)]

    if args.restart == 'n':
        finished_num = 0
    else:
        assert args.restart == 'y'
        result_file_name = ""
        for i in range(len(exchange)):
            result_file_name += str(exchange[i])

        finished_num = 0
        save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/GLM4/{args.subset}_results_{result_file_name}.json'
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data = data['data']

            for item in data:
                if 'scores' in item:
                    finished_num += 1
                else:
                    break

        print(f'已完成{finished_num}个样本')

    for item in tqdm(data[finished_num:]):
        item['label'] = exchange[item['label']]
        finished_num+=1
        input_text = construct_input_query(args.dataset, letters, labels, prompt_templates, item)
        
        history = build_history(knowledge_base, shot_indices, args.dataset, letters, labels, prompt_templates)
        history.append({'role': 'user', 'content': input_text})

        inputs = build_inputs(model, tokenizer, history)
        with torch.no_grad():
            outputs_set = model.generate(**inputs, max_new_tokens=1, do_sample=False, num_beams=1, temperature=None,
                                         top_p=None, return_dict_in_generate=True, output_scores=True)
        outputs = outputs_set.sequences.tolist()[0][len(inputs["input_ids"][0]):]
        logits = outputs_set.scores
        response = tokenizer.decode(outputs, skip_special_tokens=True)

        if finished_num == 1:  # 打印第一道题的输入/输出示例
            print("\nExample: ")
            print("\nThe input pass to the model is:", tokenizer.decode(inputs["input_ids"][0]))
            print("\nThe output by chat is:", response)

        # Suppose that the first output token must be option ID
        logits = logits[0]
        # Otherwise, th following code is alternative

        # if outputs[0] in aim_token_ids_1 or outputs[0] in aim_token_ids_2:
        #     logits = logits[0]
        # else:
        #     print("The unformatted response: ", response)
        #     target_indice = find_unstructured_answer(outputs, aim_token_ids_1, aim_token_ids_2)
        #     # print("The extract letter is: ", tokenizer.decode(outputs[target_indice]))
        #     logits = logits[target_indice]

        torch.cuda.empty_cache()

        aim_token_ids_1 = aim_token_ids_1[:len(letters)]
        logits_trans_1 = logits[0, aim_token_ids_1]
        item['scores'] = logits_trans_1.tolist()

        if finished_num % 1000 == 0:
            final_result = {"data": data}
            result_file_name = ""
            for i in range(len(exchange)):
                result_file_name += str(exchange[i])

            save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/GLM4/{args.subset}_results_{result_file_name}.json'
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f)

            del final_result
            gc.collect()
            print('临时保存已完成')

    final_result = pd.DataFrame(data)
    result_file_name = ""
    for i in range(len(exchange)):
        result_file_name += str(exchange[i])
    save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/GLM4/{args.subset}_results_{result_file_name}.jsonl'
    final_result.to_json(save_path, orient='records', lines=True)

    del final_result
    gc.collect()




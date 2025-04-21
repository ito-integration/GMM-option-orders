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
import random

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--GPU', type=int, default=0, help='指定使用的GPU')
parser.add_argument('--begin', type=int, default=0, help='指定从哪个选项顺序开始')
parser.add_argument('--num', type=int, default=1, help='指定跑多少个选项顺序')
parser.add_argument('--prompt', type=int, default=0, help='指定prompt模板')
parser.add_argument('--dataset', type=int, default=0, help='指定数据集')
parser.add_argument('--subset', type=int, default=0, help='指定子集')
parser.add_argument('--restart', type=str, default='n', help='指定是否延续之前的结果进行')
# 解析参数，即从命令行获取这些待定参数
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9'
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
device_ids = list(range(args.GPU,10))
torch.cuda.set_device(9)


model_path = '/Data/transformers/llama2-7b'

config = LlamaConfig.from_pretrained(model_path)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) #加载到meta设备中，不需要耗时，不需要消耗内存和显存

device_map = device_map = OrderedDict([('', args.GPU)])
# device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes)

load_checkpoint_in_model(model,model_path,device_map=device_map) #加载权重

model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上

tokenizer = LlamaTokenizer.from_pretrained(model_path,padding_side="left")

torch.set_grad_enabled(False)
model.eval()


data_path = f'/Data/datasets/Option order results/{args.dataset}/raw data/{args.subset}.parquet'

if args.dataset == 'MMLU':
    knowledge_path = f'/Data/datasets/Option order results/{args.dataset}/raw data/dev.parquet'
else:
    knowledge_path = f'/Data/datasets/Option order results/{args.dataset}/raw data/train.parquet'


if args.dataset == 'CSQA':
    num_options = 5
else:
    num_options = 4

prompt_templates = ['%s\nChoose a correct option from the following options:%s',
                    '%s\nThe following are the candidate options:%s\nThe correct answer is',
                    '%s\nThe following are the candidate options:%s\nOutput the correct option without any analysis or explanation.',
                    'Question: %s\nOptions:%s\nIdentify the correct answer from the provided options. Please be concise in your response, NO VERBOSITY.',
                    'Question: %s\nOptions:%s\nAnswer the question without any explanation.']

prompt_template = prompt_templates[args.prompt]

letters = [chr(i) for i in range(65, 65+num_options)]

knowledge_base = pd.read_parquet(knowledge_path)
knowledge_base = knowledge_base.to_dict(orient='records')

if args.dataset == 'MMLU':
    indices_dict = {}
    subject_list = list(set(knowledge_base['subject'].values))
    for subject in subject_list:
        indices_in_subject = knowledge_base[knowledge_base['subject'] == subject].index.tolist()

        indices_dict[subject] = indices_in_subject
else:
    fixed_indices_dict = {'MedQA': [1, 15, 12, 0],
                          'MedMCQA': [1, 3, 0, 8],
                          'ARC': [0, 1, 6, 3],
                          'CSQA': [0, 1, 4, 3, 6]}

    shot_indices = fixed_indices_dict[args.dataset]

def format_choice(letters, labels):
    results = ""
    for i in range(len(letters)):
        results += f"\n{letters[i]}. {labels[i]}"
    return results

#构造问答中问询的文本
def construct_input_query(letters, prompt_template, item):
    options_text = format_choice(letters, item['choices'])

    input_query = prompt_template % (item['question'], options_text)

    return input_query

#为了更好地利用例题的特点，决定将其作为history，输入至chat中
def build_history(knowledge_base:list,indices, letters, prompt_template):
    history: List[Tuple[str,str]] = []
    if args.prompt == 4:
        system_prompt = "As a knowledgeable biomedical professional, your task is identifying the correct answer from the provided options. Please be concise in your response, NO VERBOSITY."
        history.append({'role': 'system', 'content': system_prompt})
    for indice in indices:
        labels = knowledge_base[indice]['choices']
        query = construct_input_query(letters, prompt_template, knowledge_base[indice])
        answer = f"{letters[knowledge_base[indice]['answer']]}. {labels[knowledge_base[indice]['answer']]}"
        history.append({'role': 'user', 'content': query})
        history.append({'role': 'assistant', 'content': answer})
    return history


def construct_input(dialog, tokenizer, model):
    if dialog[0]["role"] == "system":
        dialog = [
                     {
                         "role": dialog[1]["role"],
                         "content": B_SYS
                                    + dialog[0]["content"]
                                    + E_SYS
                                    + dialog[1]["content"],
                     }
                 ] + dialog[2:]

    dialog_text = ""
    for prompt, answer in zip(dialog[::2], dialog[1::2]):
        dialog_text += f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}</s>"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    dialog_tokens_1 = tokenizer([dialog_text], return_tensors="pt")

    if dialog[-1]['role'] == 'user':
        dialog_text = f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False
        dialog_tokens_2 = tokenizer([dialog_text], return_tensors="pt")

        dialog_tokens_2['input_ids'] = torch.cat((dialog_tokens_2['input_ids'], torch.tensor([[29871]])), dim=1)
        dialog_tokens_2['attention_mask'] = torch.cat((dialog_tokens_2['attention_mask'], torch.tensor([[1]])), dim=1)

        dialog_tokens_1['input_ids'] = torch.cat((dialog_tokens_1['input_ids'], dialog_tokens_2['input_ids']), dim=1)
        dialog_tokens_1['attention_mask'] = torch.cat(
            (dialog_tokens_1['attention_mask'], dialog_tokens_2['attention_mask']), dim=1)
    else:
        dialog_tokens_1['input_ids'] = torch.cat((dialog_tokens_1['input_ids'], torch.tensor([[29871]])), dim=1)
        dialog_tokens_1['attention_mask'] = torch.cat((dialog_tokens_1['attention_mask'], torch.tensor([[1]])), dim=1)

    return dialog_tokens_1.to(model.device)


def find_unstructured_answer(output,aim_token_ids_1,aim_token_ids_2):#从非结构化的文本中寻找答案（返回答案在输出的token_ids中的索引）
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


def find_position(input_ids,answer, aim_token_ids_1, aim_token_ids_2, letters):#从输入+输出+tokenizer化之后，寻找答案对应的logits向量的索引
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


if num_options == 4:
    orders_list = [[0, 1, 2, 3],
                   [3, 0, 1, 2],
                   [2, 3, 0, 1],
                   [1, 2, 3, 0]]
else:
    assert num_options == 5
    orders_list = [[0, 1, 2, 3, 4],
                   [4, 0, 1, 2, 3],
                   [3, 4, 0, 1, 2],
                   [2, 3, 4, 0, 1],
                   [1, 2, 3, 4, 0]]
orders_list = orders_list[args.begin:args.begin+args.num]
print(f'此次测试的选项顺序为：{orders_list}')


#循环过程中待统计的量
for exchange in orders_list:

    data = pd.read_parquet(data_path)
    data = data.to_dict(orient='records')

    if (args.dataset != 'MMLU') and (args.subset == 'train'):
        data = data.drop(index=shot_indices)
        data = data.reset_index(drop=True)

    aim_token_ids_1 = torch.tensor([29909, 29933, 29907, 29928, 29923]).to(model.device)  # ABCD有两套ids与之对应
    aim_token_ids_1 = aim_token_ids_1[:num_options]
    
    aim_token_ids_2 = torch.tensor([319, 350, 315, 360, 382]).to(model.device)
    aim_token_ids_2 = aim_token_ids_2[:num_options]

    if args.restart == 'n':
        finished_num = 0
    else:
        assert args.restart == 'y'
        finished_num = 0
        result_file_name = ""
        for i in range(len(exchange)):
            result_file_name += str(exchange[i])

        save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/prompt{args.prompt}_{args.subset}_results_{result_file_name}.jsonl'
        finished_num = 0
        if os.path.exists(save_path):
            finished_data = pd.read_json(save_path, orient='records', lines=True)
            finished_data = finished_data.to_dict(orient='records')
            finished_num = len(finished_data)

            # 将原始数据部分替换为已经完成的数据
            for n in range(finished_num):
                data[n] = finished_data[n]

        print(f'已完成{finished_num}个样本')

    for item in tqdm(data[finished_num:]):
        labels = item['choices'].tolist()
        item['choices'] = [labels[exchange.index(i)] for i in range(len(labels))]

        item['answer'] = exchange[item['answer']]

        finished_num+=1
        input_text = construct_input_query(letters, prompt_template, item)

        if args.dataset == 'MMLU':
            shot_indices = indices_dict[item['subject']]


        history = build_history(knowledge_base, shot_indices, letters, prompt_template)
        history.append({'role': 'user', 'content': input_text})

        inputs = construct_input(history, tokenizer, model)
        with torch.no_grad():
            outputs_set = model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                         pad_token_id=model.config.eos_token_id,
                                         return_dict_in_generate=True, output_scores=True
                                         )
        outputs = outputs_set.sequences.tolist()[0][len(inputs["input_ids"][0]):]
        logits = outputs_set.scores
        response = tokenizer.decode(outputs)

        if finished_num == 1:  # 打印第一道题的输入/输出示例
            print("\nExample: ")
            print("\nThe input pass to the model is:", tokenizer.decode(inputs["input_ids"][0]))
            print("\nThe output by chat is:", response)

        if (outputs[0] in aim_token_ids_1) or (outputs[0] in aim_token_ids_2):  # 如果输出的第一个字母是ABCD中的一个，则符合格式
            logits = logits[0]
        else:
            with torch.no_grad():
                outputs_set = model.generate(**inputs, max_new_tokens=20, do_sample=False, num_beams=1, temperature=None,
                                             top_p=None, return_dict_in_generate=True, output_scores=True)
            outputs = outputs_set.sequences.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            print(response)
            logits = outputs_set.scores
            target_indice = find_unstructured_answer(outputs, aim_token_ids_1, aim_token_ids_2)  # 该函数用于从文本中提取正确选项的索引
            logits = logits[target_indice]

        # torch.cuda.empty_cache()

        # 计算ABCD各项得分/概率，以及该题目的kl散度
        aim_token_ids_1 = aim_token_ids_1[:len(letters)]
        logits_trans_1 = logits[0, aim_token_ids_1]
        item['scores'] = logits_trans_1.tolist()

        if finished_num % 1000 == 0:
            save_df = pd.DataFrame(data[:finished_num])
            result_file_name = ""
            for i in range(len(exchange)):
                result_file_name += str(exchange[i])

            save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/prompt{args.prompt}_{args.subset}_results_{result_file_name}.jsonl'
            save_df.to_json(save_path, orient='records', lines=True)

            del save_df
            gc.collect()
            print('临时保存已完成')

    final_result = pd.DataFrame(data)
    result_file_name = ""
    for i in range(len(exchange)):
        result_file_name += str(exchange[i])
    save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/prompt{args.prompt}_{args.subset}_results_{result_file_name}.jsonl'
    final_result.to_json(save_path, orient='records', lines=True)

    del final_result
    gc.collect()

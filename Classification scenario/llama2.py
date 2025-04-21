import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
import os
import torch
import json
import codecs
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import math
from typing import List, Optional, Union, Dict, Tuple
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import sys
# from llama import Llama, Dialog
import numpy as np
from collections import OrderedDict
import itertools
import argparse
import subprocess
import gc

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--GPU', type=int, default=0, help='指定使用的GPU')
parser.add_argument('--begin', type=int, default=0, help='指定从哪个选项顺序开始')
parser.add_argument('--num', type=int, default=1, help='指定跑多少个选项顺序')
parser.add_argument('--dataset', type=str, default='SST-2', help='指定跑哪个数据集')
parser.add_argument('--subset', type=str, default='train', help='指定当前跑得是测试集还是训练集')
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

raw_file_formats = {'SST-2': 'jsonl', 'IMDB': 'parquet', 'Subj': 'jsonl', 'RTE': 'jsonl',
                   'MNLI': 'parquet', 'AG News': 'parquet', 'BBC News': 'jsonl', 'Emergency_TL': 'jsonl', 'MedNLI': 'parquet'}
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
                      'MedNLI': [0, 2, 1]
                      }


labels = labels_dict[args.dataset]
letters = [chr(i) for i in range(65,70)][:len(labels)]

knowledge_base_file = f'/Data/datasets/Option order results/{args.dataset}/raw data/train.' + file_format
if file_format == 'jsonl':
    knowledge_base = pd.read_json(knowledge_base_file, orient='records', lines=True)
else:
    knowledge_base = pd.read_parquet(knowledge_base_file)

# knowledge_base_labels = knowledge_base['label'].values
# 将knowledge_base变为字典列表
knowledge_base = knowledge_base.to_dict(orient='records')

shot_indices = fixed_indices_dict[args.dataset]

# 由于BBC News的shot样本过长，因此截取部分文本
if args.dataset == 'BBC News':
    for i in shot_indices:
        knowledge_base[i]['text'] = knowledge_base[i]['text'][:knowledge_base[i]['text'].find('.')]

# # shot_indices是训练集中那些被用作few-shot的样本
# shot_indices = []
# for y in range(len(letters)):
#     shot_indices.append(np.where(knowledge_base_labels == y)[0][0])


#查找与query最相近的索引，进而从知识库knowledge_base中找到对应的例题
def search_example(query: str,top_k: int,threshold: float):
    query = [query]
    with torch.no_grad():
        query_vector = embedding_model.encode(query,normalize_embeddings=True)
    distance, indices = index.search(query_vector.astype('float32'),top_k)
    #根据阈值进行筛选
    indices = indices[distance<=threshold]
    return indices.tolist()

#为了更好地利用例题的特点，决定将其作为history，输入至chat中
def build_history(knowledge_base:list,indices, datasets, letters, labels, prompt_templates):
    history: List[Tuple[str,str]] = []
    for indice in indices:
        query = construct_input_query(datasets, letters, labels, prompt_templates, knowledge_base[indice])
        history.append({'role': 'user', 'content': query})
        answer = f"{letters[exchange[knowledge_base[indice]['label']]]}. {labels[exchange[knowledge_base[indice]['label']]]}"
        history.append({'role': 'assistant', 'content': answer})
    return history

def format_choice(letters, labels):
    results = ""
    for i in range(len(letters)):
        results += f"\n{letters[i]}. {labels[i]}"
    return results

#构造问答中问询的文本
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
    elif datasets == 'MedNLI':
        input_query = prompt % (item['sentence1'], item['sentence2'], options_text)

    return input_query

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

# def find_unstructured_answer(answer):#从非结构化的文本中寻找答案（返回答案在输出文本中的索引）
#     Targets = letters
#     indices = []
#     for traget in Targets:
#         indices.append(answer.rfind(traget))
#     target_indice = max(indices)
#     if target_indice<0:#说明没找到，即文本中不含有任何ABCD
#         return 0
#     if len(answer)-1>target_indice:#如果找到的位置不是文本的最后一个位置
#         if answer[target_indice+1]==".":#要求这个字符的后面必须是.，以防止出现vitamin C的情况
#             return target_indice
#         else:#如果后面不是.，则由于找的是最大位置，去除该位置及之后的文本，再继续寻找。
#             return find_unstructured_answer(answer[0:target_indice])
#     else:#如果是最后一个位置的话，干脆返回这个字符
#         return target_indice
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

# def find_position(input_ids,answer):#从输入+输出+tokenizer化之后，寻找答案对应的logits向量的索引
#     target_id = tokenizer.convert_tokens_to_ids(answer)
#     answer_position = torch.where(torch.isin(input_ids,target_id))[0][-1]
#     return answer_position-1

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
        # 如果是AG News中的训练集，那么抽取其中的5%的样本即可；经验证，抽取的样本中不包含那些被指定为shot的样本
        # 之所以这么别扭是因为要复现之前的结果，完成复现后会修改！！！
        # TODO: 修改此处的逻辑
        if args.dataset == 'AG News':
            data = data.sample(frac=0.05, random_state=4396)
        # 如果不是AG News的训练集，移除那些被指定为shot的样本
        else:
            data = data.drop(index = shot_indices)
            data = data.reset_index(drop=True)

    data = data.to_dict(orient='records')

    if args.dataset =='BBC News':
        if args.subset == 'train':
            abnormal_indices = [191, 751]
            for abnormal_indice in abnormal_indices:
                data[abnormal_indice]['text'] = data[abnormal_indice]['text'][:data[abnormal_indice]['text'].find('.')]
        elif args.subset == 'test' and exchange == [2, 3, 4, 0, 1]:
            abnormal_indices = [875]
            for abnormal_indice in abnormal_indices:
                data[abnormal_indice]['text'] = data[abnormal_indice]['text'][:data[abnormal_indice]['text'].find('.')]

    
    labels = labels_dict[args.dataset]  # 重置为一个默认顺序
    labels = [labels[exchange.index(i)] for i in range(len(labels))]

    aim_token_ids_1 = torch.tensor([319,  350,  315,  360,  382,  368,  402,  361,  307,  432,  497,  380, 339,  390,  419,  346, 1070,  371,  314,  304,  466,  550,  357]).to(model.device)
    aim_token_ids_1 = aim_token_ids_1[:len(labels)]
    aim_token_ids_2 = torch.tensor([29909, 29933, 29907, 29928, 29923, 30960, 30964, 30956, 30936, 30977, 30984, 30957, 30944, 30958, 30961, 30947, 31015, 30951, 30937, 30935, 30976, 30985, 30959]).to(model.device)#ABCD有两套ids与之对应
    aim_token_ids_2 =aim_token_ids_2[:len(labels)]

    if args.restart == 'n':
        finished_num = 0
    else:
        assert args.restart == 'y'
        result_file_name = ""
        for i in range(len(exchange)):
            result_file_name += str(exchange[i])

        save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/{args.subset}_results_{result_file_name}.json'
        finished_num = 0
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
        finished_num += 1

        item['label'] = exchange[item['label']]

        input_text = construct_input_query(args.dataset, letters, labels, prompt_templates, item)
        
        history = build_history(knowledge_base, shot_indices, args.dataset, letters, labels, prompt_templates)
        history.append({'role': 'user', 'content': input_text})

        inputs = construct_input(history, tokenizer, model)
        with torch.no_grad():
            outputs_set = model.generate(**inputs, do_sample=False, max_new_tokens=20,
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

        if outputs[0] in aim_token_ids_1 or outputs[0] in aim_token_ids_2:  # 如果输出的第一个字母是ABCD中的一个，则符合格式
            logits = logits[0]
        else:
            # print("The unformatted response: ", response)
            target_indice = find_unstructured_answer(outputs, aim_token_ids_1, aim_token_ids_2)  # 该函数用于从文本中提取正确选项的索引
            # print("The extract letter is: ", tokenizer.decode(outputs[target_indice]))
            logits = logits[target_indice]

        torch.cuda.empty_cache()

        # 计算ABCD各项得分/概率，以及该题目的kl散度
        aim_token_ids_1 = aim_token_ids_1[:len(letters)]
        logits_trans_1 = logits[0, aim_token_ids_1]
        item['scores'] = logits_trans_1.tolist()

        if finished_num % 1000 == 0:
            final_result = {"data": data}
            result_file_name = ""
            for i in range(len(exchange)):
                result_file_name += str(exchange[i])

            save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/{args.subset}_results_{result_file_name}.json'
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f)

            del final_result
            gc.collect()
            print('临时保存已完成')

    final_result = pd.DataFrame(data)
    result_file_name = ""
    for i in range(len(exchange)):
        result_file_name += str(exchange[i])
    save_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/llama2/{args.subset}_results_{result_file_name}.jsonl'
    final_result.to_json(save_path, orient='records', lines=True)

    del final_result
    gc.collect()




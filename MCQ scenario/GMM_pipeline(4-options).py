# 该代码估计的GMM模型拥有两套参数：CC模型和CI模型不再共享一套参数
# 且整个模型建立在z-scores而非logic scores的基础之上
import json
import codecs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from patsy import dmatrices
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import datetime
import sys
from mpmath import mp
import random
import argparse
import torch
import gc
from scipy.stats import mode
import os


def cal_prob_of_majority(lambda_0, pi, rho, mu, sigma_square, z_scores, left_censor, right_censor):
    assert mu.shape == sigma_square.shape
    assert pi.shape[0] == mu.shape[1]
    assert rho.shape == mu[0, :, :].shape
    assert z_scores.shape[1] == mu.shape[2]

    N = z_scores.shape[0]
    C = mu.shape[1]
    K = mu.shape[2]

    # 确定j=0还是1对应于“是A/B/C/D”
    aim_mu_index_CC = np.argsort(np.mean(mu[0, :, :], axis=1))[-1]
    aim_mu_index_CI = np.argsort(np.mean(mu[1, :, :], axis=1))[-1]
    if aim_mu_index_CC == aim_mu_index_CI:
        aim_mu_index = aim_mu_index_CC
    else:
        print('error!')
        if lambda_0 > 0.5:
            aim_mu_index = aim_mu_index_CC
        else:
            aim_mu_index = aim_mu_index_CI

    if aim_mu_index == 0:
        inv_aim_mu_index = 1
    else:
        inv_aim_mu_index = 0

    norm_pds = norm.pdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1),
                        (N, 2, C, K)),
        np.broadcast_to(np.expand_dims(mu, axis=0), (N, 2, C, K)),
        np.broadcast_to(np.expand_dims(np.sqrt(sigma_square), axis=0), (N, 2, C, K)))
    # 记录所有的mu、sigma对应的左、右删失概率，是一个2*C*K的张量
    left_censor_probs = norm.cdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(left_censor, axis=0), axis=0), (2, C, K)),
        mu,
        np.sqrt(sigma_square)
    )
    right_censor_probs = 1 - norm.cdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(right_censor, axis=0), axis=0),
                        (2, C, K)),
        mu,
        np.sqrt(sigma_square)
    )

    # 删失概率是2*C*K张量；发现对于某些维度k，[1,:,k]对应的两个截断/删失概率都是0，这会在之后造成数值问题。因此，引入更高精度的CDF
    # 找到那些两个c均为0的k
    error_index = np.where((left_censor_probs[1, :, :] == 0).all(axis=0))[0]
    if error_index.shape[0] > 0:
        mp.dps = 100
        # 注意：logProds的尺寸是 len(error_index)*num_pattern 而非 num_pattern*len(error_index)
        logProbs = norm.logcdf(
            np.broadcast_to(np.expand_dims(left_censor[error_index], axis=1), mu[1, :, error_index].shape),
            mu[1, :, error_index],
            np.sqrt(sigma_square[1, :, error_index]))
        for j in range(C):
            for k in range(error_index.shape[0]):
                logProb = mp.mpf(f'{logProbs[k, j]}')
                Prob = mp.exp(logProb)
                left_censor_probs[1, j, error_index[k]] = Prob

    error_index = np.where((right_censor_probs[1, :, :] == 0).all(axis=0))[0]
    if error_index.shape[0] > 0:
        mp.dps = 100
        # 注意：logProds的尺寸是len(error_index)*num_pattern 而非 num_pattern*len(error_index)
        logProbs = norm.logcdf(
            np.broadcast_to(np.expand_dims(right_censor[error_index], axis=1), mu[1, :, error_index].shape),
            mu[1, :, error_index],
            np.sqrt(sigma_square[1, :, error_index]))
        for j in range(C):
            for k in range(error_index.shape[0]):
                logProb = mp.mpf(f'{logProbs[k, j]}')
                Prob = mp.exp(logProb)
                right_censor_probs[1, j, error_index[k]] = 1 - Prob

    # 记录所有的删失元素的索引：查看是哪个样本的哪个维度发生了删失
    left_censor_index = np.where(z_scores == left_censor)
    right_censor_index = np.where(z_scores == right_censor)

    # 替换pdf为删失概率
    if left_censor_index[0].shape[0] != 0:
        norm_pds[left_censor_index[0], :, :, left_censor_index[1]] = left_censor_probs[:, :,
                                                                     left_censor_index[1]].transpose(2, 0, 1)
    if right_censor_index[0].shape[0] != 0:
        norm_pds[right_censor_index[0], :, :, right_censor_index[1]] = right_censor_probs[:, :,
                                                                       right_censor_index[1]].transpose(2, 0, 1)

    # 记录了服从CC模型且属于第j类的概率，是一个N维数组
    P_CC = lambda_0 * pi[aim_mu_index] * np.prod(norm_pds[:,0, aim_mu_index, ], axis=1)

    # 一个N*C*K的张量
    joint_distribution = [rho]*norm_pds[:,1,:,:]
    # 一个N*K的矩阵，记录了在CI模型中，第n个样本在第k个维度的概率密度/删失概率
    sum_joint = np.sum(joint_distribution, axis=1)

    # joint_distribution = np.array([[rho[j,k]*norm.pdf(z_score[k], mu[1,j,k], np.sqrt(sigma_square[1,j,k])) for k in range(K)]for j in range(C)])
    # sum_joint = np.sum(joint_distribution, axis=0) #对j求和

    Poisson_Binomial_distribution = np.zeros((N, K, K + 1))
    for k in range(K):
        Poisson_Binomial_distribution[:, k, 0] = np.prod(joint_distribution[:,inv_aim_mu_index,0:k+1], axis=1)*np.prod(sum_joint[:,k+1:], axis=1)
        Poisson_Binomial_distribution[:, k, k + 1] = np.prod(joint_distribution[:,aim_mu_index,0:k+1], axis=1)*np.prod(sum_joint[:,k+1:], axis=1)
    for k in range(1, K):
        for i in range(1, k + 1):
            # try:
            #     assert Poisson_Binomial_distribution[k - 1, i - 1] != 0, f"k-1: {k-1}, i-1: {i-1}"
            #     assert Poisson_Binomial_distribution[k - 1, i] != 0, f"k-1: {k-1}, i:{i}"
            # except AssertionError as e:
            #     print(e)
            Poisson_Binomial_distribution[:,k,i] = (joint_distribution[:, aim_mu_index, k]*Poisson_Binomial_distribution[:, k-1, i-1]
            + joint_distribution[:, inv_aim_mu_index, k]*Poisson_Binomial_distribution[:, k-1, i])/sum_joint[:, k]
    # 现在Poisson_Binomial_distribution的最后一行记录了完整的分布
    # 求超过半数的顺序属于j的概率
    P_CI = (1 - lambda_0) * np.sum(Poisson_Binomial_distribution[:, -1, (K+1) // 2:], axis=1)

    f_CCs = np.sum(np.prod(norm_pds[:, 0, :, :], axis=2) * [pi], axis=1)
    f_CIs = np.prod(np.sum(norm_pds[:, 1, :, :] * [rho], axis=1), axis=1)
    marginal_distribution_of_z_scores = lambda_0 * f_CCs + (1 - lambda_0) * f_CIs

    return (P_CC + P_CI) / marginal_distribution_of_z_scores

def cal_prob_of_incons(lambda_0, pi, rho, mu, sigma_square, z_scores, left_censor, right_censor):
    assert mu.shape == sigma_square.shape
    assert pi.shape[0] == mu.shape[1]
    assert rho.shape == mu[0, :, :].shape
    assert z_scores.shape[1] == mu.shape[2]

    N = z_scores.shape[0]
    C = mu.shape[1]
    K = mu.shape[2]

    # 确定j=0还是1对应于“是A/B/C/D”
    aim_mu_index_CC = np.argsort(np.mean(mu[0, :, :], axis=1))[-1]
    aim_mu_index_CI = np.argsort(np.mean(mu[1, :, :], axis=1))[-1]
    if aim_mu_index_CC == aim_mu_index_CI:
        aim_mu_index = aim_mu_index_CC
    else:
        print('error!')
        if lambda_0 > 0.5:
            aim_mu_index = aim_mu_index_CC
        else:
            aim_mu_index = aim_mu_index_CI

    if aim_mu_index == 0:
        inv_aim_mu_index = 1
    else:
        inv_aim_mu_index = 0

    norm_pds = norm.pdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1),
                        (N, 2, C, K)),
        np.broadcast_to(np.expand_dims(mu, axis=0), (N, 2, C, K)),
        np.broadcast_to(np.expand_dims(np.sqrt(sigma_square), axis=0), (N, 2, C, K)))
    # 记录所有的mu、sigma对应的左、右删失概率，是一个2*C*K的张量
    left_censor_probs = norm.cdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(left_censor, axis=0), axis=0), (2, C, K)),
        mu,
        np.sqrt(sigma_square)
    )
    right_censor_probs = 1 - norm.cdf(
        np.broadcast_to(np.expand_dims(np.expand_dims(right_censor, axis=0), axis=0),
                        (2, C, K)),
        mu,
        np.sqrt(sigma_square)
    )

    # 删失概率是2*C*K张量；发现对于某些维度k，[1,:,k]对应的两个截断/删失概率都是0，这会在之后造成数值问题。因此，引入更高精度的CDF
    # 找到那些两个c均为0的k
    error_index = np.where((left_censor_probs[1, :, :] == 0).all(axis=0))[0]
    if error_index.shape[0] > 0:
        mp.dps = 100
        # 注意：logProds的尺寸是 len(error_index)*num_pattern 而非 num_pattern*len(error_index)
        logProbs = norm.logcdf(
            np.broadcast_to(np.expand_dims(left_censor[error_index], axis=1), mu[1, :, error_index].shape),
            mu[1, :, error_index],
            np.sqrt(sigma_square[1, :, error_index]))
        for j in range(C):
            for k in range(error_index.shape[0]):
                logProb = mp.mpf(f'{logProbs[k, j]}')
                Prob = mp.exp(logProb)
                left_censor_probs[1, j, error_index[k]] = Prob

    error_index = np.where((right_censor_probs[1, :, :] == 0).all(axis=0))[0]
    if error_index.shape[0] > 0:
        mp.dps = 100
        # 注意：logProds的尺寸是len(error_index)*num_pattern 而非 num_pattern*len(error_index)
        logProbs = norm.logcdf(
            np.broadcast_to(np.expand_dims(right_censor[error_index], axis=1), mu[1, :, error_index].shape),
            mu[1, :, error_index],
            np.sqrt(sigma_square[1, :, error_index]))
        for j in range(C):
            for k in range(error_index.shape[0]):
                logProb = mp.mpf(f'{logProbs[k, j]}')
                Prob = mp.exp(logProb)
                right_censor_probs[1, j, error_index[k]] = 1 - Prob

    # 记录所有的删失元素的索引：查看是哪个样本的哪个维度发生了删失
    left_censor_index = np.where(z_scores == left_censor)
    right_censor_index = np.where(z_scores == right_censor)

    # 替换pdf为删失概率
    if left_censor_index[0].shape[0] != 0:
        norm_pds[left_censor_index[0], :, :, left_censor_index[1]] = left_censor_probs[:, :,
                                                                     left_censor_index[1]].transpose(2, 0, 1)
    if right_censor_index[0].shape[0] != 0:
        norm_pds[right_censor_index[0], :, :, right_censor_index[1]] = right_censor_probs[:, :,
                                                                       right_censor_index[1]].transpose(2, 0, 1)

    f_CCs = np.sum(np.prod(norm_pds[:, 0, :, :], axis=2) * [pi], axis=1)
    f_CIs = np.prod(np.sum(norm_pds[:, 1, :, :] * [rho], axis=1), axis=1)
    marginal_distribution_of_z_scores = lambda_0 * f_CCs + (1 - lambda_0) * f_CIs

    cordance_in_CI = np.sum(np.prod(norm_pds[:, 1, :, :]*[rho], axis=2), axis=1)
    discordance_prob = (1-lambda_0)*(f_CIs-cordance_in_CI)

    return discordance_prob / marginal_distribution_of_z_scores



def data_process(path1, path2, order_list, for_GMM=True):
    train_data = []
    test_data = []

    for order in order_list:
        file_suffix = ''
        for i in order:
            file_suffix += str(i)
        train_path = path1 + file_suffix + '.jsonl'
        test_path = path2 + file_suffix + '.jsonl'
        train_df = pd.read_json(train_path, orient='records', lines=True)
        test_df = pd.read_json(test_path, orient='records', lines=True)

        # if 'probability' in train_df.columns and 'probability' in test_df.columns:
        #     train_probs = np.array(train_df['probability'].values.tolist())
        #     test_probs = np.array(test_df['probability'].values.tolist())
        #
        # else:
        train_logit_scores = np.array(train_df['scores'].values.tolist())
        test_logit_scores = np.array(test_df['scores'].values.tolist())

        train_logit_scores = train_logit_scores - np.max(train_logit_scores, axis=1, keepdims=True)
        test_logit_scores = test_logit_scores - np.max(test_logit_scores, axis=1, keepdims=True)

        train_probs = np.exp(train_logit_scores)
        train_probs = train_probs / np.sum(train_probs, axis=1, keepdims=True)
        test_probs = np.exp(test_logit_scores)
        test_probs = test_probs / np.sum(test_probs, axis=1, keepdims=True)

        # train_probs = train_probs / np.sum(train_probs, axis=1, keepdims=True)
        # test_probs = test_probs / np.sum(test_probs, axis=1, keepdims=True)

        # 对异常值进行处理
        abnormal_indices = np.where(train_probs == 1)
        train_probs[abnormal_indices[0], abnormal_indices[1]] = 1 - 1e-6
        abnormal_indices = np.where(train_probs == 0)
        train_probs[abnormal_indices[0], abnormal_indices[1]] = 1e-6

        abnormal_indices = np.where(test_probs == 1)
        test_probs[abnormal_indices[0], abnormal_indices[1]] = 1 - 1e-6
        abnormal_indices = np.where(test_probs == 0)
        test_probs[abnormal_indices[0], abnormal_indices[1]] = 1e-6

        # 将每一列按照原始的ABCD进行排列
        train_probs_swap = np.zeros(train_probs.shape)
        for i in order:
            train_probs_swap[:, i] = train_probs[:, order[i]]

        test_probs_swap = np.zeros(test_probs.shape)
        for i in order:
            test_probs_swap[:, i] = test_probs[:, order[i]]

        train_data.append(train_probs_swap.tolist())
        test_data.append(test_probs_swap.tolist())

        if order == order_list[0]:
            if 'answer' in test_df.columns:
                test_labels = test_df['answer'].values
                train_labels = train_df['answer'].values
            else:
                test_labels = test_df['label'].values
                train_labels = train_df['label'].values

    # # data的形状是 样本数量 * 顺序数量 * 选项数量
    # # data[n, k, :] 的含义是：第 n 个样本在第 k 个选项顺序下，原始的4个选项的概率
    train_data = np.array(train_data)
    train_data = train_data.transpose(1, 0, 2)
    test_data = np.array(test_data)
    test_data = test_data.transpose(1, 0, 2)

    ###############################################################################################################
    if for_GMM:
        # # 下面我们要把data[n, j, :] 的含义变为：第n个样本的第j个标签（原始顺序下的第j个）在1/2/3/4个位置下的选项顺序
        new_train_data = np.zeros(train_data.shape)
        new_test_data = np.zeros(test_data.shape)
        order_list = np.array(order_list)
        for j in range(train_data.shape[2]):
            positions = order_list[:, j].tolist()
            order_in_the_label = [positions.index(p) for p in range(train_data.shape[2])]
            new_train_data[:, j, :] = train_data[:, order_in_the_label, j]
            new_test_data[:, j, :] = test_data[:, order_in_the_label, j]
    else:
        new_train_data = train_data
        new_test_data = test_data

    return new_train_data, new_test_data, train_labels, test_labels


# Processing the outputs of LLM to adapt them to PriDe
def data_process_for_pride(path1, path2, order_list):
    train_data = []
    test_data = []

    for order in order_list:
        file_suffix = ''
        for i in order:
            file_suffix += str(i)
        train_path = path1 + file_suffix + '.jsonl'
        test_path = path2 + file_suffix + '.jsonl'

        train_df = pd.read_json(train_path, orient='records', lines=True)
        test_df = pd.read_json(test_path, orient='records', lines=True)

        train_logit_scores = np.array(train_df['scores'].values.tolist())
        test_logit_scores = np.array(test_df['scores'].values.tolist())

        train_logit_scores = train_logit_scores - np.max(train_logit_scores, axis=1, keepdims=True)
        test_logit_scores = test_logit_scores - np.max(test_logit_scores, axis=1, keepdims=True)

        train_probs = np.exp(train_logit_scores)
        train_probs = train_probs / np.sum(train_probs, axis=1, keepdims=True)
        test_probs = np.exp(test_logit_scores)
        test_probs = test_probs / np.sum(test_probs, axis=1, keepdims=True)

        # train_probs = train_probs / np.sum(train_probs, axis=1, keepdims=True)
        # test_probs = test_probs / np.sum(test_probs, axis=1, keepdims=True)


        # 对异常值进行处理
        abnormal_indices = np.where(train_probs == 1)
        train_probs[abnormal_indices[0], abnormal_indices[1]] = 1 - 1e-6
        abnormal_indices = np.where(train_probs == 0)
        train_probs[abnormal_indices[0], abnormal_indices[1]] = 1e-6

        abnormal_indices = np.where(test_probs == 1)
        test_probs[abnormal_indices[0], abnormal_indices[1]] = 1 - 1e-6
        abnormal_indices = np.where(test_probs == 0)
        test_probs[abnormal_indices[0], abnormal_indices[1]] = 1e-6

        train_data.append(train_probs.tolist())
        test_data.append(test_probs.tolist())

        if order == order_list[0]:
            test_labels = test_df['label'].values

    train_data = np.array(train_data)
    train_data = train_data.transpose(1, 0, 2)
    test_data = np.array(test_data)
    test_data = test_data.transpose(1, 0, 2)

    return train_data, test_data, test_labels

def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True))
    return x


migrate_dataset = 'MMLU'
models = ['llama2', 'GLM4', 'llama3']
datasets = ['MMLU', 'ARC', 'MedQA', 'MedMCQA']
datasets = datasets[:datasets.index(migrate_dataset)] + datasets[datasets.index(migrate_dataset) + 1:]
all_orders = [[0, 1, 2, 3],
              [3, 0, 1, 2],
              [2, 3, 0, 1],
              [1, 2, 3, 0]]

prompt_id_dict = {}
for model in models:
    for dataset in datasets:
        prompt_id_dict[f'{model}-{dataset}'] = 2
        if model == 'llama2' and dataset in ['MedQA', 'MedMCQA']:
            prompt_id_dict[f'{model}-{dataset}'] = 4
        elif model == 'llama3' and dataset == 'MedQA':
            prompt_id_dict[f'{model}-{dataset}'] = 0

for model in models:

    # Using coefficients estimated based on the held-out set in MMLU dataset
    result_path = f'/Data/datasets/Option order results/{migrate_dataset}/calibrated results/{model}/ALL_single_results.json'

    with open(result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    threshold = result['thresholds']

    lambda_0 = result['GMM_coefs']['lambda']
    pi = np.array(result['GMM_coefs']['pi'])
    rho = np.array(result['GMM_coefs']['rho'])
    mu = np.array(result['GMM_coefs']['mu'])
    sigma_square = np.array(result['GMM_coefs']['sigma'])

    for dataset in datasets:
        prompt_id = prompt_id_dict[f'{model}-{dataset}']
        train_base_path = f'/Data/datasets/Option order results/{dataset}/LLM results/{model}/prompt{prompt_id}_train_results_'
        test_base_path = f'/Data/datasets/Option order results/{dataset}/LLM results/{model}/prompt{prompt_id}_test_results_'

        _, test_data, _, _ = data_process(train_base_path, test_base_path, all_orders, for_GMM=True)

        inconsistency_scores = np.zeros((test_data.shape[0], test_data.shape[1]))
        GMM_scores = np.zeros((test_data.shape[0], test_data.shape[1]))

        for aim in range(test_data.shape[1]):
            test_z_scores = norm.ppf(test_data[:, aim, :])
            GMM_scores[:, aim] = cal_prob_of_majority(lambda_0, pi, rho, mu, sigma_square, test_z_scores,
                                                      left_censor=None, right_censor=None)
            inconsistency_scores[:, aim] = cal_prob_of_incons(lambda_0, pi, rho, mu, sigma_square, test_z_scores,
                                                              left_censor=None, right_censor=None)

        ### 将参数迁移对应的输出（预测输出、不一致性分数）保存在相应的文件中
        migration_result = {}

        migration_result['inconsistent_scores'] = inconsistency_scores.tolist()
        migration_result['thresholds'] = threshold
        migration_result['GMM_scores'] = GMM_scores.tolist()

        avg_pre = np.argmax(np.mean(test_data, axis=2), axis=1)
        GMM_pre = np.argmax(GMM_scores, axis=1)

        _, test_data, _, test_labels = data_process(train_base_path, test_base_path, all_orders, for_GMM=False)
        raw_pre = np.argmax(test_data, axis=2)
        MV_pre, _ = mode(raw_pre, axis=1)

        raw_accs = []
        for k in range(test_data.shape[1]):
            raw_accs.append(accuracy_score(test_labels, raw_pre[:, k]))

        GMM_acc = accuracy_score(test_labels, GMM_pre)
        migration_result['GMM_acc'] = GMM_acc
        avg_acc = accuracy_score(test_labels, avg_pre)
        migration_result['mean_acc'] = avg_acc
        MV_acc = accuracy_score(test_labels, MV_pre)
        migration_result['MV_acc'] = MV_acc

        migration_result['raw_accs'] = raw_accs

        train_data, test_data, test_labels = data_process_for_pride(train_base_path, test_base_path, all_orders)
        pride_prior = np.mean(softmax(np.mean(np.log(train_data), axis=1)), axis=0).reshape(1, -1)
        pride_pre = np.argmax(test_data[:, 0, :] / pride_prior, axis=1)
        pride_acc = accuracy_score(test_labels, pride_pre)

        migration_result_path = f'/Data/datasets/Option order results/{dataset}/calibrated results(coefs migration)/{model}/single_results.json'
        with open(migration_result_path, 'w', encoding='utf-8') as f:
            json.dump(migration_result, f)













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

# 使用高精度mp计算截断正态分布的期望
def truncated_normal_mean_mpmath(a, b, mu=0, sigma=1):
    mp.dps = 100

    cdf_of_a = np.zeros(mu.shape)
    cdf_of_b = np.ones(mu.shape)
    pdf_of_a = np.zeros(mu.shape)
    pdf_of_b = np.zeros(mu.shape)
    scales = np.zeros(mu.shape)

    # a 和 b 输入进来就已经对应于标准正态分布了
    mp.dps = 500
    if np.any(a == -np.inf):
        logcdfs_of_b = norm.logcdf(b, 0, 1)
        logpdfs_of_b = norm.logpdf(b, 0, 1)

        logscales = logpdfs_of_b - logcdfs_of_b

        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                for k in range(mu.shape[2]):
                    logscale = mp.mpf(f'{logscales[i, j, k]}')
                    scale = -mp.exp(logscale)
                    scales[i, j, k] = scale

    elif np.any(b == np.inf):
        logcdfs_of_a = norm.logcdf(a, 0, 1)
        logpdfs_of_a = norm.logpdf(a, 0, 1)

        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                for k in range(mu.shape[2]):
                    logpdf_of_a = mp.mpf(f'{logpdfs_of_a[i, j, k]}')
                    logcdf_of_a = mp.mpf(f'{logcdfs_of_a[i, j, k]}')
                    scale = mp.exp(logpdf_of_a) / (1 - mp.exp(logcdf_of_a))
                    print(f'{i} and {j} and {k}')
                    scales[i, j, k] = 1/scale
    else:
        logcdfs_of_a = norm.logcdf(a, 0, 1)
        logpdfs_of_a = norm.logpdf(a, 0, 1)
        logcdfs_of_b = norm.logcdf(b, 0, 1)
        logpdfs_of_b = norm.logpdf(b, 0, 1)

        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                for k in range(mu.shape[2]):
                    logpdf_of_a = mp.mpf(f'{logpdfs_of_a[i, j, k]}')
                    logcdf_of_a = mp.mpf(f'{logcdfs_of_a[i, j, k]}')
                    logpdf_of_b = mp.mpf(f'{logpdfs_of_b[i, j, k]}')
                    logcdf_of_b = mp.mpf(f'{logcdfs_of_b[i, j, k]}')

                    scale = (mp.exp(logpdf_of_a) - mp.exp(logpdf_of_b)) / (mp.exp(logcdf_of_b) - mp.exp(logcdf_of_a))
                    scales[i, j, k] = 1 / scale

    return mu + sigma * scales




# Using EM algorithm to estimate coefficients (uncensored data)
def EM(z_scores, lambda_0, rho, pi, mu, sigma_square, max_iteration, left_censor=None, right_censor=None):  # z_scores的记录方式：每一行代表一个样本，每一列代表一种顺序
    # 假设left_censor和right_censor是K维向量，代表每个维度的删失值
    assert mu.shape == sigma_square.shape
    assert rho.shape == mu[0,:,:].shape
    assert rho.shape[1] == z_scores.shape[1]
    assert mu.shape[1] == pi.shape[0]

    num_sample = z_scores.shape[0]
    num_pattern = mu.shape[1]
    num_permutation = mu.shape[2]

    loglikelihood = []

    for num in range(max_iteration):
        # E-step
        ########################################################################################################################

        # 计算所有的单个正态密度/删失概率，储存为一个N*2*C*K的矩阵
        norm_pds = norm.pdf(
            np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)),
            np.broadcast_to(np.expand_dims(mu, axis=0), (num_sample, 2, num_pattern, num_permutation)),
            np.broadcast_to(np.expand_dims(np.sqrt(sigma_square), axis=0), (num_sample, 2, num_pattern, num_permutation)))

        # print(f'计算所有概率密度/删失概率所耗时间为：{end_time-start_time}')

        # print(f'np.prod(norm_pds[:,0,:,:], axis=2): {np.prod(norm_pds[:,0,:,:], axis=2)}')
        f_CCs = np.sum(np.prod(norm_pds[:,0,:,:], axis=2)*[pi], axis=1)
        f_CIs = np.prod(np.sum(norm_pds[:,1,:,:]*[rho], axis=1), axis=1)
        marginal_distribution_of_z_scores = lambda_0*f_CCs + (1-lambda_0)*f_CIs
        loglikelihood.append(np.sum(np.log(marginal_distribution_of_z_scores)))



        # except_omega的第i个元素代表了E[ω_i]
        expect_omega = (lambda_0*f_CCs)/marginal_distribution_of_z_scores

        expect_eta = 1-expect_omega
        expect_eta = np.expand_dims(np.expand_dims(expect_eta, axis=1), axis=1)
        # 现在expect_eta是一个N*C*K的张量
        expect_eta = np.broadcast_to(expect_eta, (num_sample, num_pattern, num_permutation)).copy()
        factors = norm_pds[:, 1, :, :]*[rho]

        # 高精度地计算后验分布（备选方案：强制使后验分布为均匀分布）
        error_index = np.where(np.sum(factors, axis=1)==0)
        if error_index[0].shape[0] > 0:
            # factors[error_index[0],:,error_index[1]] = (1/num_pattern)*np.ones(num_pattern)
            mp.dps = 100
            # 注意：logProds的尺寸是 len(error_index)*num_pattern
            logPDs = norm.logpdf(np.expand_dims(z_scores[error_index[0], error_index[1]], axis=1),
                                 mu[1, :, error_index[1]],
                                 np.sqrt(sigma_square[1, :, error_index[1]]))
            # 这里需要改变factors，高精度地计算概率密度的比；直接改变norm_pds会由于numpy精度不足而失败。
            for n in range(error_index[0].shape[0]):
                mpmatrix = []
                for j in range(num_pattern):
                    logPD = mp.mpf(f'{logPDs[n, j]}')
                    PD = mp.exp(logPD)
                    mpmatrix.append(PD * rho[j, error_index[1][n]])
                local_sum = mp.fsum(mpmatrix)
                for j in range(num_pattern):
                    factors[error_index[0][n], j, error_index[1][n]] = mpmatrix[j] / local_sum

        factors = factors/np.sum(factors, axis=1, keepdims=True)
        expect_eta = expect_eta*factors

        # expect_xi是一个N*C的矩阵
        expect_xi = lambda_0*np.prod(norm_pds[:,0,:,:], axis=2)
        factors = pi.reshape(1, -1)/marginal_distribution_of_z_scores.reshape(-1, 1)
        expect_xi = factors*expect_xi

        # M-step
        ##########################################################################################################################

        renew_expect = np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)).copy()

        lambda_0 = np.mean(expect_omega)

        rho = np.sum(expect_eta, axis=0)/[np.sum(np.sum(expect_eta, axis=1), axis=0)]

        pi = np.sum(expect_xi, axis=0)/np.sum(expect_xi)

        renew_mu = np.zeros(mu.shape)

        # 开始更新mu[0,:,:]
        elements = np.broadcast_to(np.expand_dims(expect_xi, axis=2), (num_sample, num_pattern, num_permutation))*renew_expect[:,0,:,:]
        norm_factor = np.sum(expect_xi, axis=0).reshape(-1, 1)
        renew_mu[0,:,:] = np.sum(elements, axis=0)/norm_factor

        # 开始更新mu[1,:,:]
        elements = expect_eta*renew_expect[:,1,:,:]
        renew_mu[1,:,:] = np.sum(elements, axis=0)/np.sum(expect_eta, axis=0)



        # 首先，以z_score与期望的差的平方作为更新sigma时所需的参数
        renew_var = np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)).copy()
        renew_var = (renew_var - np.expand_dims(renew_mu, axis=0))**2


        # 过去的参数mu已经完成了历史使命
        mu = renew_mu.copy()

        # 开始更新sigma_square[0,:,:]
        elements = np.broadcast_to(np.expand_dims(expect_xi, axis=2),
                                   (num_sample, num_pattern, num_permutation)) * renew_var[:, 0, :, :]
        norm_factor = np.sum(expect_xi, axis=0).reshape(-1, 1)
        sigma_square[0, :, :] = np.sum(elements, axis=0) / norm_factor

        # 开始更新sigma_square[1,:,:]
        elements = expect_eta * renew_var[:, 1, :, :]
        sigma_square[1, :, :] = np.sum(elements, axis=0) / np.sum(expect_eta, axis=0)

        # The remain following code is for debug
        error_index = np.where(sigma_square == 0)
        if error_index[0].shape[0] > 0:
            # 只考察1个错误的方差来源
            if error_index[0][0] == 0:
                print('问题出在CC模型')
                the_j = error_index[1][0]
                the_k = error_index[2][0]
                print(f'这个顺序下当前估计的期望为\n{mu[0, :, the_k]}')
                print(f'这个顺序下当前估计的方差为\n{sigma_square[0, :, the_k]}')
                print(f'对应的平方差为：\n{renew_var[:, 0, the_j, the_k]}')
                print(f'对应的权重为：\n{expect_xi[:,the_j]}')
            else:
                assert error_index[0][0] == 1
                print('问题出在CI模型')
                the_j = error_index[1][0]
                the_k = error_index[2][0]
                print(f'这个顺序下当前估计的期望为\n{mu[1, :, the_k]}')
                print(f'这个顺序下当前估计的方差为\n{sigma_square[1, :, the_k]}')
                print(f'对应的平方差为：\n{renew_var[:, 1, the_j, the_k]}')
                print(f'对应的权重为：\n{expect_eta[:, the_j, the_k]}')

    # if np.all(np.diff(loglikelihood) > 0):
    #     print('True! ')
    # else:
    #     print('False! ')
    #     print(loglikelihood)
    return lambda_0, rho, pi, mu, sigma_square

# Using EM algorithm to estimate coefficients (censored data)
def EM_with_censor(z_scores, lambda_0, rho, pi, mu, sigma_square, max_iteration, left_censor=None, right_censor=None):
    # 假设left_censor和right_censor是K维向量，代表每个维度的删失值
    assert mu.shape == sigma_square.shape
    assert rho.shape == mu[0,:,:].shape
    assert rho.shape[1] == z_scores.shape[1]
    assert mu.shape[1] == pi.shape[0]

    num_sample = z_scores.shape[0]
    num_pattern = mu.shape[1]
    num_permutation = mu.shape[2]

    for num in range(max_iteration):
        # E-step
        ########################################################################################################################


        # 计算所有的单个正态密度/删失概率，储存为一个N*2*C*K的矩阵
        norm_pds = norm.pdf(
            np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)),
            np.broadcast_to(np.expand_dims(mu, axis=0), (num_sample, 2, num_pattern, num_permutation)),
            np.broadcast_to(np.expand_dims(np.sqrt(sigma_square), axis=0), (num_sample, 2, num_pattern, num_permutation)))
        # 记录所有的mu、sigma对应的左、右删失概率，是一个2*C*K的张量
        left_censor_probs = norm.cdf(
            np.broadcast_to(np.expand_dims(np.expand_dims(left_censor, axis=0), axis=0), (2, num_pattern, num_permutation)),
            mu,
            np.sqrt(sigma_square)
        )
        right_censor_probs = 1 - norm.cdf(
            np.broadcast_to(np.expand_dims(np.expand_dims(right_censor, axis=0), axis=0), (2, num_pattern, num_permutation)),
            mu,
            np.sqrt(sigma_square)
        )

        # 删失概率是2*C*K张量；发现对于某些维度k，[1,:,k]对应的两个截断/删失概率都是0，这会在之后造成数值问题。因此，引入更高精度的CDF
        # 找到那些两个c均为0的k
        error_index = np.where((left_censor_probs[1,:,:] == 0).all(axis=0))[0]
        if error_index.shape[0]>0:
            mp.dps = 100
            # 注意：logProds的尺寸是 len(error_index)*num_pattern 而非 num_pattern*len(error_index)
            logProbs = norm.logcdf(np.broadcast_to(np.expand_dims(left_censor[error_index], axis=1), mu[1, :, error_index].shape),
                                   mu[1,:,error_index],
                                   np.sqrt(sigma_square[1,:,error_index]))
            for j in range(num_pattern):
                for k in range(error_index.shape[0]):
                    logProb = mp.mpf(f'{logProbs[k, j]}')
                    Prob = mp.exp(logProb)
                    left_censor_probs[1,j,error_index[k]] = Prob

        error_index = np.where((right_censor_probs[1, :, :] == 0).all(axis=0))[0]
        if error_index.shape[0] > 0:
            mp.dps = 100
            # 注意：logProds的尺寸是len(error_index)*num_pattern 而非 num_pattern*len(error_index)
            logProbs = norm.logcdf(np.broadcast_to(np.expand_dims(right_censor[error_index], axis=1), mu[1, :, error_index].shape),
                                   mu[1, :, error_index],
                                   np.sqrt(sigma_square[1, :, error_index]))
            for j in range(num_pattern):
                for k in range(error_index.shape[0]):
                    logProb = mp.mpf(f'{logProbs[k, j]}')
                    Prob = mp.exp(logProb)
                    right_censor_probs[1, j, error_index[k]] = 1 - Prob

        # 记录所有的删失元素的索引：查看是哪个样本的哪个维度发生了删失
        left_censor_index = np.where(z_scores == left_censor)
        right_censor_index = np.where(z_scores == right_censor)

        # 替换pdf为删失概率
        if left_censor_index[0].shape[0] != 0:
            norm_pds[left_censor_index[0], :, :, left_censor_index[1]] = left_censor_probs[:,:,left_censor_index[1]].transpose(2,0,1)
        if right_censor_index[0].shape[0] != 0:
            norm_pds[right_censor_index[0], :, :, right_censor_index[1]] = right_censor_probs[:,:,right_censor_index[1]].transpose(2,0,1)



        f_CCs = np.sum(np.prod(norm_pds[:,0,:,:], axis=2)*[pi], axis=1)
        f_CIs = np.prod(np.sum(norm_pds[:,1,:,:]*[rho], axis=1), axis=1)
        marginal_distribution_of_z_scores = lambda_0*f_CCs + (1-lambda_0)*f_CIs

        # 发现有可能发生数值下溢问题，因此，把marginal_distribution中所有等于0的元素变为其中的最小值
        error_index = np.where(marginal_distribution_of_z_scores == 0)[0]
        if error_index.shape[0] > 0:
            print(f"问题样本有：{error_index}")
            # print(f'问题样本的概率密度：\n{norm_pds[error_index,:,:,:]}')
            example_left_censor = np.where(z_scores[error_index[0],:]==left_censor)[0]
            example_right_censor = np.where(z_scores[error_index[0], :] == right_censor)[0]
            print(f'第一个问题样本具体是：{z_scores[error_index[0],:]}')
            if example_left_censor.shape[0]>0:
                print(f'它的第{example_left_censor}个维度是左删失的')
                print(f"它确实在{left_censor_index[1][np.where(left_censor_index[0]==error_index[0])[0]]}处被删失了")
                print(f'替换它的截断概率应该为：{left_censor_probs[:,:,np.where(left_censor_index[0]==error_index[0])[0]]}')
            if example_right_censor.shape[0]>0:
                print(f'它的第{example_right_censor}个维度是右删失的')
                print(f"它确实在{right_censor_index[1][np.where(right_censor_index[0] == error_index[0])[0]]}处被删失了")
                print(f'替换它的截断概率应该为：{right_censor_probs[:, :, right_censor_index[1][np.where(right_censor_index[0] == error_index[0])[0]]]}')
            print(f'此时的均值参数为：{mu}')
            print(f'此时的方差参数为：{sigma_square}')
            print(f'它对应的概率密度具体是：{norm_pds[error_index[0],:,:,:]}')
            error_sample_broad = np.broadcast_to(np.expand_dims(np.expand_dims(z_scores[error_index[0],:], axis=0), axis=0), (2, num_pattern, num_permutation))
            print(f'如果没有删失，那么原始的概率密度应该是：{norm.pdf(error_sample_broad,mu,np.sqrt(sigma_square))}')
            print(f'截断概率分别为：{left_censor_probs}\n和\n{right_censor_probs}')
            pos_min = np.min(marginal_distribution_of_z_scores[marginal_distribution_of_z_scores>0])
            marginal_distribution_of_z_scores[error_index] = pos_min



        # except_omega的第i个元素代表了E[ω_i]
        expect_omega = (lambda_0*f_CCs)/marginal_distribution_of_z_scores

        expect_eta = 1-expect_omega
        # 把expect_eta变成一个N*C*K的张量
        expect_eta = np.expand_dims(np.expand_dims(expect_eta, axis=1), axis=1)
        expect_eta = np.broadcast_to(expect_eta, (num_sample, num_pattern, num_permutation)).copy()
        factors = norm_pds[:, 1, :, :]*[rho]

        # 强制使得那些异常的后验分布为均匀分布
        error_index = np.where(np.sum(factors, axis=1)==0)
        if error_index[0].shape[0] > 0:
            print(f'出问题的维度有：{set(error_index[1])}')
            print(f'第{error_index[0]}个样本的第{error_index[1]}个维度有问题')
            # # print(f'第一个问题样本在这个维度的概率密度为：{norm_pds[error_index[0][0],1,:,error_index[1][0]]}')
            if (error_index[0][0] in left_censor_index[0]) and error_index[1][0] in left_censor_index[1]:
                print(f'第{error_index[0][0]}个样本的第{error_index[1][0]}个维度发生了左删失')
                uncensor_prob = norm.pdf(z_scores[error_index[0][0], error_index[1][0]], mu[1,:,error_index[1][0]], np.sqrt(sigma_square[1,:,error_index[1][0]]))
                print(f'如果没有删失，那么这个样本的这个维度对应的概率密度应该是：{uncensor_prob}')
                print(f'发生删失后，实际的概率密度是：{norm_pds[error_index[0][0],1,:,error_index[1][0]]}')
                print(f'对应的截断概率本应该是：{left_censor_probs[1,:,error_index[1][0]]}')
                print(f'这个样本在这个维度下的值是：{z_scores[error_index[0][0], error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的期望为：{mu[1, :, error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的方差为：{sigma_square[1, :, error_index[1][0]]}')
            elif error_index[0][0] in right_censor_index[0] and error_index[1][0] in right_censor_index[1]:
                print(f'第{error_index[0][0]}个样本的第{error_index[1][0]}个维度发生了右删失')
                uncensor_prob = norm.pdf(z_scores[error_index[0][0], error_index[1][0]], mu[1, :, error_index[1][0]],np.sqrt(sigma_square[1, :, error_index[1][0]]))
                print(f'如果没有删失，那么这个样本的这个维度对应的概率密度应该是：{uncensor_prob}')
                print(f'发生删失后，实际的概率密度是：{norm_pds[error_index[0][0], 1, :, error_index[1][0]]}')
                print(f'对应的截断概率本应该是：{right_censor_probs[1, :, error_index[1][0]]}')
                print(f'这个样本在这个维度下的值是：{z_scores[error_index[0][0], error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的期望为：{mu[1, :, error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的方差为：{sigma_square[1, :, error_index[1][0]]}')
            else:
                print(f'这个样本的这个维度没有发生任何删失')
                print(f'这个样本发生过右删失：{error_index[0][0] in right_censor_index[0]}')
                print(f'这个样本发生右删失的维度也恰好是出问题的维度：{error_index[1][0] in right_censor_index[1][right_censor_index[0]==error_index[0][0]]}')
                print(f'对应的截断概率本应该是：{right_censor_probs[1, :, error_index[1][0]]}')
                error_prob = norm.pdf(z_scores[error_index[0][0], error_index[1][0]], mu[1,:,error_index[1][0]], np.sqrt(sigma_square[1,:,error_index[1][0]]))
                print(f'这个样本的这个维度对应的概率密度应该是：{error_prob}')
                print(f'这个样本的这个维度对应的概率密度实际上是：{norm_pds[error_index[0][0],1,:,error_index[1][0]]}')
                print(f'这个样本在这个维度下的值是：{z_scores[error_index[0][0], error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的期望为：{mu[1,:,error_index[1][0]]}')
                print(f'这个维度在CI模型中对应的方差为：{sigma_square[1,:,error_index[1][0]]}')
            factors[error_index[0], :, error_index[1]] = (1 / num_pattern) * np.ones(num_pattern)

        factors = factors/np.sum(factors, axis=1, keepdims=True)
        expect_eta = expect_eta*factors

        # expect_xi是一个N*C的矩阵
        expect_xi = lambda_0*np.prod(norm_pds[:,0,:,:], axis=2)
        expect_xi = (pi.reshape(1, -1)*expect_xi)/marginal_distribution_of_z_scores.reshape(-1, 1)


        # M-step
        ##########################################################################################################################
        # 一个N*2*C*K的张量，作用是：把被删失z_score替换为其在截断分布下的期望

        renew_expect = np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)).copy()
        # 在迭代中，替代原本的z_score的值（对应于左删失的情况）
        truncate_UB_unit = (np.broadcast_to(np.expand_dims(np.expand_dims(left_censor, axis=0), axis=0), (2, num_pattern, num_permutation))-mu)/np.sqrt(sigma_square)
        alternative_z_left = truncnorm.mean(-np.inf,
                                            truncate_UB_unit,
                                            mu,
                                            np.sqrt(sigma_square)
                                            )
        # 在迭代中，替代原本z_score的值（对应于右删失的情况）
        truncate_LB_unit = (np.broadcast_to(np.expand_dims(np.expand_dims(right_censor, axis=0), axis=0), (2, num_pattern, num_permutation)) - mu) / np.sqrt(sigma_square)
        # print('truncate_LB_unit: ', truncate_LB_unit)
        # print('mu: ', mu)
        # print('sigma_square: ', sigma_square)
        # try:
        alternative_z_right = truncnorm.mean(truncate_LB_unit,
                                        np.inf,
                                        mu,
                                        np.sqrt(sigma_square)
                                        )
        # except:
        #     alternative_z_right = truncated_normal_mean_mpmath(truncate_LB_unit, np.inf, mu, np.sqrt(sigma_square))
        # 如果发生删失，则开始替代
        if left_censor_index[0].shape[0] != 0:
            renew_expect[left_censor_index[0], :, :, left_censor_index[1]] = alternative_z_left[:, :, left_censor_index[1]].transpose(2,0,1)
        if right_censor_index[0].shape[0] != 0:
            renew_expect[right_censor_index[0], :, :, right_censor_index[1]] = alternative_z_right[:, :, right_censor_index[1]].transpose(2,0,1)


        lambda_0 = np.mean(expect_omega)

        rho = np.sum(expect_eta, axis=0)/[np.sum(np.sum(expect_eta, axis=1), axis=0)]

        pi = np.sum(expect_xi, axis=0)/np.sum(expect_xi)

        renew_mu = np.zeros(mu.shape)

        # 开始更新mu[0,:,:]
        elements = np.broadcast_to(np.expand_dims(expect_xi, axis=2), (num_sample, num_pattern, num_permutation))*renew_expect[:,0,:,:]
        norm_factor = np.sum(expect_xi, axis=0).reshape(-1, 1)
        renew_mu[0,:,:] = np.sum(elements, axis=0)/norm_factor

        # 开始更新mu[1,:,:]
        elements = expect_eta*renew_expect[:,1,:,:]
        renew_mu[1,:,:] = np.sum(elements, axis=0)/np.sum(expect_eta, axis=0)

        # print(f'更新期望以及混合系数共耗时：{end_time - start_time}')

        # 用旧截断方差+(旧期望-新mu)^2替代(z_score-新mu)^2

        # 首先，以z_score与期望的差的平方作为更新sigma时所需的参数
        renew_var = np.broadcast_to(np.expand_dims(np.expand_dims(z_scores, axis=1), axis=1), (num_sample, 2, num_pattern, num_permutation)).copy()
        renew_var = (renew_var - np.expand_dims(renew_mu, axis=0))**2
        # 然后，计算删失时的替代值（2*C*K的张量）
        alternative_z_square_left = truncnorm.var(-np.inf, truncate_UB_unit, mu, np.sqrt(sigma_square)) + (alternative_z_left - renew_mu) ** 2
        alternative_z_square_right = truncnorm.var(truncate_LB_unit, np.inf, mu, np.sqrt(sigma_square)) + (alternative_z_right - renew_mu) ** 2
        # 如果发生删失，则开始替代
        if left_censor_index[0].shape[0] != 0:
            renew_var[left_censor_index[0], :, :, left_censor_index[1]] = alternative_z_square_left[:, :, left_censor_index[1]].transpose(2,0,1)
        if right_censor_index[0].shape[0] != 0:
            renew_var[right_censor_index[0], :, :, right_censor_index[1]] = alternative_z_square_right[:, :, right_censor_index[1]].transpose(2,0,1)


        # 过去的参数mu已经完成了历史使命
        mu = renew_mu.copy()

        # 开始更新sigma_square[0,:,:]
        elements = np.broadcast_to(np.expand_dims(expect_xi, axis=2),
                                   (num_sample, num_pattern, num_permutation)) * renew_var[:, 0, :, :]
        norm_factor = np.sum(expect_xi, axis=0).reshape(-1, 1)
        sigma_square[0, :, :] = np.sum(elements, axis=0) / norm_factor

        # 开始更新sigma_square[1,:,:]
        elements = expect_eta * renew_var[:, 1, :, :]
        sigma_square[1, :, :] = np.sum(elements, axis=0) / np.sum(expect_eta, axis=0)
        

    return lambda_0, rho, pi, mu, sigma_square

# The calculation of majority probability
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

            Poisson_Binomial_distribution[:,k,i] = (joint_distribution[:, aim_mu_index, k]*Poisson_Binomial_distribution[:, k-1, i-1]
            + joint_distribution[:, inv_aim_mu_index, k]*Poisson_Binomial_distribution[:, k-1, i])/sum_joint[:, k]
    # 现在Poisson_Binomial_distribution的最后一行记录了完整的分布
    # 求超过半数的顺序属于j的概率
    P_CI = (1 - lambda_0) * np.sum(Poisson_Binomial_distribution[:, -1, (K+1) // 2:], axis=1)

    f_CCs = np.sum(np.prod(norm_pds[:, 0, :, :], axis=2) * [pi], axis=1)
    f_CIs = np.prod(np.sum(norm_pds[:, 1, :, :] * [rho], axis=1), axis=1)
    marginal_distribution_of_z_scores = lambda_0 * f_CCs + (1 - lambda_0) * f_CIs

    return (P_CC + P_CI) / marginal_distribution_of_z_scores

# The calculation of inconsistent probability
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

# 输入：全部数据data（格式为字典，key为subject名称，value为N×4(选项排序)×4(选项内容)的张量）; 记录训练和测试索引的路径（path1, 2）
# 需要统一读取训练集和测试集
# 每次随机选择 每道题的某个选项内容，读取该选项内容在4个位置的z-score
# 估计GMM的参数（未删失→删失），并计算测试集样本的概率
# 最终输出预测结果
# Processing the outputs of LLM to adapt them to GMM
def data_process(path1, path2, order_list):
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
            test_labels = test_df['label'].values

    train_data = np.array(train_data)
    train_data = train_data.transpose(1, 0, 2)
    test_data = np.array(test_data)
    test_data = test_data.transpose(1, 0, 2)

    return train_data, test_data, test_labels

# Coefficients estimation and calculation of posterior probability
def GMM_pipeline(train_data, test_data):

    GMM_calibrated_scores = np.zeros((test_data.shape[0], test_data.shape[2]))
    inconsistent_scores = np.zeros((test_data.shape[0], test_data.shape[2]))
    thresholds = np.zeros(test_data.shape[2])
    coefs_summary = []

    for aim in range(test_data.shape[2]):
        train_probs = train_data[:, :, aim]
        train_z_scores = norm.ppf(train_probs)
        test_z_scores = norm.ppf(test_data[:, :, aim])

        pos_mu = []
        neg_mu = []

        for k in range(train_z_scores.shape[1]):
            median = np.percentile(train_z_scores[:, k], 50, axis=0)
            pos_mu.append(np.mean(train_z_scores[train_z_scores[:, k] > median, k], axis=0))
            neg_mu.append(np.mean(train_z_scores[train_z_scores[:, k] <= median, k], axis=0))

        mu = np.array([[pos_mu, neg_mu],
                       [pos_mu, neg_mu]])

        pos_var = []
        neg_var = []

        for k in range(train_z_scores.shape[1]):
            median = np.percentile(train_z_scores[:, k], 50, axis=0)
            pos_var.append(np.var(train_z_scores[train_z_scores[:, k] > median, k], axis=0))
            neg_var.append(np.var(train_z_scores[train_z_scores[:, k] <= median, k], axis=0))

        sigma_square = np.array([[pos_var, neg_var],
                                 [pos_var, neg_var]])

        lambda_0 = 0.5
        # rho = np.array([[0.5], [0.5]]) * np.ones((2, 2))
        rho = np.ones((2, train_z_scores.shape[1]))
        rho[0, :] = 1/train_data.shape[2]
        rho[1, :] = 1-1/train_data.shape[2]
        pi = np.array([1/train_data.shape[2], 1-1/train_data.shape[2]])

        lambda_0, rho, pi, mu, sigma_square = EM(train_z_scores, lambda_0, rho, pi, mu, sigma_square, 50)

        # coefs = {'lambda': lambda_0, 'rho': rho, 'pi': pi, 'mu': mu, 'sigma': sigma_square}
        # print(f'在aim为{aim}时，未删失参数为{coefs}')

        # 对train_data进行删失
        left_censor_rate = 1
        right_censor_rate = 99
        # 获取每个维度的概率的1%和99%分位数（这意味着删失比例为2%）
        prob_left_censor = np.percentile(train_probs, left_censor_rate, axis=0)  # #
        prob_right_censor = np.percentile(train_probs, right_censor_rate, axis=0)
        # 获取删失索引                                                                                             #
        left_censor_index = np.where(train_probs <= prob_left_censor)  #
        right_censor_index = np.where(train_probs >= prob_right_censor)  #
        # 人工删失
        if left_censor_index[0].shape[0] > 0:
            train_probs[left_censor_index[0], left_censor_index[1]] = prob_left_censor[left_censor_index[1]]
        if right_censor_index[0].shape[0] > 0:
            train_probs[right_censor_index[0], right_censor_index[1]] = prob_right_censor[right_censor_index[1]]  #

        # 记录删失界限以供估计参数、计算后验概率时使用
        left_censor = norm.ppf(prob_left_censor)
        right_censor = norm.ppf(prob_right_censor)

        # 更新train_z_scores为删失状态
        train_z_scores = norm.ppf(train_probs)

        lambda_0, rho, pi, mu, sigma_square = EM_with_censor(train_z_scores, lambda_0, rho, pi, mu, sigma_square, 50, left_censor=left_censor, right_censor=right_censor)

        coefs = {'lambda': lambda_0, 'rho': rho.tolist(), 'pi': pi.tolist(), 'mu': mu.tolist(), 'sigma': sigma_square.tolist()}
        coefs_summary.append(coefs)
        # print(f'在aim为{aim}时，删失参数为{coefs}')

        # 获取删失索引                                                                                             #
        left_censor_index = np.where(test_z_scores <= left_censor)  #
        right_censor_index = np.where(test_z_scores >= right_censor)  #
        # 人工删失
        if left_censor_index[0].shape[0] > 0:
            test_z_scores[left_censor_index[0], left_censor_index[1]] = left_censor[left_censor_index[1]]
        if right_censor_index[0].shape[0] > 0:
            test_z_scores[right_censor_index[0], right_censor_index[1]] = right_censor[right_censor_index[1]]

        GMM_calibrated_scores[:, aim] = cal_prob_of_majority(lambda_0, pi, rho, mu, sigma_square, test_z_scores, left_censor, right_censor)
        inconsistent_scores[:, aim] = cal_prob_of_incons(lambda_0, pi, rho, mu, sigma_square, test_z_scores, left_censor, right_censor)
        thresholds[aim] = (1-lambda_0)*(1-np.sum(np.prod(rho, axis=1), axis=0))

    # # 返回：基于GMM的矫正概率、各样本的不一致性（取C种标签中的最高不一致性）、各样本具体的“最不一致”的标签、各标签对应的“不一致性”阈值
    return GMM_calibrated_scores, inconsistent_scores, thresholds, coefs_summary

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

# 创建解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str, default='chatglm2', help='指定哪个LLM的结果')
parser.add_argument('--dataset', type=str, default='SST-2', help='指定哪个数据集')
# 解析参数，即从命令行获取这些待定参数
args = parser.parse_args()

dict_num_labels = {'SST-2': 2, 'IMDB': 2, 'RTE': 2, 'Subj': 2, 'Emergency_DC': 2, 'Emergency_TL': 2, 'PubmedQA_binary': 2,
                   'MNLI': 3,
                   'MedNLI': 3,
                   'PubmedQA': 3,
                   'AG News': 4,
                   'BBC News': 5}
num_labels = dict_num_labels[args.dataset]

if num_labels != 5:
    all_orders = list(itertools.permutations(list(range(num_labels))))
else:
    all_orders = [[0, 1, 2, 3, 4],
                  [4, 0, 1, 2, 3],
                  [3, 4, 0, 1, 2],
                  [2, 3, 4, 0, 1],
                  [1, 2, 3, 4, 0]]


train_base_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/{args.model}/train_results_'
test_base_path = f'/Data/datasets/Option order results/{args.dataset}/LLM results/{args.model}/test_results_'

train_data, test_data, test_labels = data_process(train_base_path, test_base_path, all_orders)

GMM_calibrated_scores, inconsistent_scores, inconsistency_thresholds, coefs_summary = GMM_pipeline(train_data, test_data)


GMM_pre = np.argmax(GMM_calibrated_scores, axis=1)
GMM_acc = accuracy_score(test_labels, GMM_pre)

mean_calibrated_scores = np.mean(test_data, axis=1)
mean_pre = np.argmax(mean_calibrated_scores, axis=1)
mean_acc = accuracy_score(test_labels, mean_pre)

raw_pre = np.argmax(test_data, axis=2)
MV_pre, _ = mode(raw_pre, axis=1)
MV_acc = accuracy_score(test_labels, MV_pre)

train_data, test_data, test_labels = data_process_for_pride(train_base_path, test_base_path, all_orders)
pride_prior = np.mean(softmax(np.mean(np.log(train_data), axis=1)), axis=0).reshape(1, -1)
pride_pre = np.argmax(test_data[:, 0, :] / pride_prior, axis=1)
pride_acc = accuracy_score(test_labels, pride_pre)

raw_accs = []
for k in range(test_data.shape[1]):
    raw_accs.append(accuracy_score(test_labels, raw_pre[:, k]))

print(f'GMM的准确率为：{GMM_acc}')
print(f'均值的准确率为：{mean_acc}')
print(f'MV的准确率为：{MV_acc}')
print(f'PriDe的准确率为：{pride_acc}')
print(f'原始的准确率为：{raw_accs}')

results_summary = {'GMM_acc': GMM_acc, 'mean_acc': mean_acc, 'MV_acc': MV_acc, 'raw_accs': raw_accs,
                   'GMM_scores': GMM_calibrated_scores.tolist(),
                   'inconsistent_scores': inconsistent_scores.tolist(), 'thresholds': inconsistency_thresholds.tolist(),
                   'coefs': coefs_summary}

save_path = f'/Data/datasets/Option order results/{args.dataset}/calibrated results/{args.model}/results.json'
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f)













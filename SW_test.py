import numpy as np

import scipy
from numpy import linalg as LA

from scipy import special
import UtilityMetrics as UM



def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2  ##2b
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)  ##映射到0-1之间
    randoms = np.random.uniform(0, 1, len(samples))  ##randoms是一组随机概率值，模拟每一次扰动过程

    noisy_samples = np.zeros_like(samples)

    # report                                              ##将随机值通过每个v的累积分布函数映射回[-b~1+b]区间得到扰动后的报告值
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins  ## m=~d,扰动值的分桶数
    n = domain_bins  ## n=d,估计分布的分桶数
    m_cell = (1 + w) / m  ## 扰动桶宽
    n_cell = 1 / n  ## 估计桶宽

    transform = np.ones((m, n)) * q * m_cell  ##概率转移矩阵M
    for i in range(n):
        left_most_v = (i * n_cell)  ##当前真实桶的左值
        right_most_v = ((i + 1) * n_cell)  ##当前真实桶的右值

        ll_bound = int(left_most_v / m_cell)  ##扰动分布中，真实左值所在桶号
        lr_bound = int((left_most_v + w) / m_cell)  ##+2b
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)  ##+2b

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * (
                        (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
                        rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (
                            rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
                        (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))  ##扰动值频数直方图
    # return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    # return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta


# def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
#     theta = np.ones(n) / float(n)       ##初始估计分布，每个概率都是1/桶数
#     theta_old = np.zeros(n)
#     r = 0
#     sample_size = sum(ns_hist)
#     old_logliklihood = 0
#
#     while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
#         theta_old = np.copy(theta)
#         X_condition = np.matmul(transform, theta_old)
#
#         TMP = transform.T / X_condition
#
#         P = np.copy(np.matmul(TMP, ns_hist))
#         P = P * theta_old
#
#         theta = np.copy(P / sum(P))
#
#         logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
#         imporve = logliklihood - old_logliklihood
#
#         if r > 1 and abs(imporve) < loglikelihood_threshold:
#             # print("stop when", imporve, loglikelihood_threshold)
#             break
#
#         old_logliklihood = logliklihood
#
#         r += 1
#     return theta


def count(data, l, h, num):
    data_0_1 = (data - l) / (h - l)
    rs_hist, _ = np.histogram(data_0_1, bins=num, range=(0, 1))  ##真实值频数直方图
    # print(rs_hist)
    return rs_hist


if __name__ == "__main__":
    K = 256

    for file in ['beta_numerical.npy']:

        kl_errors = []

        emd_errors = []

        ks_errors = []

        mean_errors = []

        Quantiles_errors = []

        range1_errors = []

        range4_errors = []

        variance_errors = []

        for eps in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]:

            kl_error = []

            emd_error = []

            ks_error = []

            mean_error = []

            Quantiles_error = []

            range1_error = []

            range4_error = []

            variance_error = []

            for time in range(20):

                if time % 10 == 0:
                    print(file, eps, time)

                samples = np.load(file)

                # l = np.min(samples)
                l = 0
                # h = np.max(samples)
                h = 1

                theta = sw(samples, l, h, eps, K, K)

                real = count(samples, l, h, K) / len(samples)

                # KL
                KL = UM.KL(real, theta)
                kl_error.append(KL)

                # emd
                EMD = UM.EMD_value(theta, real, K)
                emd_error.append(EMD)

                # ks
                KS = UM.KS_value(theta, real, K)
                ks_error.append(KS)

                # mean
                MEAN = UM.Mean_value(theta, real, K)
                mean_error.append(MEAN)

                # Quantiles
                QUANTITLE = UM.Quantiles_value(theta, real, K)
                Quantiles_error.append(QUANTITLE)

                # Range_Query
                RANGEQUERY_1 = UM.Range_Query(theta, real, 0.1, K)
                RANGEQUERY_4 = UM.Range_Query(theta, real, 0.4, K)
                range1_error.append(RANGEQUERY_1)
                range4_error.append(RANGEQUERY_4)

                # Variance
                VARIANCE = UM.Variance_value(theta, real, K)
                variance_error.append(VARIANCE)

            kl_errors.append((np.mean(kl_error)))
            emd_errors.append(np.mean(emd_error))
            ks_errors.append(np.mean(ks_error))
            mean_errors.append(np.mean(mean_error))
            Quantiles_errors.append(np.mean(Quantiles_error))
            range1_errors.append(np.mean(range1_error))
            range4_errors.append(np.mean(range4_error))
            variance_errors.append(np.mean(variance_error))

            print("--------------------------------------")
            print("dataset and epsilon:", file, eps)
            print("SW+EMS estimate KL:", kl_errors[-1])
            print("SW+EMS estimate EMD:", emd_errors[-1])
            print("SW+EMS estimate KS:", ks_errors[-1])
            print("SW+EMS estimate MEAN:", mean_errors[-1])
            print("SW+EMS estimate QUANTITLE:", Quantiles_errors[-1])
            print("SW+EMS estimate RANGEQUERY_1:", range1_errors[-1])
            print("SW+EMS estimate RANGEQUERY_4:", range4_error[-1])
            print("SW+EMS estimate VARIANCE:", variance_errors[-1])
            print("--------------------------------------")

        print("--------------------------------------")
        print("dataset:", file)
        print("SW+EMS estimate KL:", kl_errors)
        print("SW+EMS estimate EMD:", emd_errors)
        print("SW+EMS estimate KS:", ks_errors)
        print("SW+EMS estimate MEAN:", mean_errors)
        print("SW+EMS estimate QUANTITLE:", Quantiles_errors)
        print("SW+EMS estimate RANGEQUERY_1:", range1_errors)
        print("SW+EMS estimate RANGEQUERY_4:", range4_errors)
        print("SW+EMS estimate VARIANCE:", variance_errors)
        print("--------------------------------------")

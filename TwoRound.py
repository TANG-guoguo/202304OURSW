import numpy as np
import UtilityMetrics as UM
import scipy
from random import shuffle
import SW_test as SW
import Aggregation as AG


def smoothing(theta, n):
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
    theta_smooth = np.matmul(smoothing_matrix, theta)
    return theta_smooth




def ROUND1(K, sample, eps):
    theta = SW.sw(sample, 0, 1, eps, K, K)  #K个桶面积
    bound = np.array([i/K for i in range(K+1)])#等距边界
    return theta, bound

def Find_Bucket(K, data, Bound):     #找data位于哪个桶
    result = 0
    # 每个桶 桶号k∈0~K-1
    right = K
    left = 0
    if data >= Bound[-1]:
        return K - 1
    while left <= right:     #二分查找
        middle = (left + right) // 2
        if Bound[middle] <= data < Bound[middle + 1]:
            result = middle
            break
        elif Bound[middle + 1] <= data:
            left = middle + 1
        else:
            right = middle - 1
    return result

def Fmap(theta1, sample2, K):
    num = len(sample2)
    width = 1/K
    Fx = np.zeros(num)
    EqualBound = [i / K for i in range(K + 1)]  #均匀横坐标边界
    for i in range(num):
        B_i = Find_Bucket(K, sample2[i], EqualBound)
        Fx[i] = sum(theta1[0:B_i]) + (sample2[i]-EqualBound[B_i])*(theta1[B_i]/width)
    return Fx

#计算quantiles
def Quantiles(f, K, alpha_list):
    Quantiles_list=[]
     # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 桶宽
    width = [EquaDisQuantile[i+1] - EquaDisQuantile[i] for i in range(K)]
    # 桶高
    h = [f[i]/width[i] for i in range(K)]
    #计算分位点
    CumulativeFrequency = np.zeros(K + 1)  # 每个分位点对应一个概率累计值
    for i in range(K):
        CumulativeFrequency[i + 1] = h[i] * width[i] + CumulativeFrequency[i]  # 第一个为0,计算后面k个端点的概率累加值
    for data in alpha_list:
        index = Find_Bucket(K, data, CumulativeFrequency)  # 先找到位于第几个区间内
        ReF = EquaDisQuantile[index] + (data - CumulativeFrequency[index]) / (CumulativeFrequency[index + 1] - CumulativeFrequency[index]) * width[index]
        Quantiles_list.append(ReF)
    return np.array(Quantiles_list)

def Rmap(theta, K, bound):
    R_bound = Quantiles(theta, K, bound)
    R_bound[-1] = 1.0  #保证域为0-1
    return R_bound

def ROUND2(theta1, K, sample2, eps):
    #将第二轮样本映射到第一轮的累积分布
    MappedSample = Fmap(theta1, sample2, K)
    #扰动
    theta2 = SW.sw(MappedSample, 0, 1, eps, K, K)  # K个桶面积
    #映射回去(实际上是下alpha分位点)
    EqualboundY = np.array([i / K for i in range(K + 1)])  # 等距边界Y
    bound = Rmap(theta1, K, EqualboundY)    #非等距边界
    #print("第一轮边界:", EqualboundY)
    #print("第二轮边界:", bound)
    return theta2, bound


def restore(Alltheta, Allbound, K):
    Alltheta = np.append(Alltheta, 0)
    F_hat = np.zeros(K)
    EqualBound = [i / K for i in range(K + 1)]  # 均匀横坐标边界
    for i in range(K):
        i1 = Allbound >= EqualBound[i]
        i2 = Allbound < EqualBound[i+1]
        index = i1 & i2
        F_hat[i] = np.sum(Alltheta[index])
    print(F_hat)
    return F_hat




def scheme1(K, sample1, sample2, eps):
    '''
    :param K: 分桶数
    :param sample1: 第一轮样本
    :param sample2: 第二轮样本
    :param eps: 隐私预算
    :return: 两轮汇总的估计分布F_hat
    '''
    #第一轮统计 直接调用sw 得到对真实分布的摸底theta1和第一轮边界
    theta1, bound1 = ROUND1(K, sample1, eps)
    #第二轮统计 映射后再sw扰动 得到第二轮估计和边界
    theta2, bound2 = ROUND2(theta1, K, sample2, eps)
    #聚合两轮分布结果
    F_hat = AG.aggregation(theta1, bound1, theta2, bound2)

    return F_hat





def count(data, l, h, num):
    data_0_1 = (data - l) / (h - l)
    rs_hist, _ = np.histogram(data_0_1, bins=num, range=(0, 1))  ##真实值频数直方图
    # print(rs_hist)
    return rs_hist


if __name__=='__main__':
    # income_numerical.npy, 524200, 2308374
    # Retirement_numerical.npy, 59690.74, 178012
    # taxi_pickup_time_numerical.npy, 86399, 2189968

    K=256
    # for file in ['income_numerical.npy', 'Retirement_numerical.npy', 'taxi_pickup_time_numerical.npy','beta_numerical.npy']:  #遍历数据集
    for file in ['beta_numerical.npy']:
        kl_errors = []

        emd_errors = []

        ks_errors = []

        mean_errors = []

        Quantiles_errors = []

        range1_errors = []

        range4_errors = []

        variance_errors = []

        for eps in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]:   #遍历eps
        #for eps in [0.5]:  # 遍历eps
            kl_error = []

            emd_error = []

            ks_error = []

            mean_error = []

            Quantiles_error = []

            range1_error = []

            range4_error = []

            variance_error = []

            for time in range(20):  #重复测试次数time

                if time % 10 == 0:
                    print(file, eps, time)

                samples = np.load(file)

                samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))     #值映射到0-1之间#############取最值不会泄露隐私吗？？
                # 随机分组
                shuffle(samples)
                sample1 = samples[:int(len(samples) * 0.3)]    #三十万
                sample2 = samples[int(len(samples) * 0.3):]    #七十万
                # 真实频率
                real = count(samples, 0, 1, K) / len(samples)
                # 估计频率
                F_hat = scheme1(K, sample1, sample2, eps)
                #print(F_hat)

                # KL
                KL = UM.KL(real, F_hat)
                # print("KL散度",KL)
                kl_error.append(KL)

                # emd
                EMD = UM.EMD_value(F_hat, real, K)
                emd_error.append(EMD)

                # ks
                KS = UM.KS_value(F_hat, real, K)
                ks_error.append(KS)

                # mean
                MEAN = UM.Mean_value(F_hat, real, K)
                mean_error.append(MEAN)

                # Quantiles
                QUANTITLE = UM.Quantiles_value(F_hat, real, K)
                Quantiles_error.append(QUANTITLE)

                # Range_Query
                RANGEQUERY_1 = UM.Range_Query(F_hat, real, 0.1, K)
                RANGEQUERY_4 = UM.Range_Query(F_hat, real, 0.4, K)
                range1_error.append(RANGEQUERY_1)
                range4_error.append(RANGEQUERY_4)

                # Variance
                VARIANCE = UM.Variance_value(F_hat, real,K)
                variance_error.append(VARIANCE)

            kl_errors.append(np.mean(kl_error))
            emd_errors.append(np.mean(emd_error))
            ks_errors.append(np.mean(ks_error))
            mean_errors.append(np.mean(mean_error))
            Quantiles_errors.append(np.mean(Quantiles_error))
            range1_errors.append(np.mean(range1_error))
            range4_errors.append(np.mean(range4_error))
            variance_errors.append(np.mean(variance_error))

            print("-------------------------------------------")
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate KL:", kl_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate EMD:", emd_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate KS:", ks_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate MEAN:", mean_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate RANGE_1:", range1_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate RANGE_4:", range4_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate QUANTILES:", Quantiles_errors[-1])
            print("dataset and epsilon:", file, eps)
            print("our_solution estimate VARIANCE:", variance_errors[-1])
            print("-------------------------------------------")

        print("-------------------------------------------")
        print("dataset:", file)
        print("our_solution estimate KL:", kl_errors)
        print("our_solution estimate EMD:", emd_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate KS:", ks_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate MEAN:", mean_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate RANGE_1:", range1_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate RANGE_4:", range4_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate QUANTILES:", Quantiles_errors)
        print("dataset and epsilon:", file, eps)
        print("our_solution estimate VARIANCE:", variance_errors)
        print("-------------------------------------------")

pass








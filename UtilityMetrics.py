
import scipy.stats
import numpy as np


def Find_Quantile(K, data, Quantile):
    result = 0
    # 每个桶 桶号k∈0~K-1
    right = K
    left = 0
    if data == Quantile[-1]:
        return K - 1
    while left <= right:
        middle = (left + right) // 2
        if Quantile[middle] <= data < Quantile[middle + 1]:
            result = middle
            break
        elif Quantile[middle + 1] <= data:
            left = middle + 1
        else:
            right = middle - 1
    return result


# 完整区间的累积概率
def getPComplete(h, width, index):
    result = 0
    for i in range(index):
        result += (h[i] * width[i])
    return result


#计算EMD距离
def EMD_value(f,f_hat,num):
    c1 = np.zeros(num)
    c2 = np.zeros(num)
    sum = 0
    for i in range(num):
        if i == 0:
            c1[i] = f[i]
            c2[i] = f_hat[i]
            sum += abs(c1[i]-c2[i])
        else:
            c1[i] = c1[i-1] + f[i]
            c2[i] = c2[i-1] + f_hat[i]
            sum += abs(c1[i]-c2[i])
    return sum/num     ##需要平均一下

#计算KS距离
def KS_value(f,f_hat,num):
    c1 = np.zeros(num)
    c2 = np.zeros(num)
    c =  np.zeros(num)
    sum = 0
    for i in range(num):
        if i == 0:
            c1[i] = f[i]
            c2[i] = f_hat[i]
            c[i] = abs(c1[i]-c2[i])
        else:
            c1[i] = c1[i-1] + f[i]
            c2[i] = c2[i-1] + f_hat[i]
            c[i] = abs(c1[i]-c2[i])
    sum = max(c)
    return sum

#计算均值
def Mean_value(f,f_hat,K):
    sum = 0
    u = 0.0
    u_hat = 0.0
    Middle_EquaDisQuantile = np.zeros(K)
    # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 计算每个桶的中点坐标
    for i in range(K):
        Middle_EquaDisQuantile[i] = (EquaDisQuantile[i+1] + EquaDisQuantile[i])/2
        u += Middle_EquaDisQuantile[i] * f[i]
        u_hat += Middle_EquaDisQuantile[i] * f_hat[i]
    sum = abs(u - u_hat)
    return sum

#计算quantiles
def Quantiles_value(f,f_hat,K):
    Quantiles_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    sum = 0.0
     # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 桶宽
    width = [EquaDisQuantile[i+1] - EquaDisQuantile[i] for i in range(K)]
    # 桶高
    h = [f[i]/width[i] for i in range(K)]
    h_hat = [f_hat[i]/width[i] for i in range(K)]
    #计算分位点
    CumulativeFrequency = np.zeros(K + 1)  # 每个分位点对应一个概率累计值
    CumulativeFrequency_hat = np.zeros(K + 1)
    for i in range(K):
        CumulativeFrequency[i + 1] = h[i] * width[i] + CumulativeFrequency[i]  # 第一个为0,计算后面k个端点的概率累加值
        CumulativeFrequency_hat[i + 1] = h_hat[i] * width[i] + CumulativeFrequency_hat[i]
    for data in Quantiles_list:
        index = Find_Quantile(K, data, CumulativeFrequency)  # 先找到位于第几个区间内
        index_hat = Find_Quantile(K, data, CumulativeFrequency_hat)
        ReF = EquaDisQuantile[index] + (data - CumulativeFrequency[index]) / (
                    CumulativeFrequency[index + 1] - CumulativeFrequency[index]) * width[index]
        ReF_hat = EquaDisQuantile[index_hat] + (data - CumulativeFrequency_hat[index_hat]) / (
                    CumulativeFrequency_hat[index_hat + 1] - CumulativeFrequency_hat[index_hat]) * width[index_hat]
        sum += abs(ReF- ReF_hat)/9
    return sum

#计算alpha范围查询
def Range_Query(f,f_hat,alpha,K):
    sum = 0.0
     # 分位点坐标
    EquaDisQuantile = [i/K for i in range(K+1)]
    # 桶宽
    width = [EquaDisQuantile[i+1] - EquaDisQuantile[i] for i in range(K)]
    # 桶高
    h = [f[i]/width[i] for i in range(K)]
    h_hat = [f_hat[i]/width[i] for i in range(K)]
    #随机采样i
    c1 = np.random.uniform(0,1-alpha)
    #计算范围查询上界
    c2 = c1 + alpha
    #计算范围查询函数
    Q_index1 = Find_Quantile(K, c1, EquaDisQuantile)
    Q_index2 = Find_Quantile(K, c2, EquaDisQuantile)
    r = getPComplete(h, width, Q_index2) + h[Q_index2] * (c2 - EquaDisQuantile[Q_index2])-getPComplete(h, width, Q_index1) - h[Q_index1] * (c1 - EquaDisQuantile[Q_index1])
    r_hat =  getPComplete(h_hat, width, Q_index2) + h_hat[Q_index2] * (c2 - EquaDisQuantile[Q_index2])-getPComplete(h_hat, width, Q_index1) - h_hat[Q_index1] * (c1 - EquaDisQuantile[Q_index1])
    sum = abs(r - r_hat)
    return sum


#计算方差
def Variance_value(f, f_hat, K):
    u = 0.0
    u_hat = 0.0

    Middle_EquaDisQuantile = np.zeros(K)
    # 分位点坐标
    EquaDisQuantile =np.array([i/K for i in range(K+1)])
    # 计算每个桶的中点坐标
    for i in range(K):
        Middle_EquaDisQuantile[i] = (EquaDisQuantile[i+1] + EquaDisQuantile[i])/2
        u += Middle_EquaDisQuantile[i] * f[i]
        u_hat += Middle_EquaDisQuantile[i] * f_hat[i]
    var1 = (np.square(Middle_EquaDisQuantile-u)*f).sum()
    var2 = (np.square(Middle_EquaDisQuantile - u_hat) * f_hat).sum()
    disvar=abs(var2-var1)
    return disvar


#计算KL散度
def KL( f, f_hat):
    return scipy.stats.entropy(f,f_hat)



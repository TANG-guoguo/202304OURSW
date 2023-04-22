from treelib import Tree
import numpy as np
import copy
import scipy


class Nodex(object):
    def __init__(self, interval, frequency, flag=True):
        self.interval = interval
        self.frequency = frequency
        self.flag = flag   #为F表示为低频区间，为T表示为高频区间，默认为T


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









def decompose_1(mean_theta1, theta1, bound1):  # 第一层分解
    judge = theta1 >= mean_theta1
    result = []
    frequency = []
    flag = judge[0]
    if flag == True:
        IFHIGH = True
    else:
        IFHIGH = False
    node_interval = [bound1[0]]
    node_leftindex = 0
    for i in range(len(bound1) - 1):  # 0-255
        if judge[i] != flag:  # 变化
            node_interval.append(bound1[i])
            node_rightindex = i
            result.append(node_interval)
            print("分割索引：",node_leftindex, node_rightindex)
            frequency.append(np.sum(theta1[node_leftindex: node_rightindex]))
            node_interval = [bound1[i]]
            node_leftindex = i
            flag = judge[i]  # 换flag
    else:
        node_interval.append(bound1[-1])
        node_rightindex = len(bound1)  # 256
        result.append(node_interval)
        frequency.append(np.sum(theta1[node_leftindex: node_rightindex]))
    # print(result)
    # print(frequency)
    assert len(result) == len(frequency)
    return np.array(result), np.array(frequency), IFHIGH


def Find_Bucket(K, data, Bound):  # 找data位于哪个桶
    result = 0
    # 每个桶 桶号k∈0~K-1
    right = K
    left = 0
    if data >= Bound[-1]:
        return K - 1
    while left <= right:  # 二分查找
        middle = (left + right) // 2
        if Bound[middle] <= data < Bound[middle + 1]:
            result = middle
            break
        elif Bound[middle + 1] <= data:
            left = middle + 1
        else:
            right = middle - 1
    return result


def Fmap(theta1, sample2, K):    #根据第一轮分布将sample映射到累积分布Fx
    num = len(sample2)
    width = 1/K
    Fx = np.zeros(num)
    EqualBound = [i / K for i in range(K + 1)]  #均匀横坐标边界
    for i in range(num):
        B_i = Find_Bucket(K, sample2[i], EqualBound)
        Fx[i] = sum(theta1[0:B_i]) + (sample2[i]-EqualBound[B_i])*(theta1[B_i]/width)
    return Fx


def get_frequency(cut, bound, theta):
    '''
    从输入区间和分布返回区间对应频率
    :param cut: 区间列表
    :param bound: 分布下标
    :param theta: 分布频率
    :return: 频率列表frequency
    '''
    K = len(bound) - 1
    frequency = []
    for i in range(len(cut)):
        left_index = Find_Bucket(K, cut[i][0], bound)
        right_index = Find_Bucket(K, cut[i][1], bound)
        f_sum = 0
        if left_index == right_index:  # 左右边界在一个桶里
            f_sum += (cut[i][1] - cut[i][0]) * (theta[left_index] / (bound[left_index + 1] - bound[left_index]))
            frequency.append(f_sum)
            continue
        if (left_index + 1) <= (right_index - 1):  # 左右边界距离>=2
            f_sum += np.sum(theta[(left_index + 1): (right_index - 1)])
        # 左边界频率
        f_sum += (bound[left_index + 1] - cut[i][0]) * (theta[left_index] / (bound[left_index + 1] - bound[left_index]))
        # print("左边界频率：", (bound[left_index + 1] - cut[i][0]) * (theta[left_index] / (bound[left_index + 1] - bound[left_index])))
        # 右边界频率
        f_sum += (cut[i][1] - bound[right_index]) * (theta[right_index] / (bound[right_index + 1] - bound[right_index]))
        # print("右边界频率：", (cut[i][1] - bound[right_index]) * (theta[right_index] / (bound[right_index + 1] - bound[right_index])))
        frequency.append(f_sum)

    assert len(frequency) == len(cut)
    #print("第二轮频率和=", sum(frequency))
    return np.array(frequency)



def weighted_averaging(f1,f2,cut,theta1):
    '''
    根据分割点原始长度、分割点映射到累积分布后的长度对两轮估计频率f1和f2进行加权平均
    :param f1: 对应第一轮频率
    :param f2: 对应第二轮频率
    :param cut: 分割点
    :param theta1: 第一轮分布
    :return: f：加权平均后的频率
    '''
    f=[]
    i=0
    for c in cut:
        lenth1 = c[1]-c[0]
        F_c = Fmap(theta1,c,len(theta1))
        lenth2 = F_c[1]-F_c[0]
        tmpf = f1[i]*(lenth2/(lenth1+lenth2))+f2[i]*(lenth1/(lenth1+lenth2))
        #print(tmpf)
        f.append(tmpf)
        i+=1
    return np.array(f)

# def norm_sub(f):    #非负且和为1
#     n = len(f)
#     f = np.array(f)
#     while(True):
#         index_nega = f<0
#         f[index_nega] = 0  #负值置零
#         f_sum = np.sum(f)  #总频率
#         x = f_sum - 1  #总差值
#         index_posi = f>0
#         positive_num = np.sum(index_posi)
#         y = x / positive_num  # 平均差值
#         f[index_posi] -= y
#         if(np.sum(f<0)==0):  #全正退出
#             break
#     print("norm_sub后频率",f)
#     print("频率之和",sum(f))
#     return f

def norm_sub(f,SUM=1):    #非负且和为1
    n = len(f)
    f = np.array(f)
    while(True):
        index_nega = f<0
        f[index_nega] = 0  #负值置零
        f_sum = np.sum(f)  #总频率
        x = f_sum - SUM  #总差值
        index_posi = f>0
        positive_num = np.sum(index_posi)
        y = x / positive_num  # 平均差值
        f[index_posi] -= y
        if(np.sum(f<0)==0):  #全正退出
            break
    print("norm_su后频率",f)
    print("频率之和",sum(f))
    return f


def consistency(theta, bound, frequency_1, cut_1):
    '''
    根据第1层节点加权平均后的频率frequency_1对theta的频率一致化
    :param theta:原始频率
    :param bound:theta的边界
    :param frequency_1:第1层节点加权平均后的频率（已非负且和为1）
    :param cut_1:第1层节点的分割点
    :return:一致化的theta和cut_1内部小区间<'i':[频率,间隔]>字典
    '''
    flag=-1
    f_dict={}
    for i in range(len(cut_1)):
        #获取区间内所有分割点
        leftbound = cut_1[i][0]
        rightbound = cut_1[i][1]
        i1 = bound > leftbound
        i2 = bound < rightbound
        bound_index = i1 & i2
        tmp_bound = bound[bound_index]
        tmp_bound = np.insert(tmp_bound, 0, leftbound)
        tmp_bound = np.append(tmp_bound, rightbound)
        #print(tmp_bound)
        # 分割点化为区间
        tmp_interval = [[tmp_bound[i], tmp_bound[i+1]] for i in range(len(tmp_bound)-1)]
        f_list = get_frequency(tmp_interval, bound, theta)
        #利用norm_sub一致化
        f_list_consistent = norm_sub(f_list, frequency_1[i])
        #恢复theta在bound上的频率
        if flag == -1:   #初始
            theta_consistent = f_list_consistent
            if tmp_bound[-1] in bound:
                flag = 0  ##当前末尾区间为完整区间
            else:
                flag = 1  ##当前末尾区间为不完整区间
        elif flag == 0:  #上步末尾区间为完整区间
            theta_consistent = np.append(theta_consistent, f_list_consistent)  #直接合并
            if tmp_bound[-1] in bound:
                flag = 0  ##当前末尾区间为完整区间
            else:
                flag = 1  ##当前末尾区间为不完整区间
        elif flag == 1:  #上步末尾区间为不完整区间
            theta_consistent[-1] += f_list_consistent[0]  #头尾相加后合并????????????????????????????????????????????????
            if len(f_list_consistent) > 1:#可能没有索引1
                theta_consistent = np.append(theta_consistent, f_list_consistent[1:])
            if tmp_bound[-1] in bound:
                flag = 0  ##当前末尾区间为完整区间
            else:
                flag = 1  ##当前末尾区间为不完整区间
        f_dict[str(i)] = [copy.deepcopy(f_list_consistent), copy.deepcopy(tmp_interval)]
    if len(theta_consistent) != len(theta):
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    assert len(theta_consistent) == len(theta)
    return theta_consistent, f_dict







def get_Fdict(theta1, bound1, theta2, bound2):   #返回<'分割点'：累积分布>字典
    Fdict={}
    for i in range(len(bound1)):
        if i==0:
            Fdict[str(bound1[i])] = 0
            continue
        Fdict[str(bound1[i])] = np.sum(theta1[0:i])

    for j in range(len(bound2)):
        if j==0 or j==len(bound2)-1:
            #print(np.sum(theta2[0:j]))
            continue
        Fdict[str(bound2[j])] = np.sum(theta2[0:j])
    #print(str(Fdict))
    return Fdict



def aggregation(theta1, bound1, theta2, bound2):  ##主函数main()
    # 建树并初始化第0层的根节点
    TREE = Tree()
    TREE.create_node(tag='L-0N-0', identifier='L-0N-0', data=Nodex(np.array([0, 1]), 1.0))  # 根节点
    # 第1层###########################################################################################################
    mean_theta1 = np.mean(theta1)  # 频率平均值
    # 对theta1按mean分割区间形成第1层区间和区间对应第一轮频率
    cut_1, frequency_1_1, flag_lowORhigh = decompose_1(mean_theta1, theta1, bound1)
    # 获取结点区间对应的第二轮频率
    frequency_1_2 = get_frequency(cut_1, bound2, theta2)
    # 加权平均两轮频率得到第1层结点频率
    frequency_1 = weighted_averaging(frequency_1_1,frequency_1_2,cut_1,theta1)
    #norm—sub
    frequency_1 = norm_sub(frequency_1)
    #建第一层树
    for i in range(len(cut_1)):
        temp_tag = 'L-1N-' + str(i)
        TREE.create_node(tag=temp_tag, identifier=temp_tag, data=Nodex(cut_1[i], frequency_1[i], flag_lowORhigh), parent='L-0N-0')
        flag_lowORhigh = not flag_lowORhigh  #flag翻转
    TREE.show()
    #################################################################################################################

    # 第2层###########################################################################################################
    #根据frequency_1对theta1和theta2进行norm_sub一致化处理
    theta1_consistent, theta1_dict = consistency(theta1, bound1, frequency_1, cut_1) #<'i':[频率,间隔]>字典
    theta2_consistent, theta2_dict = consistency(theta2, bound2, frequency_1, cut_1)
    #建第二层树
    node_num = 0
    for i in range(len(cut_1)):
        father_tag = 'L-1N-' + str(i)
        if TREE[father_tag].data.flag == False: #低频区间
            tmp_freq_2 = theta2_dict[str(i)][0]   #第二轮频率
            tmp_cut_2 = theta2_dict[str(i)][1]   #第二轮间隔
        else: #高频区间
            tmp_freq_2 = theta1_dict[str(i)][0]  # 第一轮频率
            tmp_cut_2 = theta1_dict[str(i)][1]  # 第一轮间隔
        for j in range(len(tmp_cut_2)):
            temp_tag = 'L-2N-' + str(node_num)
            TREE.create_node(tag=temp_tag, identifier=temp_tag, data=Nodex(tmp_cut_2[j], tmp_freq_2[j]), parent=father_tag)
            node_num += 1
    TREE.show()
    #################################################################################################################

    # 第3层##########################################################################################################
    AllBound = np.append(bound1, bound2[1:-1])  #合并两轮分割点
    AllBound.sort() #2d个分割点
    Fdict = get_Fdict(theta1_consistent, bound1, theta2_consistent, bound2)
    rawfreq = []
    for i in range(len(AllBound)-1):
        tmp_f = Fdict[str(AllBound[i+1])]-Fdict[str(AllBound[i])]
        rawfreq.append(tmp_f)
    print("rawfreq",rawfreq)
    #对rawfreq非负化处理
    finalfreq = norm_sub(rawfreq)
    print("finalfreq",finalfreq)
    return finalfreq, AllBound













    ####################################################################################################################


if __name__ == '__main__':
    theta1 = [0.00012, 0.00012, 0.00012, 0.00012, 0.00012, 0.00012, 0.00012, 0.00012, 0.00013, 0.00013, 0.00013,
              0.00013, 0.00013, 0.00013, 0.00013, 0.00014, 0.00014, 0.00014, 0.00014, 0.00015, 0.00015, 0.00015,
              0.00016, 0.00016, 0.00017, 0.00017, 0.00017, 0.00018, 0.00018, 0.00019, 0.00019, 0.00020, 0.00021,
              0.00021, 0.00022, 0.00023, 0.00023, 0.00024, 0.00025, 0.00026, 0.00027, 0.00028, 0.00029, 0.00030,
              0.00031, 0.00032, 0.00033, 0.00034, 0.00036, 0.00037, 0.00039, 0.00040, 0.00042, 0.00043, 0.00045,
              0.00047, 0.00048, 0.00050, 0.00052, 0.00054, 0.00056, 0.00058, 0.00060, 0.00062, 0.00064, 0.00067,
              0.00069, 0.00071, 0.00073, 0.00075, 0.00077, 0.00079, 0.00081, 0.00083, 0.00086, 0.00088, 0.00090,
              0.00092, 0.00095, 0.00097, 0.00099, 0.00102, 0.00104, 0.00107, 0.00110, 0.00113, 0.00116, 0.00119,
              0.00122, 0.00125, 0.00129, 0.00132, 0.00136, 0.00140, 0.00144, 0.00148, 0.00153, 0.00157, 0.00161,
              0.00166, 0.00171, 0.00176, 0.00181, 0.00187, 0.00192, 0.00198, 0.00204, 0.00210, 0.00217, 0.00223,
              0.00229, 0.00235, 0.00241, 0.00247, 0.00253, 0.00260, 0.00266, 0.00272, 0.00278, 0.00285, 0.00292,
              0.00299, 0.00307, 0.00315, 0.00323, 0.00331, 0.00340, 0.00348, 0.00357, 0.00366, 0.00376, 0.00385,
              0.00395, 0.00404, 0.00414, 0.00423, 0.00433, 0.00442, 0.00452, 0.00462, 0.00472, 0.00481, 0.00490,
              0.00500, 0.00509, 0.00518, 0.00528, 0.00538, 0.00548, 0.00558, 0.00568, 0.00579, 0.00590, 0.00601,
              0.00613, 0.00624, 0.00636, 0.00648, 0.00660, 0.00672, 0.00684, 0.00696, 0.00708, 0.00720, 0.00732,
              0.00744, 0.00755, 0.00767, 0.00778, 0.00789, 0.00800, 0.00810, 0.00821, 0.00831, 0.00841, 0.00850,
              0.00860, 0.00868, 0.00877, 0.00884, 0.00891, 0.00898, 0.00904, 0.00909, 0.00914, 0.00919, 0.00923,
              0.00926, 0.00929, 0.00930, 0.00931, 0.00930, 0.00929, 0.00927, 0.00924, 0.00920, 0.00916, 0.00911,
              0.00906, 0.00900, 0.00894, 0.00887, 0.00880, 0.00873, 0.00866, 0.00860, 0.00853, 0.00846, 0.00839,
              0.00833, 0.00827, 0.00821, 0.00814, 0.00808, 0.00801, 0.00795, 0.00787, 0.00780, 0.00772, 0.00764,
              0.00756, 0.00747, 0.00738, 0.00729, 0.00719, 0.00709, 0.00699, 0.00688, 0.00677, 0.00666, 0.00654,
              0.00643, 0.00631, 0.00619, 0.00607, 0.00595, 0.00583, 0.00571, 0.00558, 0.00546, 0.00534, 0.00522,
              0.00510, 0.00499, 0.00488, 0.00477, 0.00466, 0.00456, 0.00446, 0.00436, 0.00426, 0.00417, 0.00408,
              0.00400, 0.00391, 0.00383]
    bound1 = [i / 256 for i in range(256 + 1)]
    theta2 = [0.00393, 0.00393, 0.00394, 0.00394, 0.00395, 0.00395, 0.00396, 0.00396, 0.00396, 0.00397, 0.00397,
              0.00397, 0.00398, 0.00398, 0.00398, 0.00398, 0.00398, 0.00398, 0.00398, 0.00397, 0.00397, 0.00397,
              0.00397, 0.00397, 0.00397, 0.00396, 0.00396, 0.00395, 0.00395, 0.00394, 0.00393, 0.00393, 0.00392,
              0.00391, 0.00390, 0.00389, 0.00388, 0.00387, 0.00386, 0.00385, 0.00384, 0.00383, 0.00382, 0.00380,
              0.00379, 0.00378, 0.00377, 0.00376, 0.00375, 0.00374, 0.00373, 0.00372, 0.00371, 0.00371, 0.00370,
              0.00370, 0.00369, 0.00369, 0.00368, 0.00368, 0.00367, 0.00367, 0.00366, 0.00366, 0.00365, 0.00365,
              0.00365, 0.00365, 0.00365, 0.00364, 0.00365, 0.00365, 0.00365, 0.00366, 0.00366, 0.00367, 0.00368,
              0.00368, 0.00369, 0.00370, 0.00371, 0.00372, 0.00372, 0.00373, 0.00374, 0.00375, 0.00376, 0.00377,
              0.00378, 0.00379, 0.00380, 0.00381, 0.00382, 0.00383, 0.00384, 0.00385, 0.00385, 0.00386, 0.00387,
              0.00387, 0.00388, 0.00388, 0.00389, 0.00389, 0.00389, 0.00389, 0.00389, 0.00389, 0.00389, 0.00389,
              0.00390, 0.00390, 0.00391, 0.00392, 0.00393, 0.00395, 0.00397, 0.00398, 0.00400, 0.00402, 0.00404,
              0.00407, 0.00409, 0.00411, 0.00413, 0.00415, 0.00417, 0.00418, 0.00420, 0.00421, 0.00423, 0.00424,
              0.00424, 0.00425, 0.00426, 0.00426, 0.00426, 0.00426, 0.00426, 0.00426, 0.00425, 0.00424, 0.00424,
              0.00422, 0.00421, 0.00420, 0.00419, 0.00418, 0.00416, 0.00415, 0.00413, 0.00412, 0.00410, 0.00408,
              0.00407, 0.00405, 0.00403, 0.00402, 0.00400, 0.00399, 0.00397, 0.00396, 0.00395, 0.00394, 0.00393,
              0.00393, 0.00392, 0.00392, 0.00392, 0.00392, 0.00392, 0.00392, 0.00392, 0.00392, 0.00393, 0.00393,
              0.00393, 0.00393, 0.00392, 0.00392, 0.00391, 0.00390, 0.00389, 0.00388, 0.00387, 0.00387, 0.00386,
              0.00385, 0.00385, 0.00384, 0.00384, 0.00384, 0.00384, 0.00385, 0.00385, 0.00386, 0.00386, 0.00387,
              0.00388, 0.00389, 0.00390, 0.00391, 0.00392, 0.00393, 0.00394, 0.00395, 0.00396, 0.00396, 0.00397,
              0.00398, 0.00398, 0.00399, 0.00399, 0.00399, 0.00400, 0.00400, 0.00400, 0.00401, 0.00401, 0.00402,
              0.00402, 0.00403, 0.00403, 0.00403, 0.00404, 0.00404, 0.00404, 0.00404, 0.00403, 0.00403, 0.00402,
              0.00401, 0.00399, 0.00398, 0.00396, 0.00394, 0.00392, 0.00390, 0.00388, 0.00385, 0.00383, 0.00381,
              0.00379, 0.00376, 0.00374, 0.00372, 0.00369, 0.00367, 0.00364, 0.00361, 0.00359, 0.00356, 0.00353,
              0.00350, 0.00347, 0.00344]
    bound2 = [0.00000, 0.10065, 0.16954, 0.21571, 0.24916, 0.27507, 0.29613, 0.31373, 0.32878, 0.34185, 0.35338,
              0.36371, 0.37306, 0.38168, 0.38969, 0.39723, 0.40438, 0.41121, 0.41779, 0.42412, 0.43026, 0.43619,
              0.44195, 0.44753, 0.45295, 0.45820, 0.46330, 0.46826, 0.47308, 0.47777, 0.48235, 0.48682, 0.49119,
              0.49546, 0.49964, 0.50374, 0.50775, 0.51168, 0.51553, 0.51931, 0.52302, 0.52667, 0.53024, 0.53375,
              0.53720, 0.54059, 0.54393, 0.54721, 0.55042, 0.55358, 0.55669, 0.55975, 0.56276, 0.56571, 0.56862,
              0.57148, 0.57430, 0.57707, 0.57980, 0.58250, 0.58516, 0.58778, 0.59037, 0.59291, 0.59543, 0.59792,
              0.60037, 0.60280, 0.60520, 0.60757, 0.60992, 0.61223, 0.61452, 0.61679, 0.61903, 0.62126, 0.62346,
              0.62565, 0.62781, 0.62996, 0.63209, 0.63420, 0.63630, 0.63838, 0.64045, 0.64250, 0.64455, 0.64657,
              0.64859, 0.65060, 0.65259, 0.65457, 0.65655, 0.65850, 0.66046, 0.66239, 0.66432, 0.66624, 0.66815,
              0.67005, 0.67194, 0.67382, 0.67570, 0.67756, 0.67942, 0.68127, 0.68312, 0.68495, 0.68678, 0.68860,
              0.69042, 0.69223, 0.69404, 0.69584, 0.69763, 0.69942, 0.70120, 0.70298, 0.70475, 0.70652, 0.70829,
              0.71005, 0.71181, 0.71356, 0.71532, 0.71707, 0.71882, 0.72056, 0.72231, 0.72405, 0.72579, 0.72753,
              0.72926, 0.73099, 0.73272, 0.73445, 0.73618, 0.73791, 0.73963, 0.74135, 0.74306, 0.74478, 0.74649,
              0.74820, 0.74991, 0.75161, 0.75332, 0.75502, 0.75671, 0.75841, 0.76010, 0.76180, 0.76349, 0.76518,
              0.76686, 0.76855, 0.77023, 0.77191, 0.77360, 0.77527, 0.77695, 0.77863, 0.78030, 0.78198, 0.78365,
              0.78532, 0.78699, 0.78866, 0.79033, 0.79200, 0.79367, 0.79534, 0.79701, 0.79868, 0.80035, 0.80201,
              0.80368, 0.80536, 0.80703, 0.80870, 0.81038, 0.81205, 0.81373, 0.81541, 0.81710, 0.81878, 0.82047,
              0.82217, 0.82387, 0.82558, 0.82729, 0.82901, 0.83073, 0.83246, 0.83420, 0.83595, 0.83771, 0.83947,
              0.84125, 0.84303, 0.84483, 0.84663, 0.84845, 0.85027, 0.85211, 0.85397, 0.85583, 0.85771, 0.85960,
              0.86151, 0.86343, 0.86537, 0.86732, 0.86930, 0.87128, 0.87330, 0.87533, 0.87739, 0.87945, 0.88156,
              0.88367, 0.88582, 0.88799, 0.89019, 0.89242, 0.89467, 0.89696, 0.89928, 0.90164, 0.90403, 0.90645,
              0.90893, 0.91145, 0.91400, 0.91662, 0.91929, 0.92200, 0.92479, 0.92764, 0.93055, 0.93353, 0.93660,
              0.93976, 0.94300, 0.94634, 0.94978, 0.95334, 0.95703, 0.96087, 0.96486, 0.96904, 0.97342, 0.97804,
              0.98294, 0.98816, 0.99384, 1.00000]
    # decompose_1(0.0039062890625,np.array(theta1),bound1)
    aggregation(np.array(theta1), np.array(bound1), np.array(theta2), np.array(bound2))

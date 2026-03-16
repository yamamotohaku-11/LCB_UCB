import numpy as np
import scipy.stats as stats 
from config import k,n,success_probability,delta
import random

def pull(k,success_probability):
    pulls = [0 for i in range(k)]
    for i in range(k):
        pulls[i] = stats.bernoulli(success_probability[i])
    return pulls
        

def LCB(each_attempt,sum_result):#i回目におけるLCBを計算してLCBのリストを返す。入力はそれぞれのスロットのi回目までのリスト、それぞれのスロットの累計報酬
    return np.array(sum_result)/np.array(each_attempt) - np.sqrt((1/(2*np.array(each_attempt)))*(np.log(2*2*np.array(each_attempt)**2/(delta))))

def UCB(each_attempt,sum_result):
    return np.array(sum_result)/np.array(each_attempt) + np.sqrt((1/(2*np.array(each_attempt)))*(np.log(2*2*np.array(each_attempt)**2/(delta))))

def max_index(k,UCB):
    max_UCB = max(UCB)
    amount_max = 0
    max_index = []
    for i in range(k):
        if UCB[i] == max_UCB:
            amount_max +=1
            max_index.append(i)
    if amount_max > 1:
        return max_index[random.randint(0,len(max_index)-1)]
    else:
        return max_index[0]


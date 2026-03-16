import numpy as np


def make_list(n):
    x = np.array(n)*np.array(5)
    return x

def LCB(n_i,N_i):#i回目におけるLCBを計算してLCBのリストを返す。入力はそれぞれのスロットのi回目まで回した回数のリスト、それぞれのスロットの累計報酬
    return np.array(N_i)/np.array(n_i) - np.roots((1/(2*np.array(n_i)))*(np.log(2*2*np.array(n_i)**2/(0.05))))

print(LCB([5,6],[2,3]))
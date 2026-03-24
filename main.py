from config import k,success_probability
import scipy.stats as stats
import function  as f
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
def main():
    pulls = f.pull(k,success_probability)
    sum_result = [pulls[i].rvs(1)[0] for i in range(k)]
    each_attempt = [1 for i in range(k)]
    
    count = 1
    LCBi = [0 for i in range(k)]
    UCBi = [0 for i in range(k)]
    others =[50]


    fig, ax = plt.subplots()
    lines = []

    for i in range(k):
        line, = ax.plot([], [], marker="|")
        lines.append(line)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, k)

    plt.ion()
    fig.show()
    while  max(LCBi) <= max(others):
        print(count,UCBi,others,max(others) == max(UCBi))
        for i in range(k):
            result = pulls[i].rvs(1)[0]
            each_attempt[i] += 1
            sum_result[i] += result
            LCBi = f.LCB(each_attempt,sum_result)
            UCBi = f.UCB(each_attempt,sum_result)
            idx = np.argmax(LCBi)
            count += 1
            for j in range(k):
                lines[j].set_data([UCBi[j],LCBi[j]], [j, j]) 
            others = UCBi.copy()
            others[idx] = 0
            plt.pause(0.01)


            
    plt.ioff()
    print(f"best_arm: {np.argmax(UCBi)}",LCBi,UCBi,max(LCBi),max(others))
        

    fig,ax = plt.subplots()
    for i in range(k):
        ax.plot([LCBi[i],UCBi[i]],[i,i],marker="|")
        ax.plot(sum_result[i]/each_attempt[i],i,marker="o",color="black")
    ax.set_box_aspect(1)
    plt.savefig("LCB_UCB.png",dpi = 600)
    plt.show()
            

if __name__ == "__main__":
    main()

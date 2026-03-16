from config import k,n,success_probability
import scipy.stats as stats
import function  as f
import matplotlib.pyplot as plt
from tqdm import tqdm
def main():
    sum_result = [0 for i in range(k)]    
    each_attempt = [1 for i in range(k)]
    cumulative_result = 0
    pulls = f.pull(k,success_probability)

    for i in range(k):
        result = pulls[i].rvs(1)[0]
        sum_result[i] += result
        cumulative_result += result

    LCBi = f.LCB(each_attempt,sum_result)
    UCBi = f.UCB(each_attempt,sum_result)

    max_index = f.max_index(k,UCBi)
    for i in tqdm(range(k,n)):
        result = pulls[max_index].rvs(1)[0]
        sum_result[max_index] += result
        cumulative_result += result
        each_attempt[max_index] += 1
        max_index = f.max_index(k,UCBi)
        LCBi = f.LCB(each_attempt,sum_result)
        UCBi = f.UCB(each_attempt,sum_result)
        if max(LCBi) > min(UCBi):
            print("break")
            break

    print(i)
    print(f"BestArm: {f.max_index(k,UCBi)}")
    print(each_attempt)
    fig,ax = plt.subplots()
    ax.set_yticks([i for i in range(k)])
    for i in range(k):
        ax.plot(
            [LCBi[i], UCBi[i]],
            [i, i],
            marker="|",
            color="black"
        )

        ax.plot(
            sum_result[i]/each_attempt[i],
            i,
            marker="o",
            color="black"
        )    
    ax.set_ylim(-0.5, k-0.5)
    ax.set_xlim(min(LCBi)-0.05, 1.1)
    ax.set_xlabel("value")
    ax.set_ylabel("index")


    plt.savefig("graph.png")

if __name__ == "__main__":
    main()

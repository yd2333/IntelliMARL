import matplotlib as plt
import numpy as np

def save_visualization_to(agent,file):
    plt.figure(figsize=[8,8])
    plt.subplot(3,1,1)
    plt.plot(np.array(agent.loss).reshape([-1, 25]).mean(1))
    plt.ylabel("L1 Loss")
    #plt.ylim([-0,0.001])
    plt.subplot(3,1,2)
    plt.plot([agent.ws[i][4,6].item() for i in range(len(agent.rewards))])
    plt.ylabel("Sample weight")
    plt.subplot(3,1,3)
    plt.plot(agent.rewards)
    plt.ylabel("Rewards")
    plt.xlabel("# runs (in total or last train)")
    plt.savefig(file)
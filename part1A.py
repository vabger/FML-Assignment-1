from matplotlib import pyplot as plt
from tqdm import tqdm 
import numpy as np
import random
import os



num_coins = 100
def toss(num_trials):
    '''
    num_trials: number of trials to be performed.
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial
    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''  
    global num_coins
   # results = []
    
    ## Write your code here
    head_count = np.zeros(num_trials)
    ishead = np.zeros(num_coins)
    for j in tqdm(range(num_trials)):
        for i in range(num_coins):
            #if 1 then head
             head_count[j]+=np.random.randint(0,2)

    return head_count
    #return results
    

def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.
    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    fig, axs = plt.subplots(figsize =(10, 7), tight_layout=True)
    
    ## Write your code here
    # Plot the histogram
    axs.hist(trial, bins=range(num_coins + 2), edgecolor='black', alpha=0.7)
    axs.set_title('Histogram of Number of Heads')
    axs.set_xlabel('coins')
    axs.set_ylabel('Number of Heads')

    filename = f"hist_{len(trial)}.png"
    fig.savefig( os.path.join('histograms', filename),dpi=300,bbox_inches="tight")


if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000,100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)

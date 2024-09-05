import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

from closedForm import LinearRegressionClosedForm
from closedForm import plot_learned_equation

def transform_input(x):
    '''
    This function transforms the input to generate new features.

    Args:
      x: 2D numpy array of input values. Dimensions (n' x 1)

    Returns:
      2D numpy array of transformed input. Dimensions (n' x K+1)
      
    '''
    # Write your code here
    columns = [x**0,x**2,np.cos(x)]
    X_transformed = np.column_stack(columns)
    return X_transformed
    raise NotImplementedError()
    
def read_dataset(filepath):
    '''
    This function reads the dataset and creates train and test splits.
    
    n = 500
    n' = 0.9*n

    Args:
      filename: string containing the path of the csv file

    Returns:
      X_train: 2D numpy array of input values for training. Dimensions (n' x 1)
      y_train: 2D numpy array of target values for training. Dimensions (n' x 1)
      
      X_test: 2D numpy array of input values for testing. Dimensions ((n-n') x 1)
      y_test: 2D numpy array of target values for testing. Dimensions ((n-n') x 1)
      
    '''

    df = pd.read_csv(filepath)
    x = df['x'].values
    y = df['y'].values

    n = len(x)

    split_index = int(0.9*n)

    X_train = x[:split_index]
    X_test = x[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train,y_train,X_test,y_test
    raise NotImplementedError()


############################################
#####        Helper functions          #####
############################################

def plot_dataset(X, y):
    '''
    This function generates the plot to visualize the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 1)
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      None
    '''
    plt.title('Plot of the unknown dataset')
    plt.scatter(X, y, color='r')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.savefig('dataset.png')

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Starting experiment #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_test, y_test = read_dataset('dataset.csv')
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET +  "Plotting dataset: ",end="")
    try:
        plot_dataset(X_train, y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Performing input transformation: ", end="")
    try:
        X_train = transform_input(X_train)
        # X = X_test
        X_test = transform_input(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
        
    print(RESET + "Caclulating weights: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Checking closeness: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        # plt.scatter(X,y_hat,color='green')
        if np.allclose(y_hat, y_test, atol=1e-02):
          print(GREEN + "done")
        else:
          print(RED + "failed")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
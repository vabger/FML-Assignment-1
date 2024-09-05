import pandas as pd
import numpy as np
from closedForm import LinearRegressionClosedForm
from batchGradientDescent import LinearRegressionBatchGD

def polynomial_features(X, degree):
    X_poly = X.copy()
    
    for d in range(2, degree + 1):
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                X_poly = np.hstack((X_poly, (X[:, i] * X[:, j]).reshape(-1, 1)))
                print(X_poly)
    
    return X_poly


def read_dataset(filepath_train,filepath_test):

    df_train = pd.read_csv(filepath_train)
    df_test = pd.read_csv(filepath_test)

    X_train = df_train.iloc[:,1:65].values
    y_train = df_train['score'].values
    id_train = df_train['ID'].values

    X_test = df_test.iloc[:,1:65].values
    id_test = df_test.iloc[:,0].values

    return X_train,y_train,X_test,id_test,id_train

def generate_kaggle_csv(score_pred,index):

    data = {
        "ID":index.flatten(),
        "score":score_pred.flatten()
    }

    df = pd.DataFrame(data)
    df.to_csv("kaggle.csv",index=False)

def select_features(X_train,y_train,X_test,threshold=0.1):
    data_with_target = np.column_stack((X_train, y_train))

    # Compute correlation matrix
    correlation_matrix = np.corrcoef(data_with_target, rowvar=False)

    # Extract correlations of features with the target variable
    target_corr = correlation_matrix[-1, :-1]  # Exclude the last row and column
    
    # Get indices of features with high correlation
    high_corr_indices = np.where(np.abs(target_corr) > threshold)[0]

    # Select high-correlation features from the original data
    X_train = X_train[:, high_corr_indices]
    X_test = X_test[:,high_corr_indices]

    return X_train,X_test


def main():
    X_train,y_train,X_test,id_test,id_train = read_dataset("train.csv","test.csv")
    
    lr = LinearRegressionClosedForm()


    # X_train,X_test = select_features(X_train,y_train,X_test,0.001)

    X_train = polynomial_features(X_train,2)
    X_test = polynomial_features(X_test,2)


    X_train = np.column_stack((np.ones(X_train.shape[0]),X_train))
    X_test = np.column_stack((np.ones(X_test.shape[0]),X_test))

    lr.fit(X_train,y_train)
    score_pred = np.round(lr.predict(X_train))
    generate_kaggle_csv(score_pred,id_train)


if __name__ == "__main__":
    main()
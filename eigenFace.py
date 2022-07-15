import pandas as pd
import numpy as np

def eigenFace():
    data = pd.read_csv('face_data.csv')
    # 80% training, 20% validation
    train = data.sample(frac=0.8,random_state=0,axis=0)
    validation = data[~data.index.isin(train.index)]

    # Calculate covariance
    covariance = np.cov(train)
    # Calculate eigenvalues and feature vector of covariance
    eigenvalue, featurevector = np.linalg.eig(covariance)
    # According to the matrix indices, sort the eigenvalues in decreasing order
    sorted_Index = np.argsort(eigenvalue)
    # remain the top K eigenvectors corresponding to eigenvalues
    topk_evecs = featurevector[:,sorted_Index[:-150-1:-1]]
    # Acquire the eigen face space of train samples
    eigenface = np.dot(np.transpose(train),topk_evecs)
    # Fit the project of train samples on the eigenface space
    eigen_train = np.dot(train,eigenface)

    # Calculate the project of validation samples on the eigenface space
    eigen_test = np.dot(validation,eigenface)
    # print(eigen_test[0])

    right = 0 # Marking the right counts
    for k in range(1,eigen_test.shape[0]):
        # Calculate the Euclidean distance to match the right face
        # Assign minDistance the value of the Euclidean distance between eigen_train[0] and eigen_test
        minDistance = np.linalg.norm(eigen_train[0] - eigen_test[k])
        num = 1
        for i in range(1,eigen_train.shape[0]):
            distance = np.linalg.norm(eigen_train[i] - eigen_test[k])
            if(minDistance > distance):
                minDistance = distance
                num = train.values[i][4096]
        if num == validation.values[k][4096]:
            right+=1

    print('Correct number:', right)
    print('Total validation number:', eigen_test.shape[0])
    print('Validation Precision:',right/eigen_test.shape[0] * 100, '%')

eigenFace()


#Name: Anooshka Bajaj


import pandas as pd
df=pd.read_csv(r'E:\seismic_bumps1.csv',usecols=['seismic','seismoacoustic','shift','genergy','gpuls', 'gdenergy','gdpuls','ghazard','energy','maxenergy','class'])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np


#1
#splitting the data into training and test data
[X_train, X_test, X_label_train, X_label_test] =train_test_split(df[df.columns[:-1]], df['class'], test_size=0.3, random_state=42,shuffle=True)

training_data = pd.concat((X_train,X_label_train),axis=1)
training_data.to_csv('seismic_bumps_train.csv',index = False)

test_data = pd.concat((X_test,X_label_test),axis=1)
test_data.to_csv('seismic_bumps_test.csv',index = False)


best_result = {}                           #dictionary to store best_result for each method

#function for K nearest neighbours classification
def KNN(X_train,X_label_train,X_test):
    accuracy = {}
    
    for k in [1,3,5]:
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,X_label_train)       
        y_pred = knn.predict(X_test)
        s = metrics.accuracy_score(X_label_test,y_pred)
        accuracy[k] = s
        print('\nk = ',k)
        print('confusion matrix :\n',metrics.confusion_matrix(X_label_test,y_pred))
        print('classification accuracy : ',s)
        
    print('\nAccuracy is high for k = ',max(accuracy,key = accuracy.get))
    return [accuracy[max(accuracy)]]
    
best_result['KNN'] = KNN(X_train,X_label_train,X_test)


#2
#normalising the training and test data
X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())


normalised_training_data = pd.concat((X_train,X_label_train),axis=1)
normalised_training_data.to_csv('seismic_bumps_train_normalized.csv',index = False)

normalised_test_data = pd.concat((X_test,X_label_test),axis=1)
normalised_test_data.to_csv('seismic_bumps_test_normalized.csv',index = False)

best_result['KNN after normalisation of data'] = KNN(X_train,X_label_train,X_test)


#3
df_train=pd.read_csv('seismic_bumps_train.csv')
df_test=pd.read_csv('seismic_bumps_test.csv')

#dividing the training data with respect to the classes
C0=df_train[df_train['class']==0][df_train.columns[0:-1]]
C1=df_train[df_train['class']==1][df_train.columns[0:-1]]
X_test = df_test[df_test.columns[0:-1]]
Y_test = df_test[df_test.columns[-1]]
Y_Predicted=[]

#mean and covariance for the training data with class 0
Mean_C0=C0.mean().values
Cov_C0=C0.cov().values

#mean and covariance for the training data with class 1
Mean_C1=C1.mean().values
Cov_C1=C1.cov().values

#Prior Probability of class 0
P_C0=len(C0)/(len(C0)+len(C1))

#Prior Probability of class 1
P_C1=len(C1)/(len(C0)+len(C1))

d=len(X_test.columns)                                        #no of dimensions

#doing bayes classification for each test vector
for x in X_test.values:
    #likelihood of class 0
    p_x_C0=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C0)**0.5)*np.exp(-0.5*np.linalg.multi_dot([(x-Mean_C0).T,np.linalg.inv(Cov_C0),(x-Mean_C0)]))
    #likelihood of class 1
    p_x_C1=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C1)**0.5)*np.exp(-0.5*np.linalg.multi_dot([(x-Mean_C1).T,np.linalg.inv(Cov_C1),(x-Mean_C1)]))
    #Evidence
    P_x=p_x_C0*P_C0+p_x_C1*P_C1
    #Posterior Probability of class 0
    P_C0_x=p_x_C0*P_C0/P_x
    #Posterior Probability of class 1
    P_C1_x=p_x_C1*P_C1/P_x
    #Assigning class to the test vector
    if (P_C0_x>P_C1_x):
        Y_Predicted.append(0)
    else:
        Y_Predicted.append(1)
        
print('\nConfusion Matrix: ')
print(metrics.confusion_matrix(Y_test,Y_Predicted))
print('Accuracy score:',metrics.accuracy_score(Y_test,Y_Predicted))
best_result['Bayes classifier'] = [metrics.accuracy_score(Y_test,Y_Predicted)]


#4
#printing best_result dictionary as a dataframe
print(pd.DataFrame(best_result,index = ['Accuracy']).transpose())




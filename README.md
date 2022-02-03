# Data-Classification
Data Classification using K-Nearest Neighbor Classifier, Bayes Classifier with Unimodal Gaussian Density and Bayes Classifier with Gaussian Mixture Model (GMM).

I am given the Seismic-Bumps Data Set as a csv file (seismic-bumps.csv). The data describe the problem of high energy (higher than 104 J) seismic bumps forecasting in a coal mine. This data is collected from two of longwalls located in a Polish coal mine. Mining activity was and is always connected with the occurrence of dangers which are commonly called mining hazards. A special case of such threat is a seismic hazard which frequently occurs in many underground mines. Seismic hazard is the hardest detectable and predictable of natural hazards and in this respect it is comparable to an earthquake. More and more advanced seismic and seismoacoustic monitoring systems allow a better understanding rock mass processes and definition of seismic hazard prediction methods. Accuracy of so far created methods is however far from perfect. Complexity of seismic processes and big disproportion between the number of low-energy seismic events and the number of high-energy phenomena causes the statistical techniques to be insufficient 
to predict seismic hazard.\

This dataset contains recorded features from the seismic activity in the rock mass and seismoacoustic activity with the possibility of rockburst occurrence to predict the hazardous and non-hazardous state. It consists 2584 tuples each having 19 attributes. The last attribute for every tuple signifies the class label (0 for hazardous state and 1 for non-hazardous state). It is a two class problem. Other attributes are input features.

I have written a python program to split the data of each class from seismic-bumps.csv into train data and test data and classified every test tuple using K-nearest neighbor (KNN) method for the different values of K=1, 3, and 5. 

1. I performed the following analysis: 
a. Confusion matrix for each K.
b. Classification accuracy for each K. 

2. Normalised all the attributes of seismic-bumps-train.csv using Min-Max normalization to transform the data in the range [0-1] and Normalized the test dataset using the minimum and 
maximum values of train dataset and save the test data as seismic-bumps-testnormalised.csv.
Classifed every normalized test tuple using K-nearest neighbor (KNN) method for the different values of 
K=1, 3, and 5. Perform the following analysis:
a. Find confusion matrix for each K.
b. Find the classification accuracy for each K. Note the value of K for which the accuracy is 
high.

3. Build a Bayes classifier (with unimodal Gaussian density used to model the distribution of 
the data) on the training data seismic-bumps-train.csv. Test the performance on 
seismic-bumps-test.csv and give confusion matrix and accuracy.
Note: Compute mean vector and covariance matrix from the training data of each classes 
separately. Use them to compute likelihood for a class. For computing likelihood use the 
expression of multivariate Gaussian density. (Do not use Gaussian Naïve Bayes function
from sklearn). 

4. Compared the best result of KNN classifier, best result of KNN classifier on normalised data, and result of Bayes classifier using unimodal Gaussian density. 


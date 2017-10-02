#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:11:37 2017

@author: fmercermoss
"""
import genderize
import dataPreparation
import time
import machineLearning
import matplotlib.pyplot as plt


# BBC Data Scientist technical task: Predict gender from Face images.  
######################################################################

# This program explores some of the ways machine learning tools can be used to differentiate between male and female face images.
# All face images used were downloaded from the LFW dataset (http://vis-www.cs.umass.edu/lfw/). The version of the dataset used here
# contained images that were preproceesed using "deep funneling" to align the eyes.
#



# Task Part 1: Preparing the data

# As the gender of each face is not included in the dataset, the first stage was to acquire this information.
# The easiest way to do this is to use genderize.io API to make a prediction of the gender based upon the associated 
# name of the face. A "genderize.py" class that does just this was downloaded and modified (https://github.com/Pletron/LFWgender).
# This procedure resulted in a 'males' folder containing 8988 images, and a 'females' folder containing 2759 images.

# genderize()  # Commented out as only needs to be called once. (Actually 3 times due to the request limit being reached, but hey-ho).
 

# Define where to find the images.
malesFolder = "/Users/mercef02/Dropbox (BBC)/Code/BBC_test/faces_project/male"
femalesFolder = "/Users/mercef02/Dropbox (BBC)/Code/BBC_test/faces_project/female"

# Define some options
test_size = 0.2

# Ensure there are equal numbers of examples from each class.
imsPerMaleClass = 1200
imsPerFemaleClass = 1200

# Choose the resoution of the images.
res = (24,24)

# Create data object to read in the data (see class comments for details).
data = dataPreparation.calculateImageVectorsFromDirectories(femalesFolder, malesFolder, imsPerFemaleClass, imsPerMaleClass, res, test_size)



#######################################################

# Task Part 2: Machine Learning

########################################################

# The input to machine learning algorithms come in many different forms. Images as inputs are typically very high-dimensional which is 
# computationally intensive (although modern GPUs engineered for CNNs are making this less of an issue). Therefore, instead of 
# inputting the raw images, "features engineering" is a process to reduce the dimensionality the data (and complexity of the model) while capturing
# the important structure. 

# In addition to using the raw pixels, we will try two  different approaches to engineer features for the current project: 
    
#      1. The first is an unsupervised approach called principal conponents analysis. This procedure analyses the covariance matrix of the features 
#         and is then able to calculate a subspace of orthogonal and uncorrelated dimensions that maximise variance in the data. There is an implicit
#         assumption that the data being modelled is fits a multivariate Gaussian distribution as it operates on the second moment (variance) and no higher. 
#         PCA is unsupervised as it does not use any class information. 
#
#      2. The next dimensionality reduction technique we shall try is a supervised variant of PCA which, given the class labels, identifies a linear 
#         subspace that maximises variances between the two classes. Unlike PCA, however, LDA makes a more assumptions about the structure of the data 
#         it is being used to model. Assumptions are: normality of the input variables; homogeneity of variance between classes (the variance in the feature
#         distributiuons are equal in the two classes); multicolinearity (that the input variables are not correlated); and independence of the obervations.
#         From this list the only one that is clearly being violated is multicolinearity as image pixels are highly correlated (there exists a particularly strong                                          ) 
#         correlation between two pixel value and their proximity. To reduce the collinearity we could first project to PCA space but from my experience it is fairly
#         robust so we won't do this here.  



#  So first we will try a linear SVM with stepwise feature selection
t1 = time.time()
PCA_L_acc = machineLearning.generateModel(data,'pca','SVM_LIN')
timePCA_L = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of linear SVM in PCA subspace:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(PCA_L_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(PCA_L_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(PCA_L_acc.accuracyTotal  *100))


# Next we shall try a non-linear kernel with no stepwise feature selection 
t1 = time.time() 
PCA_NL_acc = machineLearning.generateModel(data,'pca','SVM_NL')
timePCA_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of non-linear SVM in PCA subspace:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(PCA_NL_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(PCA_NL_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(PCA_NL_acc.accuracyTotal  *100))



# Linear SVM in linear discriminant analysis space 
t1 = time.time() 
LDA_L_acc = machineLearning.generateModel(data,'lda','SVM_LIN')
timeLDA_L = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of linear SVM in LDA subspace:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(LDA_L_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(LDA_L_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(LDA_L_acc.accuracyTotal  *100))


#Non-Linear SVM in linear discriminant analysis space 
t1 = time.time() 
LDA_NL_acc = machineLearning.generateModel(data,'lda','SVM_NL')
timeLDA_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of non-linear SVM in LDA subspace:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(LDA_NL_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(LDA_NL_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(LDA_NL_acc.accuracyTotal  *100))


#Linear SVM using raw pixel vectors 
t1 = time.time() 
RAW_L_acc = machineLearning.generateModel(data,'raw','SVM_LIN')
timeRAW_L = time.time()-t1
print('   ')
print('------------------------------------------')
print('------------------------------------------')
print("Performance of linear SVM on raw pixels:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(RAW_L_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(RAW_L_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(RAW_L_acc.accuracyTotal  *100))

#Non linear SVM using raw pixel vectors
t1 = time.time() 
RAW_NL_acc = machineLearning.generateModel(data,'raw','SVM_NL')
timeRAW_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print('------------------------------------------')
print("Performance of non-linear SVM on raw pixels:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(RAW_NL_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(RAW_NL_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(RAW_NL_acc.accuracyTotal  *100))


#Keras CNN on raw pixels
#t1 = time.time() 
#CNN_acc = machineLearning.generateModel(data,'raw','DEEP')
#timeCNN = time.time()-t1
#print('   ')
#print('------------------------------------------')
#print('------------------------------------------')
#print("Performance of CNN on raw pixels:")
#print('------------------------------------------')
#print("Male Accuracy: %.2f"%(CNN_acc.accuracyMales*100))
#print("Female Accuracy: %.2f"%(CNN_acc.accuracyFemales*100))
#print("Total Accuracy: %.2f"%(CNN_acc.accuracyTotal  *100))


results = [PCA_L_acc.accuracyTotal,PCA_NL_acc.accuracyTotal,LDA_L_acc.accuracyTotal,LDA_L_acc.accuracyTotal, RAW_L_acc.accuracyTotal, RAW_NL_acc.accuracyTotal]
labels = ["PCA-L", "PCA-NL", "LDA-L", "LDA-NL", "RAW-L","RAW-NL" ]
timeResults = [timePCA_L,timePCA_NL,timeLDA_L,timeLDA_NL,timeRAW_L,timeRAW_NL]
f = plt.figure()
ax = f.add_subplot(121)
ax.set_title("Performance of models on uncropped images.")

ax.bar(range(6),results,tick_label=labels)
ax.set_ylabel("Percent correct") 
ax2 = f.add_subplot(122)
ax2.set_title("Time taken of models on uncropped images.")
ax2.bar(range(6),timeResults,tick_label=labels)
ax2.set_ylabel("Seconds") 
#f.show()



# OK so the results so far have been OK but not outstanding with any of the classifier/feature combinations.
# One problem might be that there is too much unuseful data un the images surrounding the face. By cropping 
# each image around the face, we would be dropping  a significant amount of that unuseful data and allow the machine-learning
# algorithms to concentrate more upon the useful (i.e. face) data.


#Each image starts off as 250x250, so here we shave off 80 pixels from each  leaving just a 90x90 image left before it is resized.
cropParams = (80, 80, 170, 170)
data.calculateImageVectorsCropped(cropParams)


t1 = time.time()
PCA_L_accCROP = machineLearning.generateModel(data,'pca','SVM_LIN')
timePCA_L = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of linear SVM in PCA subspace (CROPPED):")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(PCA_L_accCROP.accuracyMales*100))
print("Female Accuracy: %.2f"%(PCA_L_accCROP.accuracyFemales*100))
print("Total Accuracy: %.2f"%(PCA_L_accCROP.accuracyTotal  *100))


# Next we shall try a non-linear kernel with no stepwise feature selection 
t1 = time.time() 
PCA_NL_accCROP = machineLearning.generateModel(data,'pca','SVM_NL')
timePCA_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of non-linear SVM in PCA subspace (CROPPED):")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(PCA_NL_accCROP.accuracyMales*100))
print("Female Accuracy: %.2f"%(PCA_NL_accCROP.accuracyFemales*100))
print("Total Accuracy: %.2f"%(PCA_NL_accCROP.accuracyTotal  *100))



# Linear SVM in linear discriminant analysis space 
t1 = time.time() 
LDA_L_accCROP = machineLearning.generateModel(data,'lda','SVM_LIN')
timeLDA_L = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of linear SVM in LDA subspace (CROPPED):")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(LDA_L_accCROP.accuracyMales*100))
print("Female Accuracy: %.2f"%(LDA_L_accCROP.accuracyFemales*100))
print("Total Accuracy: %.2f"%(LDA_L_accCROP.accuracyTotal  *100))


#Non-Linear SVM in linear discriminant analysis space 
t1 = time.time() 
LDA_NL_accCROP = machineLearning.generateModel(data,'lda','SVM_NL')
timeLDA_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print("Performance of non-linear SVM in LDA subspace (CROPPED):")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(LDA_NL_accCROP.accuracyMales*100))
print("Female Accuracy: %.2f"%(LDA_NL_accCROP.accuracyFemales*100))
print("Total Accuracy: %.2f"%(LDA_NL_accCROP.accuracyTotal  *100))


#Non linear SVM using raw pixel vectors
t1 = time.time() 
RAW_L_CROPPEDacc = machineLearning.generateModel(data,'raw','SVM_LIN')
timeRAW_L_CROPPED = time.time()-t1
print('   ')
print('------------------------------------------')
print('------------------------------------------')
print("Performance of linear SVM on CROPPED raw pixels:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(RAW_L_CROPPEDacc.accuracyMales*100))
print("Female Accuracy: %.2f"%(RAW_L_CROPPEDacc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(RAW_L_CROPPEDacc.accuracyTotal  *100))



#Non non-linear SVM using raw pixel vectors
t1 = time.time() 
RAW_NL_CROPPEDacc = machineLearning.generateModel(data,'raw','SVM_NL')
timeRAW_NL = time.time()-t1
print('   ')
print('------------------------------------------')
print('------------------------------------------')
print("Performance of linear SVM on CROPPED raw pixels:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(RAW_NL_CROPPEDacc.accuracyMales*100))
print("Female Accuracy: %.2f"%(RAW_NL_CROPPEDacc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(RAW_NL_CROPPEDacc.accuracyTotal  *100))


results = [PCA_L_accCROP.accuracyTotal,PCA_NL_accCROP.accuracyTotal,LDA_L_accCROP.accuracyTotal,LDA_NL_accCROP.accuracyTotal, RAW_L_CROPPEDacc.accuracyTotal, RAW_NL_CROPPEDacc.accuracyTotal]
labels = ["PCA-L", "PCA-NL", "LDA-L", "LDA-NL", "RAW-L","RAW-NL" ]
timeResults = [timePCA_L,timePCA_NL,timeLDA_L,timeLDA_NL,timeRAW_L,timeRAW_NL]
f = plt.figure()
ax = f.add_subplot(121)
ax.bar(range(6),results,tick_label=labels)
ax.set_ylabel("Percent correct") 
ax.set_title("Performance of models on cropped images.")
ax2 = f.add_subplot(122)
ax2.set_title("Time taken of models on cropped images.")
ax2.bar(range(6),timeResults,tick_label=labels)
ax2.set_ylabel("Seconds") 

# So looking at the results we can see that the best performance 0f 80-85% was the non-linear SVM on the raw pixels, LDA came second, while PCA produced the lowest 
#accuracy scores. Another thing to note is that the non-linear kernel produced very modest improvements in most cases so it is likely the gender information in these 
# Feature spaces were linearly separable. Finally, the cropping of the images resulted in a modest improvement in almost all cases. To improve results further I would look into 
# using a CNN on the raw pixels. CNNs are effective at merging the feature engineering and classification stages and would likely improve performance significantly on this
# dataset.

 
 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:41:32 2017

@author: Felix
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


#Keras CNN on raw pixels
t1 = time.time() 
CNN_acc = machineLearning.generateModel(data,'raw','DEEP')
timeCNN = time.time()-t1
print('   ')
print('------------------------------------------')
print('------------------------------------------')
print("Performance of CNN on raw pixels:")
print('------------------------------------------')
print("Male Accuracy: %.2f"%(CNN_acc.accuracyMales*100))
print("Female Accuracy: %.2f"%(CNN_acc.accuracyFemales*100))
print("Total Accuracy: %.2f"%(CNN_acc.accuracyTotal  *100))
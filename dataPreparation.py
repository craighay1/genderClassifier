#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 21:47:38 2017

@author: fmercermoss
"""
import glob
import numpy as np
from PIL import Image
import scipy.stats as sp
from sklearn.model_selection import train_test_split


class calculateImageVectorsFromDirectories():
    
    #Class variables...
    maleFiles = []
    femaleFiles = []
    
    maleImages = []
    femaleImages = []
    
    maleVectors = []
    femaleVectors = []
    
    imsPerMaleClass = []
    imsPerFemaleClass = []
    
    resolution = []
    
    y_female = []
    y_male = []
    
    x = []
    y = []
    
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    
    trainX1 = []
    trainX0 = []
    testX1 = []
    testX0 = []
    
    testSize = []
    
    #Initialise class 
    def __init__(self, femalesFolder, malesFolder, imsPerFemaleClass, imsPerMaleClass, resolution, testSize):
        
        # Get the file names for the two classes. 
        # Only get one image per individual to maximise variation and prevent potential
        # leak of information from the training to the test set.
        self.maleFiles = glob.glob(malesFolder + "/*001.jpg") 
        self.femaleFiles  = glob.glob(femalesFolder + "/*001.jpg")
        self.imsPerMaleClass = imsPerMaleClass
        self.imsPerFemaleClass = imsPerFemaleClass
        self.resolution = resolution
        self.testSize = testSize
        self.maleVectors = np.zeros(shape=(imsPerMaleClass,resolution[0]*resolution[1]))
        self.femaleVectors = np.zeros(shape=(imsPerFemaleClass,resolution[0]*resolution[1]))
        self.calculateImageVectorsGreyScale()
        
        
        
    def calculateImageVectorsGreyScale(self):
        
        # Female images...    
        for file in enumerate(self.femaleFiles[0:self.imsPerFemaleClass]):
            
            # Open and convert to greyscale.   
            # By throwing away the colour data we are making an assumption 
            # that little of this contains gender specific information.
            im_tmp = Image.open(file[1]).convert('LA')
            
            # The original images are 250x250 pixels. However, intuitively, a resolution this high is not 
            # needed to identify gender. Therefore to reduce complexity and for computational efficiency it is a good idea to 
            # resize the image to be smaller. The Lanczos filter used here performs better at downsampling than the standard bicubic 
            # approach. 
            im = im_tmp.resize(size=(self.resolution[0],self.resolution[1]),resample=Image.LANCZOS) 
            
            # Next we convert the image to a more versatile numpy array then flatten it into a feature vector that can be easily digested 
            # by machine learning algorithms.
            v = np.array(im)
            v2 = v[:,:,0].flatten()
            
            # Normalise the values
            v3 = sp.zscore(v2)
            self.femaleVectors[file[0],:] = v3
            
            
         # Male images...
        for file in enumerate(self.maleFiles[0:self.imsPerMaleClass]):
                # Open and convert to greyscale.
                im_tmp = Image.open(file[1]).convert('LA')
                
                # Resize.
                im = im_tmp.resize(size=(self.resolution[0],self.resolution[1]),resample=Image.LANCZOS)
                v = np.array(im)
                v2 = v[:,:,0].flatten()
                
                # Normalise the values.
                v3 = sp.zscore(v2)
                self.maleVectors[file[0],:] = v3
          
        #Calculate the training and test split after calculating the vectors  
        self.generateTrainTestSplit(self.testSize)
                
    
    def calculateImageVectorsCropped(self, cropParams):
            
            # Reread in the images, but this time crop each one according to the parameters received.  
        
            # Female images...  
            for file in enumerate(self.femaleFiles[0:self.imsPerFemaleClass]):
                
                # Open and convert to greyscale.
                im_tmp = Image.open(file[1]).convert('LA')
                #area = (70, 70, 190, 190)
                
                # Cropping the images removes much of the irrelevant data surrounding the face.
                im_tmp2 = im_tmp.crop(cropParams)
                
                # Resize.
                im = im_tmp2.resize(size=(self.resolution[0],self.resolution[1]),resample=Image.LANCZOS) 
                v = np.array(im)
                v2 = v[:,:,0].flatten()
                
                # Normalise the values.
                v3 = sp.zscore(v2)
                self.femaleVectors[file[0],:] = v3
            
            
            # Male images...
            for file in enumerate(self.maleFiles[0:self.imsPerMaleClass]):
                
                # Open and convert to greyscale.
                im_tmp = Image.open(file[1]).convert('LA')
                
                # Crop images.
                im_tmp2 = im_tmp.crop(cropParams)
                
                # Resize.
                im = im_tmp2.resize(size=(self.resolution[0],self.resolution[1]),resample=Image.LANCZOS)
                v = np.array(im)
                v2 = v[:,:,0].flatten()
                
                # Normalise the values.
                v3 = sp.zscore(v2)
                self.maleVectors[file[0],:] = v3
             
            #Calculate the training and test split after calculating the vectors      
            self.generateTrainTestSplit(self.testSize)
    
    def generateTrainTestSplit(self, testSize):
         
         # Splitting the dataset up into training and test set is a vital part of all machine learning.
         # Models are trained on a part of the dataset that is completely independent of the data that it is tested with.
         # This ensures the model does not overfit spurious idiosyncracies in the training data and is a reflection of how well
         # the model generalises to the underlying function the procedure is aiming to model.
         
         
         # N.B. Typically it is considered good form to perform k-fold (e.g. k=10) cross validation. This is where the dataset up is divided up into 
         # k test sets with the remainder acting as training data. K models are then trained and the average performance metric of the k models is considered 
         # to be more reliable than a single test score. Here I do only a single split to save computation time. 
         
         self.y_male = np.ones(shape=(self.maleVectors.shape[0],1))
         self.y_female = np.zeros(shape=(self.femaleVectors.shape[0],1))
         self.y = np.concatenate((self.y_male,self.y_female),axis=0)
         self.x = np.concatenate((self.maleVectors,self.femaleVectors),axis=0)

         self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.x, self.y, test_size=testSize, random_state=None)


         self.trainX1 = self.Xtrain[self.Ytrain[:,0]==1]
         self.trainX0 = self.Xtrain[self.Ytrain[:,0]==0]
         self.testX1 = self.Xtest[self.Ytest[:,0]==1]
         self.testX0 = self.Xtest[self.Ytest[:,0]==0]
        
    
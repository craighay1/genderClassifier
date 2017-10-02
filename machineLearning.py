#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:46:06 2017

@author: fmercermoss
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc



class generateModel():
    
    data = []
    accuracyTotal = []
    accuracyFemales = []
    accuracyMales = []
    probabilities = []
    def __init__(self, data, featureSpace, classifier):
        
        self.data = data
        
        if featureSpace == "lda":
            
            if classifier == "SVM_LIN":
                
                self.ldaSVM("linear")
              
            elif classifier == "SVM_NL":
                
                self.ldaSVM("rbf")
            
        elif featureSpace == "pca":
            
            if classifier == "SVM_LIN":
                
               self.pcaSVM("linear") 
               
            elif classifier == "SVM_NL":
                
               self.pcaSVM("rbf") 
               
        if featureSpace == "raw":
            
              if classifier == "SVM_LIN":
                  
                  self.rawSVM("linear")
                  
              if classifier == "SVM_NL":          
                    
                  self.rawSVM("rbf")
                  
              if classifier == "DEEP":
                  self.rawDEEP()
                  
                  
        
    
    def pcaSVM(self, kernel):
        
        pca = PCA()
        Xtrain_pca = pca.fit(self.data.Xtrain).transform(self.data.Xtrain)
        
        # PCA will automatically generate as many output components as there are input components but crucially, these components are sorted according to 
        # the amount of variance they explain (Eigen values). Plotting these Eigenvalues in the "Scree Plot"  gives us an indication of how the variance is spread 
        # between the principal components, and also how many of the tailing components can be dropped. Let's have a look at that now...
        
        if kernel=="linear": # Only need to display once. 
            figScree = plt.figure()
            axScree = figScree.add_subplot(111)
            axScree.plot(pca.explained_variance_, color='navy')
            axScree.set_xlabel('Principal components (Eigen Vectors)')
            axScree.set_ylabel('Explained variance (Eigen Values)')
            figScree.suptitle("Scree plot indicating how much variance is explained.")
            #figScree.show()
        
        
        # There are many criteria that could be used to eliminate principal components but a simple one we can use here is the Kaiser criterion
        # which says we csn remove all components with an eigenvalue below 1.
        kaiserCriterionN = len(pca.explained_variance_[pca.explained_variance_>1])
        pca = PCA()
        pca = PCA(n_components = kaiserCriterionN)
        
        # Now we can project the training and test vectures into the lower dimensional subspace identified by the PCA.
        Xtrain_pca = pca.fit(self.data.Xtrain).transform(self.data.Xtrain)
        Xtest_pca = pca.fit(self.data.Xtrain).transform(self.data.Xtest)
        
        
        # Time to start training our first classsifying model - a support vector machine. SVMs are not the quickest to train on big datasets
        # and difficult to analyse introspectively but from my experience they tend to give the best performance with minimal tinkering. 
        # SVMs are fundamentally a linear classifier but by using a non-linear kernel they can project non-linear data into linear space.
        # The components identified by PCA are fundamentally uncorrelated so this is a reason to believe a linear kernel should be sufficient to 
        #separate the data.
        svmPCA = SVC(kernel=kernel)
       
        if kernel == "linear":
            # It's always a good idea to reduce the complexity of the model by removing features in a stepwise process. This an iterative model selection procedure
            # whereby features are added (or subtracted) from a model based upon the significance of increase (or decrease) in the goodness-of-fit of the model. 
            # Here we will reduce the features to just 20 of the most useful.
            selector = RFE(svmPCA, 20, step=1)
            selector.fit(Xtrain_pca,self.data.Ytrain.ravel())
            selectedFeatures = [f[0] for f in enumerate(selector.support_) if f[1]]
            
            print("Selected features:")
            print(selectedFeatures)
            # We see that the 20 features selected by the stepwise process are not the top 20 principal components. This is an indication that the gender 
            # information is not represented in the dimensions of most variance.
        
        
            # Now we will train the model again with only the selected features...
            svmPCA = SVC(kernel=kernel)
            svmPCA.fit(Xtrain_pca[:,selectedFeatures], self.data.Ytrain.ravel())
        
        
            # ...and predict the test set using the same selected PCA features.
            predictedPCA = svmPCA.predict(Xtest_pca[:,selectedFeatures])
        else:
            
            #The stepwise process is currently not available in SKlearn for the non-linear kernel so we will use all the features.
            svmPCA.fit(Xtrain_pca, self.data.Ytrain.ravel())
            predictedPCA = svmPCA.predict(Xtest_pca)
                
        # And now we can calculate our first set of results.
        m_acc= np.mean(predictedPCA[self.data.Ytest.ravel()==1])
        f_acc= 1-np.mean(predictedPCA[self.data.Ytest.ravel()==0])
        acc = (f_acc+m_acc)/2
        
        self.probabilities = svmPCA.decision_function
        self.accuracyTotal = acc
        self.accuracyFemales = f_acc
        self.accuracyMales = m_acc
        #self.calculateAreaUnderROC()
        
        
        return acc
        
        
        
    def ldaSVM(self,kernel):
        
  

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda_trained = lda.fit(self.data.Xtrain, self.data.Ytrain.ravel())
        Xtrain_lda = lda_trained.transform(self.data.Xtrain)
        Xtest_lda = lda.fit(self.data.Xtrain, self.data.Ytrain.ravel()).transform(self.data.Xtest)
        Xtest0_lda = lda.fit(self.data.Xtrain, self.data.Ytrain.ravel()).transform(self.data.testX0)
        Xtest1_lda = lda.fit(self.data.Xtrain, self.data.Ytrain.ravel()).transform(self.data.testX1)
        print("got this far")
        
        #Let's have a look at how well the training data has been separated in the subspace
        n = []
        f1 = plt.figure()
        ax1 = f1.add_subplot(121)
        n.append(sum(self.data.Ytrain[:,0] == 0))
        n.append(sum(self.data.Ytrain[:,0] == 1))
        labels = ["Females", "Males"]
        colours = ["blue","darkorange"]
        for colour, i, label in zip(colours, [0, 1], labels):
            ax1.scatter(Xtrain_lda[self.data.Ytrain[:,0]==i], np.arange(0,n[i]), alpha=.8, color=colour,
                        label=label)
            ax1.set_title("")
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('LDA of Faces dataset - Training Data')
            
            
        
        #Looks good so now we have some much better separation, let's have a look at the test data
            
        ax2 = f1.add_subplot(122)
        n = []
        n.append(sum(self.data.Ytest[:,0] == 0))
        n.append(sum(self.data.Ytest[:,0] == 1))
        for colour, i, label in zip(colours, [0, 1], labels):
            ax2.scatter(Xtest_lda[self.data.Ytest[:,0]==i], np.arange(0,n[i]), alpha=.8, color=colour,
                        label=label)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('LDA of Faces dataset - Test Data')
            #plt.show()
            
                

            
        svmLDA = SVC(kernel=kernel)
        modelTrained = svmLDA.fit(Xtrain_lda, self.data.Ytrain.ravel())
        predictedAll = modelTrained.predict(Xtest_lda)
                
        m_acc_LDA= np.mean(predictedAll[self.data.Ytest.ravel()==1])
        f_acc_LDA= 1-np.mean(predictedAll[self.data.Ytest.ravel()==0])
        acc_LDA = (f_acc_LDA+m_acc_LDA)/2
   
        self.probabilities = svmLDA.decision_function        
        self.accuracyTotal = acc_LDA
        self.accuracyFemales = f_acc_LDA
        self.accuracyMales = m_acc_LDA        
        return acc_LDA
            
            
            
    def rawSVM(self,kernel):  
        
        # Now let's try an SVM on the raw pixels... 
        svmRAW = SVC(kernel=kernel)
        #
     
        svmRAW.fit(self.data.Xtrain, self.data.Ytrain.ravel())
        predictedPCA = svmRAW.predict(self.data.Xtest)
        
        m_acc_raw= np.mean(predictedPCA[self.data.Ytest.ravel()==1])
        f_acc_raw= 1-np.mean(predictedPCA[self.data.Ytest.ravel()==0])
        acc_raw = (f_acc_raw+m_acc_raw)/2
        
        self.probabilities = svmRAW.decision_function
        self.accuracyTotal = acc_raw
        self.accuracyFemales = f_acc_raw
        self.accuracyMales = m_acc_raw 
        return acc_raw
    
    
    def rawDEEP(self):
        
        # For a single-input model with 2 classes (binary classification):
        #import ipdb; ipdb.set_trace()
       # from keras.models import Sequential
       # from keras.layers import Dense, Activation
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        
        #First let's reshape the data into 2D so we can do convolution on layers
      
        
        
        deepModel = Sequential()
        deepModel.add(Dense(32, activation='relu', input_dim=(self.data.Xtrain.shape[1])))
        deepModel.add(Dense(1, activation='relu'))
        deepModel.add(Dense(1, activation='relu'))
        deepModel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

        # Generate dummy data
        #import numpy as np
       # x = np.random.random((1920, 576))
       # labels = np.random.randint(2, size=(576, 1))

        # Train the model, iterating on the data in batches of 32 samples
        deepModel.fit(self.data.Xtrain, self.data.Ytrain, epochs=10, batch_size=32)
       

        

        evalu = deepModel.evaluate(self.data.Xtest, self.data.Ytest, batch_size=32)
        
        predictedDEEP = np.round(deepModel.predict(self.data.Xtest))
       
        m_acc_raw= np.mean(predictedDEEP[self.data.Ytest.ravel()==1])
        f_acc_raw= 1-np.mean(predictedDEEP[self.data.Ytest.ravel()==0])
        acc_raw = (f_acc_raw+m_acc_raw)/2
        self.accuracyTotal = acc_raw
        self.accuracyFemales = f_acc_raw
        self.accuracyMales = m_acc_raw 
    
    def calculateAreaUnderROC(self):
        
        
       
        fpr, tpr, thresholds = roc_curve(self.data.Ytest, self.probabilities, pos_label=2)
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
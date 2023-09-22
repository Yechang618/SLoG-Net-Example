import numpy as np
from numpy import linalg as LA
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg
from timeit import default_timer as timer
import networkx as nx
import os
import pickle
import datetime
#### Import SLOG packakes
from SLoGexample import SLOGtools as SLOGtools
from SLoGexample import SLOGobjective as SLOGobj
from SLoGexample import SLOGarchitectures as SLOGarchi 
from SLoGexample import SLOGtraining as SLOGtrainer
from SLoGexample import SLOGmodel as SLOGmodel
from SLoGexample import SLOGevaluation as SLOGevaluator
from SLoGexample import SLOGdata as SLOGdata

#### Import GNN packages
# from SLOGmodules import graphTools as graphTools
# from SLOGmodules import dataTools as dataTools
# from alegnn.utils import graphML as gml

class slog_experiments():
    def __init__(self, simuParas = None, 
                 graphOptions = None,  **kwargs):
        
        # Simulation parameters
        self.simuParas = simuParas
        self.nTrain_slog = simuParas['nTrain_slog']
        self.batchsize_slog = simuParas['batchsize_slog']
        self.nValid = simuParas['nValid']
        self.nTest = simuParas['nTest']
        self.L = simuParas['L']
        self.noiseLevel = simuParas['noiseLevel']
        self.noiseType = simuParas['noiseType']
        self.filterType = simuParas['filterType']
        self.signalMode = simuParas['signalMode']
        self.trainMode = simuParas['trainMode']
        self.filterMode = simuParas['filterMode']
        self.selectMode = simuParas['selectMode']
        self.nNodes = simuParas['nNodes']
        self.nClasses = simuParas['nClasses']
        self.N_C = simuParas['N_C']
        self.S = simuParas['S']
        self.graphType = simuParas['graphType']
        self.tMax = simuParas['tMax']
        self.alpha = simuParas['alpha']
        self.nEpochs = simuParas['nEpochs']
        # if 'model_number' in simuParas.keys():
        #     self.model_number = simuParas['model_number']
        # else:
        #     self.model_number = 0
                
        # Graph options
        self.graphOptions = graphOptions
        
        # Model parameters (optional)
        if 'modelParas' in kwargs.keys():
            self.modelParas = kwargs['modelParas']
            self.C = self.modelParas['C']
            self.K = self.modelParas['K']
            self.filterTrainType = self.modelParas['filterTrainType']
        else:
            self.C = self.nNodes
            self.K = 5
            self.filterTrainType = 'g'
            self.modelParas = {}
            self.modelParas['C']= self.C
            self.modelParas['K']= self.K
            self.modelParas['filterTrainType']= self.filterTrainType
        if 'q' in self.modelParas.keys():
            self.q = self.modelParas['q']
        else:
            self.q = 4
            
        # Experiment parameters (optional)
        if 'expParas' in kwargs.keys():
            self.expParas = kwargs['expParas']
            self.nRealiz = self.expParas['nRealiz']
        else:
            self.expParas = {}
            self.nRealiz = 1
            self.expParas['nRealiz'] = self.nRealiz
            
        if 'saveSettings' in kwargs.keys():
            self.saveSettings = kwargs['saveSettings']
            self.thisFilename_SLOG = self.saveSettings['thisFilename_SLOG']
            self.saveDirRoot = self.saveSettings['saveDirRoot'] # Relative location where to save the file
            self.saveDir = self.saveSettings['saveDir']# Dir where to save all the results from each run
        else:
            self.thisFilename_SLOG = 'sourceLocSLOGNET'
            self.saveDirRoot = 'experiments' # Relative location where to save the file
            self.saveDir = os.path.join(self.saveDirRoot, self.thisFilename_SLOG) # Dir where to save all the results from each run

            self.saveSettings = {}
            self.saveSettings['thisFilename_SLOG'] = self.thisFilename_SLOG            
            self.saveSettings['saveDirRoot'] = self.saveDirRoot
            self.saveSettings['saveDir'] = self.saveDir

        self.experiment_results = []
        for i in range(self.nRealiz):
            result_i = self.run_single_experiment()
            self.experiment_results.append(result_i)
       
    def get_experiment_result(self):
        return self.experiment_results
            
    def run_single_experiment(self,**kwargs):    
        ## kwargs:

        #\\\ Create .txt to store the values of the setting parameters for easier
        # reference  when running multiple experiments
        today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Append date and time of the run to the directory, to avoid several runs of
        # overwritting each other.
        saveDir = self.saveDir + '-' + self.graphType + '-' + today
        # saveDir_dropbox = self.saveDir_dropbox + '-' + self.graphType + '-' + today
        # 
        saveDirs = {}
        saveDirs['saveDir'] = saveDir

        # Create directory
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        useGPU = True
        if useGPU and torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
        # Notify:
        print("Device selected: %s" % device)   
  
        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        #\\\ Save values:
        SLOGtools.writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        SLOGtools.writeVarValues(varsFile, self.graphOptions)
        SLOGtools.writeVarValues(varsFile, self.simuParas)
        SLOGtools.writeVarValues(varsFile, self.modelParas)           
        SLOGtools.writeVarValues(varsFile, saveDirs)
        SLOGtools.writeVarValues(varsFile, {'nTrain_slog': self.nTrain_slog,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'tMax': self.tMax,
                                  'nClasses': self.nClasses,
                                  'useGPU': useGPU})


        if 'optimAlg' in kwargs.keys():
            optimAlg = kwargs['optimAlg']
        else:
            optimAlg = 'ADAM'
            
        if 'learningRate' in kwargs.keys():
            learningRate = kwargs['learningRate']
        else:
            learningRate = 0.01
            
        if 'beta1' in kwargs.keys():
            beta1 = kwargs['beta1']
        else:
            beta1 = 0.9
            
        if 'beta2' in kwargs.keys():
            beta2 = kwargs['beta2']
        else:
            beta2 = 0.999
            
        ## Graph generation
        G = SLOGtools.Graph(self.graphType, self.nNodes, self.graphOptions, save_dir = saveDir)
        G.computeGFT()
        d_slog,An_slog, eigenvalues_slog, V_slog   = SLOGtools.get_eig_normalized_adj(G.A)
        
        ## Data generation
        data = SLOGdata.SLOG_GeneralData(G, self.nTrain_slog, self.nValid, self.nTest, self.S, V_slog, eigenvalues_slog, L = self.L, alpha = self.alpha,filterType = self.filterType, noiseLevel = self.noiseLevel, noiseType = self.noiseType)
        data.expandDims()
        
        C = self.C
        K = self.K
        filterTrainType = self.filterTrainType #'g'
        thisLoss = SLOGtools.myLoss
        thisEvaluator = SLOGevaluator.evaluate
        
        thisObject = SLOGobj.myFunction_slog_3
        SLOG_net = SLOGarchi.GraphSLoG_v3(V_slog,self.nNodes,self.q,self.K, thisObject)        

        model_name = 'SLOG-Net'

        thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
        thisTrainer = SLOGtrainer.slog_Trainer

        myModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator,device, model_name,  saveDir)

        result_train = myModel.train(data,self.nEpochs, self.batchsize_slog, validationInterval = 40,trainMode = self.trainMode,tMax = self.tMax, filterTrainType = self.filterTrainType) # model, data, nEpochs, batchSize
        
        best_model = result_train['bestModel']
        minLossValid = result_train['minLossValid']
        minLossTrain = result_train['minLossTrain']
          
        results = {}
        results['model'] = myModel
        results['training result'] = result_train
        results['Graph'] = G
        
        return results

def noiseTest_dropbox(nNodes,P,S, modelDir, **kwargs):
    ## Assertation
    
    ## Parameters loading (kwargs)
    if 'location' in kwargs.keys():
        location = kwargs['location']
    else:
        location = 'office'

    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4
        
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}        
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha
        
    if 'normalize_g_hat' in kwargs.keys():
        normalize_g_hat = kwargs['normalize_g_hat']
    else:
        normalize_g_hat = False       
        
    if 'N_C' in simuParas.keys(): 
        # Number of sources per signal in classification
        N_C = simuParas['N_C']
    else:
        N_C = 3
        simuParas['N_C'] = N_C    
        
    if 'nClasses' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        nClasses = simuParas['nClasses']
    else:
        nClasses = 3
        simuParas['nClasses'] = nClasses    
   
    if 'graphType' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType 

    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
        graphOptions['probIntra'] = 0.3 # Probability of drawing edges
        simuParas['graphOptions'] = graphOptions

    if 'L' in simuParas.keys():
        L = simuParas['L']
    else:
        L = 5
        simuParas['L'] = L
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType    
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType        

    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C        

    if 'K' in simuParas.keys():
        K = simuParas['K']
    else:
        K = 5
        simuParas['K'] = K        

    if 'N_realiz' in simuParas.keys():
        N_realiz = simuParas['N_realiz']
    else:
        N_realiz = 10
        simuParas['N_realiz'] = N_realiz  

    ## Model settings
    #
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
        
    if 'model_number' in simuParas.keys():
        model_number = simuParas['model_number']
    else:
        model_number = 0        
        
    if 'thisLoss' in modelSettings.keys():
        thisLoss = modelSettings['thisLoss']
    else:
        thisLoss = SLOGtools.myLoss
        
    if 'thisEvaluator' in modelSettings.keys():
        thisEvaluator = modelSettings['thisEvaluator']
    else:
        thisEvaluator = SLOGevaluator.evaluate   
        
    if 'thisObject' in modelSettings.keys():
        thisObject = modelSettings['thisObject']
    else:
        thisObject = SLOGobj.myFunction_slog_1
 
    model_name = 'SLOG-Net'
    device = 'gpu'
    optimAlg = 'ADAM'
    learningRate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    ## Save dir
    
    if location == 'home':
        print('Running test at ', location)
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    elif location == 'office':
        print('Running test at ', location)        
        saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
    else:
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
        
    ## Generate modelSaveDir
    # modelDirList
    label = 'Best'
    saveDir = os.path.join(saveDir_dropbox,modelDir)
    gsoName = 'gso-' + graphType + '.npy'
    gsoDir = os.path.join(saveDir, gsoName)
    GA = np.load(gsoDir)
    d,An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)
    gso = An
    SLOG_net = SLOGarchi.GraphSLoG_v1(V,nNodes,C,K, thisObject)
    thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
    thisTrainer = SLOGtrainer.slog_Trainer    
    loadedModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator, device, model_name,  None)
    loadedModel.load_from_dropBox(saveDir, label = label)        
    
    # Test begins
    result = {}
    re_x = np.zeros(N_realiz)
    re_g = np.zeros(N_realiz)    
    for n_realiz in range(N_realiz):
        X = SLOGtools.X_generate(nNodes,P,S)
        if filterType == 'g':
            g0 = SLOGtools.g_generate_gso(nNodes,alpha, eigenvalues,L)
        else:
            g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
        X = to_numpy(X)
        g0 = to_numpy(g0)
        V = to_numpy(V)
        if normalize_g_hat:
            g0 = nNodes*g0/np.sum(g0)
        else:
            g0 = C*g0/np.sum(g0)
        h0 = 1./g0
        H = np.dot(V,np.dot(np.diag(h0),V.T))
        if noiseType == 'gaussion':
            noise = np.random.normal(0,1,[nNodes, P])
            noise = noise/LA.norm(noise,'fro')*LA.norm(X,'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1,1,[nNodes, P])
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes, P])
        Y = np.dot(H,X) + noiseLevel*noise
        Y_test = to_torch(Y)
        x_hat, g_hat = loadedModel.archit(Y_test)
        g_hat = to_numpy(g_hat)  
        
        if normalize_g_hat:
            g_hat = nNodes*g_hat/np.sum(g_hat)      
        
        Z = linalg.khatri_rao(np.dot(Y.T,V),V)
#         print('Sum of g', np.sum(g_hat), np.sum(g0))
        x_recv = np.dot(Z,g_hat)
        X_recv = x_recv.reshape((P,nNodes)).T
        re_x_1 = LA.norm(X_recv - X,'fro')/LA.norm(X,'fro')
        re_g_1 = LA.norm(g0 - g_hat)/LA.norm(g0)
        re_x_2 = LA.norm(X_recv + X,'fro')/LA.norm(X,'fro')
        re_g_2 = LA.norm(g0 + g_hat)/LA.norm(g0) 
        if re_g_1 > re_g_2:
            re_g[n_realiz] = re_g_2
            re_x[n_realiz] = re_x_2
        else:
            re_g[n_realiz] = re_g_1
            re_x[n_realiz] = re_x_1               
    result['re_x'] = re_x    
    result['re_g'] = re_g     
    
    return result

############### Functions ############# 
def to_numpy(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return x
    elif 'torch' in repr(dataType):
        x1 = x.clone().detach().requires_grad_(False)
        return x1.numpy()
    
def to_torch(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return torch.tensor(x)
    elif 'torch' in repr(dataType):
        return x  
    

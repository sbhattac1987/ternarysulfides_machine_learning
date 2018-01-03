"""
Author: Dr Sandip Bhattacharya (sbhattac@tcd.ie)
Python3 or higher should be used 
The input files obtained from a database of 1400 ternary sulfides [from my paper Journal of Materials Chemistry A 4 (28), 11086-11093 (2016)] can be obtained upon request

The code uses Machine Learning (ML) methods to achieve the following:
a) Evaluate the feature importances to obtain physical insights into which features are required for good materials properties.
b) Build a ML model that can predict the necessary values of the electronic properties required for a high zT

The code is demonstrated for random forest regression (using sklearn). However, neural networks or SVM methods can easily be implemented as illustrated within the code.

For more details on the purpose of this project please refer to the README file

"""
import pylab, random,pickle,re
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import *
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
import scipy
import matplotlib.pyplot as plt

#AE_data = pickle.load( open( "AE_data.p", "rb" ) )

with open('AE_data.p', 'rb') as f:
    AE_data = pickle.load(f, encoding='latin1') 

#zT0=1.15


def minkowskiDist(v1, v2, p):
    """Assumes v1 and v2 are equal-length arrays of numbers
       Returns Minkowski distance of order p between v1 and v2"""
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)


class tern_sulfides(object):
    featureNames =('gap','SGNr','av_Z','av_eig_sl','av_omax_sl','av_kin_sl','av_infl_sl','av_eig_l','av_omax_l','av_kin_l','av_infl_l',
                                'hi_Z','hi_eig_sl','hi_omax_sl','hi_kin_sl','hi_infl_sl','hi_eig_l','hi_omax_l','hi_kin_l','hi_infl_l', 
                                'lw_Z','lw_eig_sl','lw_omax_sl','lw_kin_sl','lw_infl_sl','lw_eig_l','lw_omax_l','lw_kin_l','lw_infl_l')

    def __init__(self,gap,SGNr,av_Z,av_eig_sl,av_omax_sl,av_kin_sl,av_infl_sl,av_eig_l,av_omax_l,av_kin_l,av_infl_l,
                              hi_Z,hi_eig_sl,hi_omax_sl,hi_kin_sl,hi_infl_sl,hi_eig_l,hi_omax_l,hi_kin_l,hi_infl_l, 
                              lw_Z,lw_eig_sl,lw_omax_sl,lw_kin_sl,lw_infl_sl,lw_eig_l,lw_omax_l,lw_kin_l,lw_infl_l,highpf,pf,name):
        self.name = name
        self.pf = pf
        self.featureVec = [gap,SGNr,av_Z,av_eig_sl,av_omax_sl,av_kin_sl,av_infl_sl,av_eig_l,av_omax_l,av_kin_l,av_infl_l,
                                   hi_Z,hi_eig_sl,hi_omax_sl,hi_kin_sl,hi_infl_sl,hi_eig_l,hi_omax_l,hi_kin_l,hi_infl_l, 
                                   lw_Z,lw_eig_sl,lw_omax_sl,lw_kin_sl,lw_infl_sl,lw_eig_l,lw_omax_l,lw_kin_l,lw_infl_l]
        self.label = highpf
    def distance(self, other):
        return minkowskiDist(self.featureVec, other.featureVec, 2)
    def getgap(self):
        return self.featureVec[0]
    def getSGNr(self):
        return self.featureVec[1]
 
    def getav_Z(self):
        return self.featureVec[2]
    def getav_eig_sl(self):
        return self.featureVec[3]
    def getav_omax_sl(self):
        return self.featureVec[4]
    def getav_kin_sl(self):
        return self.featureVec[5]
    def getav_infl_sl(self):
        return self.featureVec[6]
    def getav_eig_l(self):
        return self.featureVec[7]
    def getav_omax_l(self):
        return self.featureVec[8]
    def getav_kin_l(self):
        return self.featureVec[9]
    def getav_infl_l(self):
        return self.featureVec[10]
    
    def gethi_Z(self):
        return self.featureVec[11]
    def gethi_eig_sl(self):
        return self.featureVec[12]
    def gethi_omax_sl(self):
        return self.featureVec[13]
    def gethi_kin_sl(self):
        return self.featureVec[14]
    def gethi_infl_sl(self):
        return self.featureVec[15]
    def gethi_eig_l(self):
        return self.featureVec[16]
    def gethi_omax_l(self):
        return self.featureVec[17]
    def gethi_kin_l(self):
        return self.featureVec[18]
    def gethi_infl_l(self):
        return self.featureVec[19]


    def getlw_Z(self):
        return self.featureVec[20]
    def getlw_eig_sl(self):
        return self.featureVec[21]
    def getlw_omax_sl(self):
        return self.featureVec[22]
    def getlw_kin_sl(self):
        return self.featureVec[23]
    def getlw_infl_sl(self):
        return self.featureVec[24]
    def getlw_eig_l(self):
        return self.featureVec[25]
    def getlw_omax_l(self):
        return self.featureVec[26]
    def getlw_kin_l(self):
        return self.featureVec[27]
    def getlw_infl_l(self):
        return self.featureVec[28]

    def getName(self):
        return self.name
    def getpf(self):
        return self.pf
    def getFeatures(self):
        return self.featureVec[:]
    def getLabel(self):
        return self.label
        

def getsulfideData(temp,pf0,dope):
    data = {}
    data['gap'],data['SGNr'] = [], []
    data['av_Z'],data['av_eig_sl'],data['av_omax_sl'],data['av_kin_sl'],data['av_infl_sl'],data['av_eig_l'],data['av_omax_l'],data['av_kin_l'],data['av_infl_l']=[], [],[], [],[], [],[], [],[]
    data['hi_Z'],data['hi_eig_sl'],data['hi_omax_sl'],data['hi_kin_sl'],data['hi_infl_sl'],data['hi_eig_l'],data['hi_omax_l'],data['hi_kin_l'],data['hi_infl_l']=[], [],[], [],[], [],[], [],[]
    data['lw_Z'],data['lw_eig_sl'],data['lw_omax_sl'],data['lw_kin_sl'],data['lw_infl_sl'],data['lw_eig_l'],data['lw_omax_l'],data['lw_kin_l'],data['lw_infl_l']=[], [],[], [],[], [],[], [],[]
    data['pf'],data['highpf'],data['name'] = [], [],[]
    f = open("final_%s.dat"%temp)
    lines = f.readlines()
    f.close()

    for l in range(1,len(lines)):
        tmp0=lines[l].split("\n")[0].split(" ")
        tmp=list(filter(lambda a: a != '', tmp0)) 
        #print (tmp)
        data['name'].append(tmp[0])
        data['SGNr'].append(int(tmp[2]))
        data['gap'].append(0.0 if float(tmp[4])<0.0 else float(tmp[4]))  
        if dope==2e+19:
           pip=float(tmp[5])
           data['pf'].append(pip)
        elif dope==2e+20:
           pip=float(tmp[6])
           data['pf'].append(pip)
        elif dope==2e+21:
           pip=float(tmp[7])   
           data['pf'].append(pip)
        if dope==-2e+19:
           #print("I am here")
           pip=float(tmp[8]) 
           data['pf'].append(pip)
        elif dope==-2e+20:
           pip=float(tmp[9])
           data['pf'].append(pip)
        elif dope==-2e+21:
           pip=float(tmp[10]) 
           data['pf'].append(pip)

        if pip>pf0:
           data['highpf'].append(1)
        else:
           data['highpf'].append(0)
        sul_atoms=re.findall('[a-zA-Z]+',tmp[0])
        sul_comp=re.findall('\d+',tmp[0])

        for j,dtype in enumerate(['av','hi','lw']):
            data['%s_Z'%dtype].append(atomic_data('Z',dtype,'sl',sul_atoms,sul_comp))
            for pos in ['sl','l']: 
                data['%s_eig_%s'%(dtype,pos)].append(atomic_data('eigen',dtype,pos,sul_atoms,sul_comp)) 
                data['%s_omax_%s'%(dtype,pos)].append(atomic_data('outermax',dtype,pos,sul_atoms,sul_comp))
                data['%s_kin_%s'%(dtype,pos)].append(atomic_data('ke',dtype,pos,sul_atoms,sul_comp))
                data['%s_infl_%s'%(dtype,pos)].append(atomic_data('outerinfl',dtype,pos,sul_atoms,sul_comp))
            #data['Z%s'%comp].append(AE_data[hh_comp[j]]['sec_last']['Z'])  
            #data['eig%s_sl'%comp].append(AE_data[hh_comp[j]]['sec_last']['eigen'])
            #data['%s_omax_%s'%(dtype,pos)].append(AE_data[hh_comp[j]]['sec_last']['outermax'])
            #data['%s_kin_%s'%(dtype,pos)].append(AE_data[hh_comp[j]]['sec_last']['ke'])
            #data['%s_infl_%s'%(dtype,pos)].append(AE_data[hh_comp[j]]['sec_last']['outerinfl']) 
            #data['eig%s_l'%(dtype,pos)].append(AE_data[hh_comp[j]]['last']['eigen'])
            #data['omax%s_l'%(dtype,pos)].append(AE_data[hh_comp[j]]['last']['outermax'])
            #data['kin%s_l'%(dtype,pos)].append(AE_data[hh_comp[j]]['last']['ke'])
            #data['infl%s_%s'%(dtype,pos)].append(AE_data[hh_comp[j]]['last']['outerinfl']) 
    return data

def atomic_data(descriptor,dtype,position,sul_atoms,sul_comp):
    if position=='sl':
       des_A,des_B,des_C=AE_data[sul_atoms[0]]['sec_last'][descriptor],AE_data[sul_atoms[1]]['sec_last'][descriptor],AE_data[sul_atoms[2]]['sec_last'][descriptor]
    else:
       des_A,des_B,des_C=AE_data[sul_atoms[0]]['last'][descriptor],AE_data[sul_atoms[1]]['last'][descriptor],AE_data[sul_atoms[2]]['last'][descriptor]
    if dtype=='av':
       return ((des_A*float(sul_comp[0]))+(des_B*float(sul_comp[1]))+(des_C*float(sul_comp[2])))/float(sul_comp[0]+sul_comp[1]+sul_comp[2])
    elif dtype=='hi':
       return max(des_A,des_B,des_C)
    else:
       return min(des_A,des_B,des_C) 
                
def buildsulfideExamples(temp,pf0,dope,pfmax):
    data = getsulfideData(temp,pf0,dope)
    examples = []
    for i in range(len(data['name'])):
        p=tern_sulfides(data['gap'][i],data['SGNr'][i],data['av_Z'][i],data['av_eig_sl'][i],data['av_omax_sl'][i],data['av_kin_sl'][i],data['av_infl_sl'][i],data['av_eig_l'][i],data['av_omax_l'][i],data['av_kin_l'][i],data['av_infl_l'][i],data['hi_Z'][i],data['hi_eig_sl'][i],data['hi_omax_sl'][i],data['hi_kin_sl'][i],data['hi_infl_sl'][i],data['hi_eig_l'][i],data['hi_omax_l'][i],data['hi_kin_l'][i],data['hi_infl_l'][i],data['lw_Z'][i],data['lw_eig_sl'][i],data['lw_omax_sl'][i],data['lw_kin_sl'][i],data['lw_infl_sl'][i],data['lw_eig_l'][i],data['lw_omax_l'][i],data['lw_kin_l'][i],data['lw_infl_l'][i],data['highpf'][i],data['pf'][i],data['name'][i])
        if data['pf'][i]<pfmax:
           examples.append(p)
    print('Finished processing', len(examples), 'ternary sulfides\n')    
    return examples
    
def findNearest(name, exampleSet, metric):
    for e in exampleSet:
        if e.getName() == name:
            example = e
            break
    curDist = None
    for e in exampleSet:
        if e.getName() != name:
            if curDist == None or metric(example, e) < curDist:
                nearest = e
                curDist = metric(example, nearest)
    return nearest

def accuracy(truePos, falsePos, trueNeg, falseNeg):
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator/denominator

def sensitivity(truePos, falseNeg):
    try:
        return float(truePos)/float(truePos + falseNeg)
    except ZeroDivisionError:
        return float('nan')
    
def specificity(trueNeg, falsePos):
    try:
        return float(trueNeg)/float(trueNeg + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def posPredVal(truePos, falsePos):
    try:
        return truePos/(truePos + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def negPredVal(trueNeg, falseNeg):
    try:
        return trueNeg/(trueNeg + falseNeg)
    except ZeroDivisionError:
        return float('nan')
       
def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True):
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    if toPrint:
        print(' Accuracy =', round(accur, 3))
        print(' Sensitivity =', round(sens, 3))
        print(' Specificity =', round(spec, 3))
        print(' Pos. Pred. Val. =', round(ppv, 3))
    return (accur, sens, spec, ppv)
   
def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.distance(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = example.distance(e)
        if dist < maxDist:
            #replace farther neighbor by this one
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances
    
def KNearestClassify(training, testSet, label, k):
    """Assumes training & testSet lists of examples, k an int
       Predicts whether each example in testSet has label
       Returns number of true positives, false positives,
          true negatives, and false negatives"""
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for testCase in testSet:
        nearest, distances = findKNearest(testCase, training, k)
        #conduct vote
        numMatch = 0
        for i in range(len(nearest)):
            if nearest[i].getLabel() == label:
                numMatch += 1
        if numMatch > k//2: #guess label
            if testCase.getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else: #guess not label
            if testCase.getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

def leaveOneOut(examples, method, toPrint = True):
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(examples)):
        testCase = examples[i]
        trainingData = examples[0:i] + examples[i+1:]
        results = method(trainingData, [testCase])
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    if toPrint:
        getStats(truePos, falsePos, trueNeg, falseNeg)
    return truePos, falsePos, trueNeg, falseNeg

def split80_20(examples):
    sampleIndices = random.sample(range(len(examples)),
                                  len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet
    
def randomSplits(examples, method, numSplits, toPrint = True):
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    random.seed(0)
    for t in range(numSplits):
        trainingSet, testSet = split80_20(examples)
        results = method(trainingSet, testSet)
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    getStats(truePos/numSplits, falsePos/numSplits,
             trueNeg/numSplits, falseNeg/numSplits, toPrint)
    return truePos/numSplits, falsePos/numSplits,\
             trueNeg/numSplits, falseNeg/numSplits
    
#knn = lambda training, testSet:KNearestClassify(training, testSet, 1, 3)  #1 (>zT0)
#numSplits = 10

#true_count=0
#for i in range(0,len(examples)):
#    if examples[i].getLabel()==1:
#       true_count+=1

#print('Accuracy without any ML',round(true_count/len(examples),3))


#print('\n Average of', numSplits,'80/20 splits using KNN (k=3)')
#truePos, falsePos, trueNeg, falseNeg =randomSplits(examples, knn, numSplits)

#print('Average of LOO testing using KNN (k=3)')
#truePos, falsePos, trueNeg, falseNeg =leaveOneOut(examples, knn)

def buildModel(examples, toPrint = True):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    LogisticRegression = sklearn.linear_model.LogisticRegression
    model = LogisticRegression().fit(featureVecs, labels)
    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coef_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coef_[0])):
                print('   ', HHs.featureNames[j], '=',model.coef_[0][j])
    return model

def buildModel_nn(examples, toPrint = True):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    clf = MLPClassifier#(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    model=clf().fit(featureVecs, labels)
    #clf = MLPClassifier(hidden_layer_sizes=(1200,),max_iter=400)
    #model=clf.fit(featureVecs, labels)

    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coefs_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coefs_[0])):
                print('   ', HHs.featureNames[j], '=',model.coefs_[0][j])
    return model

def rSquared(observed, predicted):
    error = ((predicted - observed)**2).sum()
    meanError = error/len(observed)
    return 1 - (meanError/np.var(observed))

def relevant_vectors(examples):
    vec_pf,vec_Name=[],[]
    vecs={}
    for feat in tern_sulfides.featureNames:
        vecs[feat]=[] 
    for e in examples:
        e_vecs=e.getFeatures()
        for j,feat in enumerate(tern_sulfides.featureNames):
            vecs[feat].append(e_vecs[j])
        vec_pf.append(e.getpf())
        vec_Name.append(e.getName())    
    return vecs,vec_pf,vec_Name

def get_predicted_pf(model,vecs):
    pred_pf=[]
    fname=tern_sulfides.featureNames
    #print(vecs[fname[0]])
    for i in range(len(vecs[fname[0]])):
        pred_pf.append(model.predict([[vecs[fname[0]][i],vecs[fname[1]][i],vecs[fname[2]][i],vecs[fname[3]][i],vecs[fname[4]][i],vecs[fname[5]][i],vecs[fname[6]][i],vecs[fname[7]][i],vecs[fname[8]][i],vecs[fname[9]][i],vecs[fname[10]][i],vecs[fname[11]][i],vecs[fname[12]][i],vecs[fname[13]][i],vecs[fname[14]][i],vecs[fname[15]][i],vecs[fname[16]][i],vecs[fname[17]][i],vecs[fname[18]][i],vecs[fname[19]][i],vecs[fname[20]][i],vecs[fname[21]][i],vecs[fname[22]][i],vecs[fname[23]][i],vecs[fname[24]][i],vecs[fname[25]][i],vecs[fname[26]][i],vecs[fname[27]][i],vecs[fname[28]][i]]])[0])
    return pred_pf

def give_featuresnlabels(examples, reg=False):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        if reg==False:
           labels.append(e.getLabel())
        else:  
           labels.append(e.getpf())   #for RandomForestRegressor
    return featureVecs, labels

def buildModel_rf(examples, toPrint = True,reg=False):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        if reg==False:
           labels.append(e.getLabel())
        else:  
           labels.append(e.getpf())   #for RandomForestRegressor
    #clf = sklearn.ensemble.RandomForestClassifier(max_features="log2",n_estimators=1000)
    if reg==False:
       clf = sklearn.ensemble.RandomForestClassifier(max_features="auto",n_estimators=2000,bootstrap=True,random_state=1,oob_score=True)
    else:
       #print("I am here")
       #clf = sklearn.ensemble.RandomForestRegressor(max_features="auto",n_estimators=2000,random_state=13,oob_score=True)
       clf = sklearn.ensemble.RandomForestRegressor(max_features=0.22,n_estimators=2000,random_state=13,max_depth=25,oob_score=True)
    mod=clf.fit(featureVecs, labels)
    importances = mod.feature_importances_
    std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)
    av = np.average([tree.feature_importances_ for tree in mod.estimators_],axis=0)
    #print("All tree.feature_importances_ Value i.e. Avg (Std):")
    #for i in range(len(std)):
    #    print(r"%d. %s: %f (%f)"%(i,HHs.featureNames[i],av[i],std[i]))


    #print("\n Actual Feature importances") 
    #indices = np.argsort(importances)[::-1]
    #for f in range(len(importances)):
    #    print ("{0}. {1} ({2})".format (f + 1, HHs.featureNames[indices[f]], importances[indices[f]]))

    if toPrint:
       std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)
       indices = np.argsort(importances)[::-1]
       print("Feature importances (model.feature_importances_) Std:%s:"%std)
       for f in range(len(importances)):
           print ("{0}. {1} ({2})".format (f + 1, tern_sulfides.featureNames[indices[f]], importances[indices[f]]))

    #rfe = RFE(estimator=classifier, n_features_to_select=15, step=1)
    #model = rfe.fit(featureVecs, labels)
    if toPrint:
       rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),scoring='accuracy')
       model=rfecv.fit(featureVecs, labels)
       if toPrint:
          print("Optimal number of features : %d" % model.n_features_)
          print("\n model.ranking_:")
          for i in range(len(model.ranking_)):
              if model.ranking_[i] == 1:
                 print (tern_sulfides.featureNames[i]) 
       return model,importances
    if toPrint==False:
       return mod,importances  #careful of model returned

def applyModel(model, testSet, label, prob = 0.5):
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    #print(testFeatureVecs) 
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

def lr(trainingData, testData, prob = 0.5):
    model = buildModel(trainingData, False)
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results

def neural_network(trainingData, testData, prob = 0.5):
    model = buildModel_nn(trainingData, False)
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results

def random_forest(trainingData, testData, prob = 0.5):
    model = buildModel_rf(trainingData, False)[0]
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results


##Delete this, rethink this part
def plotanalysis(model,zT0,gap,nn=0):
    var=np.linspace(-0.7,0.7,100)
    E_LW0=[]
    E_GW0=[]
    prob=[]

    for i in var:
        for j in var:
            E_LW0.append(i),E_GW0.append(j)
            test=np.array([i,j,gap])
            prob.append(model.predict_proba(test.reshape(1, -1))[0][1])
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(E_LW0, E_GW0, c=prob, vmin=0.0, vmax=1.0, s=35, cmap=cm,lw = 0)
    plt.colorbar(sc)
    plt.axvline(x=0.0,color='k',linewidth=0.7)
    plt.axhline(y=0.0,color='k',linewidth=0.7)
    plt.xlim(-0.7,0.7)
    plt.ylim(-0.7,0.7)
    plt.xlabel(r'$\Delta E_{\mathrm{LW}}$',color='k',fontsize=20)
    plt.ylabel(r'$\Delta E_{\mathrm{GW}}$',color='k',fontsize=20)
    plt.title("Prob. of zT>%s for gap=%s"%(zT0,gap))
    plt.tight_layout()
    if nn==0:
       plt.savefig("probplot_%s.pdf"%gap)
    else:
       plt.savefig("probplotnn_%s.pdf"%gap)
    plt.close()
    #plt.show()

def plot_correlations(vec_dftpf,predicted_pf,model):
    fig=plt.figure()
    ax=plt.subplot(1, 1, 1)
    rmse = sqrt(mean_squared_error(vec_dftpf, predicted_pf))
    mea=mean_absolute_error(vec_dftpf, predicted_pf)
    spearmanr=scipy.stats.spearmanr(vec_dftpf, predicted_pf)[0]
    pearsons=scipy.stats.pearsonr(vec_dftpf, predicted_pf)[0]
    #textstr = '$\mathrm{RMSE}=%.2f$ ($\mu W/mK^2$)\n$\mathrm{MAE}=%.2f$ ($\mu W/mK^2$)\n$\mathrm{Spearman}=%.2f$\n$\mathrm{Pearson}=%.2f$\n$\mathrm{T}=%.2f~K$\n$\mathrm{OOB score}=%.2f$\n'%(rmse,mea, spearmanr,pearsons,temp,model.oob_score_)
    textstr = '$\mathrm{RMSE}=%.2f$ ($\mu W/mK^2$)\n$\mathrm{MAE}=%.2f$ ($\mu W/mK^2$)\n$\mathrm{Spearman}=%.2f$\n$\mathrm{Pearson}=%.2f$\n$\mathrm{T}=%.2f~K$\n'%(rmse,mea, spearmanr,pearsons,temp)
    plt.scatter(vec_dftpf, predicted_pf, s=60,c='r',alpha=0.5,marker=r'o')
    plt.plot([0,np.amax(vec_dftpf)], [0,np.amax(vec_dftpf)],'k-',linewidth=1)
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top')
    plt.xlabel(r'PF$^{\mathrm{DFT}}$ ($\mu W/mK^2$)',color='k',fontsize=20)
    plt.ylabel(r'PF$^{\mathrm{ML}}$ ($\mu W/mK^2$)',color='k',fontsize=20)
    plt.xlim(-0.02,np.amax(vec_dftpf))
    plt.ylim(-0.02,np.amax(vec_dftpf))
    plt.tight_layout()
    plt.savefig("correlation.pdf")
    plt.close()
    return rmse,mea,spearmanr,pearsons

def data_pattern_imp(pfcut_min,pfcut_max,pts):
    interv=np.linspace(pfcut_min,pfcut_max,pts)
    pattern_imp={}
    for pf0 in interv:
        pattern_imp[pf0]={}
    for pf0 in interv:
        examples = buildsulfideExamples(temp,pf0,dope,pfmax)
        model_rf,importances_rf = buildModel_rf(examples, False)
        for j,feature in enumerate(tern_sulfides.featureNames):
            pattern_imp[pf0][feature]=importances_rf[j]
    return pattern_imp 

def fimportances_regural(pattern_imp,select_features,pfcut_min,pfcut_max,pts):
    interv=np.linspace(pfcut_min,pfcut_max,pts)
    fig=plt.figure()
    ax=plt.subplot(1, 1, 1)
    clist=["k-","b-","r-","c-","g-","m-","k--","b--","r--","c--","g--","m--","y-","y--","k.","g."]
    for j,feature in enumerate(select_features):
        fimp=[]
        for pf0 in interv:
            fimp.append(pattern_imp[pf0][feature])
        ax.plot(interv,fimp,clist[j],linewidth=2,label=feature)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.988, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        del fimp
    #plt.legend(loc='best')
    #plt.xlim(0.69,1.71)
    #plt.ylim(0.015,0.18)
    plt.xlabel(r'$PF_{\mathrm{cut-off}}$ ($\mu$ W/mK$^2$) ', color='k',fontsize=20)
    plt.ylabel(r'Feature Imps.', color='k',fontsize=20)
    plt.title("Random Forest with Bootstrapping")
    plt.savefig('graph.pdf')
    plt.close()


#Look at weights
numSplits = 10

pf0=2000
temp=300
dope=-2e+20
pfmax=2000



examples = buildsulfideExamples(temp,pf0,dope,pfmax)
model,rf_importances=buildModel_rf(examples,False,reg=True)
vecs,vec_dftpf,vec_Name=relevant_vectors(examples)
predicted_pf=get_predicted_pf(model,vecs)
rmse,mea,spearmanr,pearsons=plot_correlations(vec_dftpf,predicted_pf,model)
#

#rSquared(np.array(vec_dftpf), np.array(predicted_pf))
#0.93052953307202491

#Any other random forest algo possible, that does not overfit?

jkfkjj

rsq=[]
for f in range(numSplits):
    trainingSet, testSet = split80_20(examples)
    train_fvecs,train_labels=give_featuresnlabels(trainingSet, reg=True)
    test_fvecs,actual_test_labels=give_featuresnlabels(testSet, reg=True)
    model,rf_importances=buildModel_rf(trainingSet,False,reg=True)
    est_test_labels=get_predicted_pf(model,relevant_vectors(testSet)[0])
    rsq.append(rSquared(np.array(actual_test_labels), np.array(est_test_labels)))

mean = round(sum(rsq)/len(rsq), 4)
sd = round(np.std(rsq), 4)

print("CS score mean:%f std:%f"%(mean,sd))

kfkfjjj

from sklearn.cross_validation import cross_val_score
fvecs,labels=give_featuresnlabels(examples, reg=True)


print(np.mean(cross_val_score(model,fvecs,labels,cv=10)))
#param_grid = {'min_weight_fraction_leaf': [0.0,0.05,0.1,0.2,0.3,0.4,0.5],'max_leaf_nodes':[None,5,10,20,30,40,50,70,90,130],'bootstrap':[True,False]}

#from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import roc_auc_score

#grid_clf = GridSearchCV(model, param_grid, cv=10)
#grid_clf.fit(fvecs,labels)



pfcut_min=0.0
pfcut_max=4000.0
pts=100

#pattern_imp=data_pattern_imp(pfcut_min,pfcut_max,pts)
select_features=['gap', 'SGNr', 'av_infl_l','av_infl_sl','av_omax_l','av_omax_sl','av_Z','av_eig_l','av_eig_sl','av_kin_l','av_kin_sl','hi_Z','hi_kin_sl','lw_kin_l','lw_infl_l']
#fimportances_regural(pattern_imp,select_features,pfcut_min,pfcut_max,pts)   #only with RandomClassifier


def buildROC(trainingSet, testSet, title, plot = True,nn=0,rf=0):
    if nn==0 and rf==0: 
       model = buildModel(trainingSet, False)
    elif nn==1 and rf==0:
       model = buildModel_nn(trainingSet, False)
    else:
       model = buildModel_rf(trainingSet, False)[0] 

    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg =applyModel(model, testSet,1, p) #1 (>zT0)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.001
    auroc = sklearn.metrics.auc(xVals, yVals, True)
    if plot:
        pylab.plot(xVals, yVals)
        pylab.plot([0,1], [0,1])
        pylab.xlabel('1 - specificity')
        pylab.ylabel('Sensitivity')
        if nn==0 and rf==0:
           title = title + '\nAUROC, LR = ' + str(round(auroc,3))
           pylab.title(title)
           pylab.savefig("AUROC_lr.pdf")
        elif nn==1 and rf==0:
           title = title + '\nAUROC, NN = ' + str(round(auroc,3))
           pylab.title(title)
           pylab.savefig("AUROC_nn.pdf")  
        else: 
           title = title + '\nAUROC, RFE = ' + str(round(auroc,3))
           pylab.title(title)
           pylab.savefig("AUROC_rfe.pdf")  
    pylab.close()
    return auroc

#random.seed(0)
#examples = buildsulfideExamples(temp,pf0,dope,pfmax)
trainingSet, testSet = split80_20(examples)
#buildROC(trainingSet, testSet, 'ROC for Predicting zT>%s, 1 Split'%zT0)
#buildROC(trainingSet, testSet, 'ROC (nn) for Predicting zT>%s, 1 Split'%zT0,nn=1)
#buildROC(trainingSet, testSet, 'ROC (rf) for Predicting PF>%s $\mu W/mK^2$, 1 Split'%pf0,rf=1)

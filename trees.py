from math import log
import operator


#计算给定数据集的香农熵
#x(i)中包含的信息为l(xi)=-log2p(xi),p(xi)是该分类的概率
#熵：H=-∑p(xi)log2p(xi)
#用来度量数据的无序程度
def calShannonEnt(dataset):
    numEntries=len(dataset)
    #用于保存各个分类标签出现的次数
    labelCounts={}
    for featVec in dataset:
        currentLabel=featVec[-1]  #获得样本的分类标签
        labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries  #该类别出现的概率
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):     #数据集，特征下标，特征值，根据指定的特征值分类
    retDataSet=[]  #创建新的list对象，为了不改变原有数据集
    for featureVec in dataSet:
        if featureVec[axis]==value:
            reducedFeatureVec=featureVec[:axis]  #保存第0到axis-1个特征
            reducedFeatureVec.extend(featureVec[axis+1:])  #保存第axis+1到最后一个特征
            retDataSet.append(reducedFeatureVec)  #添加符合要求的样本到划分子集中
    return retDataSet

#根据信息增益选择最佳数据集分类方式
def chooseBestFeature(dataSet):
    numEntries=len(dataSet)
    numFeatures=len(dataSet[0])-1   #数据集特征个数
    baseEntropy=calShannonEnt(dataSet)  #计算划分之前的熵
    bestInfoGain=0.0   #信息增益
    bestFeature=-1       #用于划分数据集的最佳特征
    for i in range(numFeatures):
        featureValsList=[sample[i] for sample in dataSet]    #获取特征i的所有取值
        uniqueVals=set(featureValsList)  #去掉重复值
        newEntropy=0.0 #保存划分后的熵
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)  #根据特征i的值value划分数据集
            prob=len(subDataSet)/float(numEntries)
            newEntropy+=prob*calShannonEnt(subDataSet)  #累加计算根据特征i划分后的熵
        infoGain=baseEntropy-newEntropy  #信息增益
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i   #更新最佳特征的下标
    return bestFeature

def majorityCnt(classList):     #参数：数据集对应的类别列表
    classCount = {} # 类别数量统计
    for c in classList: # 遍历类别列表
        classCount[c] = classCount.get(c, 0) + 1 # 计数
    # 排序，sorted默认升序，所以要反转一下顺序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0] # 返回出现次数最多的类别

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  #所有类别
    if classList.count(classList[0])==len(classList):  #第一种终止条件：所有样本属于同一类别
        return classList
    if len(dataSet[0])==1:   #第二种终止条件：消耗完所有特征时，返回数据集中出现次数最多的类别标签
        return majorityCnt(classList)
    bestFeat=chooseBestFeature(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}  #以最佳特征作为根结点创建子树
    del(labels[bestFeat]) #在特征列表中删除最佳特征（创建结点需要消耗特征）
    featValues=[example[bestFeat] for example in dataSet]  #得到最佳特征的所有取值
    uniqueVals=set(featValues)   #去重
    for value in uniqueVals:
        subLabels=labels[:]  #深拷贝
        #递归创建决策树
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#使用决策树进行分类
def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]   #获得根结点名
    secondDict=inputTree[firstStr] #获得根结点对应特征的所有取值
    featIndex=featLabels.index(firstStr) #获得根结点所对应的特征的下标
    for key in secondDict.keys():  #遍历根结点特征的所有取值
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':  #如果分支节点是一个字典，说明不是叶子节点
                #进入决策树的下一层，参数：分支子树，特征名列表，测试样本
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
            break
    return classLabel

#使用pickle模块序列化对象，序列化对象可以在磁盘上保存对象，并在需要的时候读取出来，节省计算时间
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)


























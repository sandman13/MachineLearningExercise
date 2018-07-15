from numpy import *
def loadDataSet():
#词条切分后的文档集合，列表每一行代表一个文档
    postingList=[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['my','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    #由人工标注的每篇文档的类标签
    classVec=[0,1,0,1,0,1]   #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

#创建一个集合，保存数据集中出现的词条，保证唯一性
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

#根据词条列表中的词条是否在文档中出现（出现为1），将文档转为词条向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    #遍历文档中的每个词条
    for word in inputSet:
        #如果词条在词条列表中出现
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word :%s is not in my vocabulary!" %word)
    return returnVec

#s使用贝叶斯训练函数
def trainNBO(trainMatrix,trainCategory):
    #训练的样本数，矩阵的行数
    numTrainDocs=len(trainMatrix)
    #词条数，矩阵的列数
    numWords=len(trainMatrix[0])
    #所有文档中属于类别1的概率p(c=1),即侮辱性文档的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            #统计所有类别为1的词条向量中各个词条出现的次数
            p1Num+=trainMatrix[i]
            #统计类别为1的所有文档中出现的总词条数
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    #利用numpy数组计算p(wi|c1)
    #为防止溢出，取对数
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#使用贝叶斯进行分类
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNBO(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, "classified as :" ,classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry1 = ['stupid', 'garbage']
    # 同样转为词条向量，并转为NumPy数组的形式
    thisDoc1 = array(setOfWords2Vec(myVocabList, testEntry1))
    print(testEntry1, 'classified as:', classifyNB(thisDoc1, p0V, p1V, pAb))






















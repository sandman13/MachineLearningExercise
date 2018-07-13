from numpy import *
import operator
from os import listdir
def createDataSet():
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #numpy中的tile(A,B)函数是：重复A B次
    # 前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，
    # 后面的1保证重复完了是4行，而不是一行里有四个一样的），
    # 然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2  #乘方
    sqDistances=sqDiffMat.sum(axis=1) #axis=1表示求每行之和
    distances=sqDistances**0.5
    # argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]] #记录与目标点距离最近的第i个点的标签
        # get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，
        # 如果没有就返回0（后面写的）
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #key=operator.itemgetter(1)的意思是按照字典里的第一个(从0开始，0为key）排序
    return sortedClassCount[0][0]   #返回最多的标签

#读取数据
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()   #去除回车字符
        listFromLine=line.split('\t') #将得到的整行数据分隔成一个元素列表
        returnMat[index,:]=listFromLine[0:3] #选择前3个元素存入到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  #把最后一列的数据作为标签
        index+=1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #取出每一列的最小值
    minVals=dataSet.min(0)
    #取出每一列的最大值
    maxVals=dataSet.max(0)
    #每一个特征的取值范围
    ranges=maxVals-minVals
    #初始化待返回的归一化特征数据集
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    #tile()函数构造与原特征矩阵相同大小的矩阵，进行归一化计算：用当前值减去最小值，然后除以取值范围
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试
def datingClassTest():
    #选择10%的数据作为测试集
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('D:\\Python\\MachineLearning\\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    #选为测试集的样本数
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with :{0}, the real answer is :{1}".format(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]): errorCount+=1.0
    print("the total error rate is: {0}" .format(errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    #用户输入三个特征变量，并将输入的字符串类型转化为浮点型
    ffMiles = float(input("frequent flier miles earned per year:"))
    percentats = float(input("percentage of time spent playing video games:"))
    iceCream = float(input("liters of ice cream consumed per year:"))
    datingDataMat,datingLabels=file2matrix('D:\\Python\\MachineLearning\\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:{0}".format(resultList[classifierResult-1]))


#手写数字识别系统
#将32*32的图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        #读取一行返回字符串
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest() :
    hwLabels = []
    #获取目录内容
    trainingFileList = listdir('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch02\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m) :
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] #无后缀文件名
        classNumStr = int(fileStr.split('_')[0]) #获取文件内的数字
        hwLabels.append(classNumStr)
        #图片转换为向量
        trainingMat[i, :] = img2vector('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch02\\trainingDigits\\%s' %fileNameStr)
    testFileList = listdir('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch02\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch02\\testDigits\\%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) #分类
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr) :
            errorCount += 1.0
    print("\nthe total number of error is: %d" %errorCount)
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))
if __name__=='__main__':
    handwritingClassTest()











































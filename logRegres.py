from numpy import *
#逻辑回归梯度上升优化算法
#获得特征矩阵和标签矩阵
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        #添加x0,x1,x2的值，x0的值始终为1
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)  #转换为numpy矩阵,100*3
    labelMat=mat(classLabels).transpose() #转换为numpy矩阵，并且将行向量转置为列向量,100*1
    m,n=shape(dataMatrix)
    alpha=0.001  #向目标移动的步长
    maxCycles=500  #迭代次数
    weights=ones((n,1))   #回归系数,3*1
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)  #100*1,计算预测值
        error=(labelMat-h)   #计算预测值和真实值之间的误差
        weights=weights+alpha*dataMatrix.transpose()*error   #w:=w+alpha*▽f(w)，其中▽f(w)=dataMatrix.transpose()*error
    return weights

#画出数据集和逻辑回归最佳拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])  #把x1,x2赋给x轴，y轴
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]  #0是两个分类的分界处，因此设定w0x0+w1x1+w2x2=0，解出X2和X1的关系，画出直线
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix,labelMatrix):
    m,n =shape(dataMatrix)
    alpha = 0.1
    weight =ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix * weight))
        error = labelMatrix[i] - h
        weight = weight + alpha * error * dataMatrix[i]
    return weight

#改进的随机梯度上升算法
def stocGradAscent1(dataMat, labelMat, numIter=150):
        dataMatrix = mat(dataMat)  # translate list to matrix
        labelMatrix =mat(labelMat).transpose()  # 转置
        m, n =shape(dataMat)
        alpha = 0.1
        weight =ones(n)  # float
        # weight = np.random.rand(n)
        for j in range(numIter):
            dataIndex = list(range(m))  # range 没有del 这个函数　　所以转成list  del 见本函数倒数第二行
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))  # random.uniform(0,5) 生成0-5之间的随机数
                # 生成随机的样本来更新权重。
                h = sigmoid(sum(dataMat[randIndex] * weight))
                error = labelMat[randIndex] - h
                weight = weight + alpha * error * array(dataMat[randIndex])  # !!!!一定要转成array才行
                # dataMat[randIndex] 原来是list  list *2 是在原来的基础上长度变为原来2倍，
                del (dataIndex[randIndex])  # 从随机list中删除这个
        return weight

#从疝气病预测病马的死亡率
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain=open('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch05\\horseColicTraining.txt')
    frTest=open('D:\\Python\\MachineLearning\\machinelearninginaction\\Ch05\\horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))   #将读进来的每行的前21个str 转换为float
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))  #每一行第22列表示类别
    trainingWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is :%f" %errorRate)
    return errorRate

def multiTest():
    numTests=10; errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is : %f" %(numTests,errorSum/float(numTests)))








































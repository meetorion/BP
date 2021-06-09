import math

def read(filename):
    dataSet = []
    ans = []
    with open(filename, 'r+') as file:
        for line in file:
            tmp = list(line.split(' '))
            dataSet.append(tmp)
    for line in dataSet:
        val = line[-1][0]
        line = line[:-1]
        line.append(str(val))
        ans.append(line)
    return ans

#length表示权重向量的长度
def weigthRandom(length):
    ans = []
    for i in range(length):
        ans.append(1/length)
    return ans


def init( inputLayer, hiddenLayer, outputLayer):
    #length = len(dataSet[0]) - 1
    length = 2
    ans = []
    #初始化输入层，神经元个数为length,值为0；
    for i in range(length):
        inputLayer.append(0)
    #初始化隐层，隐层个数为1，每层神经元个数为length,值为[权重向量w,输出value,偏置b]；
    # hiddenLength = 1
    # for i in range(hiddenLength):
    #     layer = []
    #     for j in range(length):
    #         #w = weigthRandom(length)
    #         w = [.15, .20, .25, .30]
    #         value = 0
    #         b = .35
    #         layer.append([w, value, b])
    #     hiddenLayer.append(layer)
    hiddenLayer.append([[.15, .20], 0, .35])
    hiddenLayer.append([[.25, .30], 0, .35])
    #初始化输出层，神经元个数为2,值为[权重向量w,value, b]
    # for i in range(2):
    #     #w = weigthRandom(length)
    #     w = [.40, .45, .50, .55]
    #     value = 0
    #     b = .60
    #     outputLayer.append([w, value, b])
    outputLayer.append([[.40, .45], 0, .60])
    outputLayer.append([[.50, .55], 0, .60])

# sample为数据集的一个样本，inputLayer为输入层，hiddenLayer为隐层，outputLayer为输出层
def activate(sum):
    return 1/(1+math.exp(-sum))


def forward(sample, inputLayer, hiddenLayer, outputLayer):
    #将样本sample数据输入到输入层
    #attrMap = buildDict(dataSet)
    #从样本中提取输入
    input = sample[:-2]
    for i in range(len(input)): #-1是因为不统计标签
        inputLayer[i] = input[i]

    #前向传播,输入层传播到隐层
    layer = hiddenLayer
    #依次更新第1个隐层的神经元
    for j in range(len(layer)):
        w, value, b = layer[j]
        sum = 0
        #线性求和
        for k in range(len(inputLayer)):
            sum += inputLayer[k] * w[k]
        sum += b
        #激活函数非线性映射
        sum = activate(sum)
        #更新第j个神经元
        value = sum
        layer[j] = w, value, b

    # #前向传播,隐层之间传播
    # for i in range(1,len(hiddenLayer)):
    #     layer = hiddenLayer[i]
    #     pre = hiddenLayer[i-1]
    #     #依次更新layer的各个神经元的value值
    #     for j in range(len(layer)):
    #         w, value = layer[j]
    #         sum = 0
    #         for k in range(len(w)):
    #             sum += w[k] * pre[k][1]
    #         value = sum
    #         layer[j] = w, value
    #     #更新写回隐层
    #     hiddenLayer[i] = layer

    #前向传播，隐层到输出层
    for i in range(len(outputLayer)):
        w, value, b = outputLayer[i]
        sum = 0
        for j in range(len(w)):
            sum += w[j] * hiddenLayer[j][1]
        sum += b
        sum = activate(sum)
        value = sum
        outputLayer[i] = w, value, b

    #计算偏差
    #从样本中提取标签
    target = sample[-2:]
    ans = 0
    for i in range(len(target)):
        ans += 1/2*(target[i] - outputLayer[i][1])**2
    return ans

#dataSet是数据集，返回一个字典：键是属性，值是属性对应是数值
def buildDict(dataSet):
    ans = {}
    for j in range(len(dataSet[0]) - 1):
        cnt = 1
        for i in range(len(dataSet)):
            value = dataSet[i][j]
            if value in ans:
                continue
            else:
                ans[value] = cnt
                cnt += 1

    return ans


def backForward(sample, inputLayer, hiddenLayer, outputLayer, rate):
    target = sample[-2:]
    #更新隐层的权重w，注意输出层的每个神经元都有误差传播，有多条链
    for i in range(len(hiddenLayer)):
        w, value, _ = hiddenLayer[i]
        #依次更新隐层神经元各个权重
        for j in range(len(w)):
            update = 0 #传递到隐层的误差
            #每个权重误差来源与输出层所有神经元相关
            for k in range(len(outputLayer)):
                outputLayer_w,outputLayer_value, _ = outputLayer[k]
                output = outputLayer[k][1] #输出层第k个神经元的输出
                tmp = -(target[k] - output) #误差/输出层的偏导
                tmp *= output * (1 - output)  #输出层/隐层中sigmod函数的偏导
                tmp *= outputLayer_w[j] #输出层/隐层中线性求和的偏导，权重w
                update += tmp
            update *= value * (1 - value) #隐层/输入层中sigmod函数的偏导
            update *= inputLayer[j]  #隐层/输入层中线性求和的偏导
            w[j] -= rate * update

    # 更新输出层的权重w，一个权重只是一条链
    for i in range(len(outputLayer)):
        w, output, _ = outputLayer[i]
        for j in range(len(w)):
            update = -(target[i] -  output)
            update *= output * (1 - output)
            update *= hiddenLayer[j][1] #第j个神经元的输出值
            w[j] -= rate * update

if __name__ == '__main__':
    # 输入层
    inputLayer = []
    # 隐层
    hiddenLayer = []
    # 输出层
    outputLayer = []
    # dataSet = read('mytrain.data')
    # 初始化神经网络结构
    init(inputLayer, hiddenLayer, outputLayer)
    # 建立属性值与数值的映射
    #attrMap = buildDict(dataSet)
    # 前向传播
    test = [.05, .10, .01, .99]
    print("first bias: ", forward(sample=test, inputLayer=inputLayer, hiddenLayer=hiddenLayer, outputLayer=outputLayer))
    for i in range(10000):
        bias = forward(sample=test, inputLayer=inputLayer, hiddenLayer=hiddenLayer, outputLayer=outputLayer)
    # 反向传播更新权重向量w
        backForward(test, inputLayer, hiddenLayer, outputLayer, rate=0.5)
    print("final bias:" ,forward(sample=test, inputLayer=inputLayer, hiddenLayer=hiddenLayer, outputLayer=outputLayer))




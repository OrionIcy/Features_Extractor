import Operation
from Operation import Operation
from Operation import CNN
from Operation import Network

class COUNTING(object):
    def __init__(self):
        self.num_Conv = 0
        self.num_Max = 0
        self.num_Avg = 0
        self.num_Mat = 0
        self.num_Bias = 0
        self.num_LRN = 0
        self.num_Soft = 0
    def newConv(self):
        self.num_Conv += 1
    def newMax(self):
        self.num_Max += 1
    def newAvg(self):
        self.num_Avg += 1
    def newMat(self):
        self.num_Mat += 1
    def newBias(self):
        self.num_Bias += 1
    def newLRN(self):
        self.num_LRN += 1
    def newSoft(self):
        self.num_Soft += 1
    def getConv(self):
        return self.num_Conv
    def getMax(self):
        return self.num_Max
    def getAvg(self):
        return self.num_Avg
    def getMat(self):
        return self.num_Mat
    def getBias(self):
        return self.num_Bias
    def getLRN(self):
        return self.num_LRN
    def getSoft(self):
        return self.num_Soft
    def getTotal(self):
        return self.num_Avg + self.num_Bias + self.num_Conv + self.num_LRN + self.num_Mat + self.num_Max + self.num_Soft

NO = COUNTING()

def CountOperation():
    with open("OperationList.txt") as ff:
        Lines = ff.readlines()
        for Line in Lines:
            Line = Line.strip()
            if Line == "Conv2d":
                NO.newConv()
            elif Line == "MaxPool":
                NO.newMax()
            elif Line == "AvgPool":
                NO.newAvg()
            elif Line == "MatMul":
                NO.newMat()
            elif Line == "BiasAdd":
                NO.newBias()
            elif Line == "LRN":
                NO.newLRN()
            elif Line == "Softmax":
                NO.newSoft()
    return

def InitOperationList():
    CountOperation()
    OperationList = [Operation() for i in range(1,NO.getTotal()+1)]
    return OperationList

def ReadOperationFile(OperationList):
    with open("OperationList.txt") as f:
        i = 0
        while i < NO.getTotal():
            OPName = f.readline()
            OPName = OPName.strip()
            if OPName == "CNN":
                batch_size = int(f.readline())
                num_classes = int(f.readline())
            elif OPName == "Conv2d":
                in_h = int(f.readline())
                in_w = int(f.readline())
                f_h = int(f.readline())
                f_w = int(f.readline())
                in_c = int(f.readline())
                out_c = int(f.readline())
                stri = int(f.readline())
                bat_s = int(f.readline())
                OperationList[i].SetConv(in_h,in_w,f_h,f_w,in_c,out_c,stri,bat_s)#计算
                Network.app(OperationList[i])
            elif OPName == "MaxPool":
                in_h = int(f.readline())
                in_w = int(f.readline())
                ksize = int(f.readline())
                strides = int(f.readline())
                OperationList[i].SetMaxPool(in_h,in_w,ksize,strides)
                Network.app(OperationList[i])
            elif OPName == "AvgPool":
                in_h = int(f.readline())
                in_w = int(f.readline())
                ksize = int(f.readline())
                strides = int(f.readline())
                OperationList[i].SetAvgPool(in_h,in_w,ksize,strides)
                Network.app(OperationList[i])
            elif OPName == "MatMul":
                M1x = int(f.readline())
                M1y = int(f.readline())
                M2x = int(f.readline())
                M2y = int(f.readline())
                OperationList[i].SetMatMul(M1x,M1y,M2x,M2y)
                Network.app(OperationList[i])
            elif OPName == "BiasAdd":
                in_sy = int(f.readline())
                in_sx = int(f.readline())
                OperationList[i].SetBiasAdd(in_sx,in_sy)
                Network.app(OperationList[i])
            elif OPName == "LRN":
                depth = float(f.readline())
                beta = float(f.readline())
                dim1 = float(f.readline())
                dim2 = float(f.readline())
                dim3 = float(f.readline())
                dim4 = float(f.readline())
                OperationList[i].SetLRN(depth,beta,dim1,dim2,dim3,dim4)
                Network.app(OperationList[i])
            elif OPName == "Softmax":
                sftmxi = float(f.readline())
                sftmxj = float(f.readline())
                #print([sftmxi,sftmxj])
                OperationList[i].SetSoftmax(sftmxi,sftmxj)
                Network.app(OperationList[i])
            i += 1
    return
import math

class Conv2d(object):
    def __init__(self):
        self.OPName = ""
        self.in_channels = 0
        self.out_channels = 0
        self.f_height = 0
        self.f_width = 0
        self.in_height = 0
        self.in_width = 0
        self.strides = 1
        self.num_add = 0
        self.num_mul = 0
        self.batch_size = 0
    def getname(self):
        return self.OPName
    def SetConv(self,in_h,in_w,f_h,f_w,in_c,out_c,str,bat_s):
        self.OPName = "Conv2d"
        self.in_height = in_h
        self.in_width = in_w
        self.f_height = f_h
        self.f_width = f_w
        self.in_channels = in_c
        self.out_channels = out_c
        self.strides = str
        self.batch_size = bat_s
        self.num_add = (self.f_height * self.f_width * self.in_channels - 1) * (math.ceil((self.in_width - self.f_width) / self.strides) + 1) * (math.ceil((self.in_height - self.f_height)/self.strides) + 1) * self.out_channels * self.batch_size
        self.num_mul = (self.f_height * self.f_width * self.in_channels) * (math.ceil((self.in_width - self.f_width) / self.strides) + 1) * (math.ceil((self.in_height - self.f_height)/self.strides) + 1) * self.out_channels * self.batch_size
    def GetSum(self):
        return self.num_add,self.num_mul

class MaxPool(object):
    def __init__(self):
        self.OPName = ""
        self.in_height = 0
        self.in_width = 0
        self.ksize = 0
        self.strides = 0
        self.out_height = 0
        self.out_width = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetMaxPool(self,in_h,in_w,ksize,strides):
        self.OPName = "MaxPool"
        self.in_height = in_h
        self.in_width = in_w
        self.ksize = ksize
        self.strides = strides
        self.out_height = math.ceil((self.in_height - self.ksize) / self.strides) + 1
        self.out_width = math.ceil((self.in_width - self.ksize) / self.strides) + 1
        self.num_add = (self.ksize * self.ksize - 1) * (self.out_height * self.out_width)
    def GetSum(self):
        return self.num_add,self.num_mul

class AvgPool(object):
    def __init__(self):
        self.OPName = ""
        self.in_height = 0
        self.in_width = 0
        self.ksize = 0
        self.strides = 0
        self.out_height = 0
        self.out_width = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetAvgPool(self,in_h,in_w,ksize,strides):
        self.OPName = "AvgPool"
        self.in_height = in_h
        self.in_width = in_w
        self.ksize = ksize
        self.strides = strides
        self.out_height = math.ceil((self.in_height - self.ksize) / self.strides) + 1
        self.out_width = math.ceil((self.in_width - self.ksize) / self.strides) + 1
        self.num_add = (self.ksize * self.ksize - 1) * (self.out_height * self.out_width)
        self.num_mul = self.out_height * self.out_width
    def GetSum(self):
        return self.num_add,self.num_mul

class MatMul(object):
    def __init__(self):
        self.OPName = ""
        self.Mat1x = 0
        self.Mat1y = 0
        self.Mat2x = 0
        self.Mat2y = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetMatMul(self,M1x,M1y,M2x,M2y):
        self.OPName = "MatMul"
        self.Mat1x = M1x
        self.Mat1y = M1y
        self.Mat2x = M2x
        self.Mat2y = M2y
        if self.Mat1y == self.Mat2x:
            self.num_add = self.Mat1x * self.Mat1y * self.Mat2y
            self.num_mul = self.Mat1x * self.Mat2y * (self.Mat1y - 1)
    def GetSum(self):
        return self.num_add,self.num_mul

class BiasAdd(object):
    def __inti__(self):
        self.OPName = ""
        self.insizex = 0
        self.insizey = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetBiasAdd(self,in_sx,in_sy):
        self.OPName = "BiasAdd"
        self.insizex = in_sx
        self.insizey = in_sy 
        self.num_add = self.insizex * self.insizey
    def GetSum(self):
        return self.num_add,self.num_mul

class LRN(object):
    def __init__(self):
        self.OPName = ""
        self.beta = 0
        self.depth_radius = 0
        self.dim1 = 0
        self.dim2 = 0
        self.dim3 = 0
        self.dim4 = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetLRN(self,depth_radius,beta,dim1,dim2,dim3,dim4):
        self.OPName = "LRN"
        self.depth_radius = depth_radius
        self.beta = beta
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.num_add = (2 * self.depth_radius + 1) * dim1 * dim2 * dim3 * dim4
        self.num_mul = (2 * self.depth_radius + self.beta + 2) * dim1 * dim2 * dim3 * dim4
    def GetSum(self):
        return self.num_add,self.num_mul

class Softmax(object):
    def __init__(self):
        self.OPName = ""
        self.batch_size = 0
        self.num_classes = 0
        self.num_add = 0
        self.num_mul = 0
    def getname(self):
        return self.OPName
    def SetSoftmax(self,batch_size,num_classes):
        self.OPName = "Softmax"
        self.batch_size=batch_size
        self.num_classes=num_classes
        #print([batch_size,num_classes])
        self.num_add = num_classes * (num_classes - 1) * batch_size
        self.num_mul = (batch_size * num_classes) * (num_classes * (num_classes * batch_size - 1) + 1)
    def GetSum(self):
        return self.num_add,self.num_mul

class Operation(Conv2d,MaxPool,AvgPool,MatMul,BiasAdd,LRN,Softmax):
    def __init__(self):
        super(Operation,self).__init__()
        self.num_add,self.num_mul = super().GetSum()
        self.OPName = super().getname()
    def GetAdd(self):
        return self.num_add
    def GetMul(self):
        return self.num_mul
    def GetName(self):
        return self.getname()

class OPSet(object):
    def __init__(self):
        self.List = [Operation()]
        self.add_sum = 0
        self.mul_sum = 0
        self.ListLen = 0
    def app(self,A):
        self.List.append(A)
        self.ListLen += 1
    def Sum(self):
        for i in range(1,self.ListLen + 1):
            self.add_sum = self.add_sum + self.List[i].GetAdd()
            self.mul_sum = self.mul_sum + self.List[i].GetMul()
    def RetSum(self):
        return self.add_sum,self.mul_sum
    def SumLayer(self):
        L1 = []
        for i in range(1,self.ListLen + 1):
            L1.append([self.List[i].GetName(),self.List[i].GetAdd(),self.List[i].GetMul()])
        return L1

class CNN(OPSet):
    def __init__(self):
        super(CNN,self).__init__()
        self.num_classes = 0
        self.total_add = 0
        self.total_mul = 0
        self.EveryLayer = []
        self.EveryClassOfLayer = []
    def set_Model(self,num_classes):
        self.num_classes = num_classes
    def app(self,A):
        super().app(A)
    def GetTotal(self):
        super().Sum()
        self.total_add ,self.total_mul = super().RetSum()
        return [self.total_add,self.total_mul]
    def GetEveryLayer(self):
        self.EveryLayer = super().SumLayer()
        return self.EveryLayer
    def GetEveryClassOfLayer(self):
        SConv = [0,0]
        SAvgPool = [0,0]
        SMaxPool = [0,0]
        SBias = [0,0]
        SMat = [0,0]
        SLRN = [0,0]
        SSftmx = [0,0]
        Layer = []
        Layer = self.GetEveryLayer()
        for i in range(len(Layer)):
            if Layer[i][0] == "Conv2d":
                SConv[0]+=int(Layer[i][1])
                SConv[1]+=int(Layer[i][2])
            elif Layer[i][0] == "AvgPool":
                SAvgPool[0]+=int(Layer[i][1])
                SAvgPool[1]+=int(Layer[i][2])
            elif Layer[i][0] == "MaxPool":
                SMaxPool[0]+=int(Layer[i][1])
                SMaxPool[1]+=int(Layer[i][2])
            elif Layer[i][0] == "BiasAdd":
                SBias[0]+=int(Layer[i][1])
                SBias[1]+=int(Layer[i][2])
            elif Layer[i][0] == "MatMul":
                SMat[0]+=int(Layer[i][1])
                SMat[1]+=int(Layer[i][2])
            elif Layer[i][0] == "LRN":
                SLRN[0]+=int(Layer[i][1])
                SLRN[1]+=int(Layer[i][2])
            elif Layer[i][0] == "Softmax":
                SSftmx[0]+=int(Layer[i][1])
                SSftmx[1]+=int(Layer[i][2])
        SConv.insert(0,"Conv2D")
        SAvgPool.insert(0,"AvgPool")
        SMaxPool.insert(0,"MaxPool")
        SBias.insert(0,"BiasAdd")
        SMat.insert(0,"MatMul")
        SLRN.insert(0,"LRN")
        SSftmx.insert(0,"Softmax")
        self.EveryClassOfLayer.append(SConv)
        self.EveryClassOfLayer.append(SAvgPool)
        self.EveryClassOfLayer.append(SMaxPool)
        self.EveryClassOfLayer.append(SBias)
        self.EveryClassOfLayer.append(SMat)
        self.EveryClassOfLayer.append(SLRN)
        self.EveryClassOfLayer.append(SSftmx)
        return self.EveryClassOfLayer

    #Print 仅供调试使用
    def PrintTotal(self):
        print(self.GetTotal())
    def PrintEveryLayer(self):
        for i in range(0,len(self.GetEveryLayer())):
            print(self.GetEveryLayer()[i])
    def PrintEveryClassOfLayer(self):
        for i in range(0,len(self.GetEveryClassOfLayer())):
            print(self.GetEveryClassOfLayer()[i])

Network = CNN()
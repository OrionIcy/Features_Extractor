import math
#import ReadFileModule as RFM
import copy

MatMulList = []

class MatTree(object):
    def __init__(self,data = None,lchild = None,rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild
    def Lappend(self,LNode):
        self.lchild = LNode
    def Rappend(self,RNode):
        self.rchild = RNode
    def Traverse(self):
        if self.lchild != None:
            if not isinstance(self.lchild,str):
                self.lchild.Traverse()
        if self.rchild != None:
            if not isinstance(self.rchild,str):
                self.rchild.Traverse()
        if self.lchild == None and self.rchild == None:
            MatMulList.append(self.data)
            #print(self.data)
            
        return
    def Destroy(self):
        if self.lchild != None:
            if not isinstance(self.lchild,str):
                self.lchild.Destroy()
        if self.rchild != None:
            if not isinstance(self.rchild,str):
                self.rchild.Destroy()
        if self.lchild == None and self.rchild == None:
            self = None
        return

def FindPos(List,Target,start,end):
    RetList = []
    for i in List[start:end]:
        i = i.strip()
        i = i.replace(' ','')
        if i.startswith(Target):
            Begin = i.find('[')
            Finish = i.find(']')
            Pos = i.find(',',Begin,Finish)
            if Pos == -1:
                return
            RetList.extend([i[Begin + 1:Pos],i[Pos + 1:Finish]])
            break
    return RetList
        
def Append(TreeNode,List,start,end):
    Pos = FindPos(List,TreeNode.data,start,end)
    if Pos:
        P = MatTree(Pos[0])
        TreeNode.Lappend(P)
        Append(P,List,start,end)
        P = MatTree(Pos[1])
        TreeNode.Rappend(P)
        Append(P,List,start,end)
    else:
        return
#以上为获得矩阵乘法参数的二叉树


def TextAnalyser(filename):

    KernelOpList = []
    #存放定义kernel的语句

    KernelList = []
    #存放kernel的尺寸，每个元素是四元组


    #存入List，第一个元素为操作名，第二个元素为参数List
    #Conv2d:Input,Kernel,Strides,batch_size
    #MaxPool:
    #AvgPool:Input,ksize,Strides
    #MatMul:Mat1,Mat2
    #LRN:input,depth_radius,beta
    #Softmax
    ArgsList = []
    batch_size = 0
    NextShape = [0,0,0,0]
    #存放全部操作名和操作所需参数
    StrideList = []
    #存放步长
    ksizeList = []
    #存放ksize
    LrnList = []
    #存放LRN参数
    MatList = []
    #存放MatMul参数
    NextFc = []

    InputList = []

    OutputList = []

    Lines = []

    with open(filename,encoding = 'utf-8-sig') as f:
        Lines = f.readlines()
        
        OpSeq = []

        OpArgs = []

        Input = ""

        for i in range(0, len(Lines)):
            if "tf.nn.conv2d" in Lines[i]:
                #卷积
                Strides = 0
                OpSeq.append("Conv2d_" + (str)(i))
                ArgsPos = Lines[i].find('(')
                ArgsConv = Lines[i][ArgsPos + 1: len(Lines[i]) - 2]
                ArgsConv = ArgsConv.strip()
                ArgsConv = ArgsConv.replace(' ','')
                Begin = 0
                Pos = 0
                while ',' in ArgsConv[Begin:len(ArgsConv)]:
                    Pos = ArgsConv.find(',',Begin)
                    if '[' in ArgsConv[Begin:Pos]:
                        Pos = ArgsConv.find(']')
                        OpArgs.append(ArgsConv[Begin:Pos + 1])
                        Begin = Pos + 2
                    else:
                        OpArgs.append(ArgsConv[Begin:Pos])
                        Begin = Pos + 1
                #获得第一层参数，获得参数
                if "stride" in OpArgs[2] or '[' in OpArgs[2]:
                    Pos = OpArgs[2].find('[')
                    OpArgs[2] = OpArgs[2][Pos + 1: len(OpArgs[2]) - 1]
                    Pos = OpArgs[2].find(',')
                    Strides = OpArgs[2][Pos + 1]
                    OpArgs[2] = Strides
                    StrideList.append((int)(Strides))
                temp = OpArgs.copy()
                ArgsList.append(["Conv2d",temp])
                OpArgs.clear()
                #字典保存参数，遍历字典值判断是否获得所有参数
                
            elif "tf.nn.max_pool" in Lines[i]:
                #最大池化
                Strides = 0
                OpSeq.append("MaxPool_" + (str)(i))
                ArgsPos = Lines[i].find('(')
                ArgsMP = Lines[i][ArgsPos + 1: len(Lines[i]) - 2]
                ArgsMP = ArgsMP.strip()
                ArgsMP = ArgsMP.replace(' ','')
                Begin = 0
                Pos = 0
                while ',' in ArgsMP[Begin:len(ArgsMP)]:
                    Pos = ArgsMP.find(',',Begin)
                    if '[' in ArgsMP[Begin:Pos]:
                        Pos = ArgsMP.find(']',Begin) + 1
                        OpArgs.append(ArgsMP[Begin:Pos])
                        Begin = Pos + 1
                    else:
                        OpArgs.append(ArgsMP[Begin:Pos])
                        Begin = Pos + 1
                if "stride" in OpArgs[2]:
                    Pos = OpArgs[2].find('[')
                    OpArgs[2] = OpArgs[2][Pos + 1: len(OpArgs[2]) - 1]
                    Pos = OpArgs[2].find(',')
                    Strides = OpArgs[2][Pos + 1]
                    OpArgs[2] = Strides
                    StrideList.append((int)(Strides))
      
                if "=" in OpArgs[1]:
                    Pos = OpArgs[1].find('=')
                    Pos = OpArgs[1].find('[')
                    OpArgs[1] = OpArgs[1][Pos + 1: len(OpArgs[1]) - 1]
                    Pos = OpArgs[1].find(',')
                    ksize = OpArgs[1][Pos + 1]
                    OpArgs[1] = ksize
                    ksizeList.append((int)(ksize))
                OpArgs.pop(3)
                temp = OpArgs.copy()
                ArgsList.append(["MaxPool",temp])
                OpArgs.clear()
            elif "tf.nn.avg_pool" in Lines[i]:
                #平均池化
                Strides = 0
                OpSeq.append("AvgPool_" + (str)(i))
                ArgsPos = Lines[i].find('(')
                ArgsAP = Lines[i][ArgsPos + 1: len(Lines[i]) - 2]
                ArgsAP = ArgsAP.strip()
                ArgsAP = ArgsAP.replace(' ','')
                Begin = 0
                Pos = 0
                while ',' in ArgsAP[Begin:len(ArgsAP)]:
                    Pos = ArgsAP.find(',',Begin)
                    if '[' in ArgsAP[Begin:Pos]:
                        Pos = ArgsAP.find(']',Begin) + 1
                        OpArgs.append(ArgsAP[Begin:Pos])
                        Begin = Pos + 1
                    else:
                        OpArgs.append(ArgsAP[Begin:Pos])
                        Begin = Pos + 1
                if "stride" in OpArgs[2]:
                    Pos = OpArgs[2].find('[')
                    OpArgs[2] = OpArgs[2][Pos + 1: len(OpArgs[2]) - 1]
                    Pos = OpArgs[2].find(',')
                    Strides = OpArgs[2][Pos + 1]
                    OpArgs[2] = Strides
                    StrideList.append((int)(Strides))
                if "=" in OpArgs[1]:
                    Pos = OpArgs[1].find('=')
                    OpArgs[1] = OpArgs[1][Pos + 1]
                temp = OpArgs.copy()
                ArgsList.append(["AvgPool",temp])
                OpArgs.clear()
            elif "tf.matmul" in Lines[i]:
                OpSeq.append("MatMul_" + (str)(i))
                ArgsPos = Lines[i].find('(')
                ArgsMM = Lines[i][ArgsPos + 1]
                ArgsMM = Lines[i][ArgsPos + 1: len(Lines[i]) - 2]
                ArgsMM = ArgsMM.strip()
                ArgsMM = ArgsMM.replace(' ','')
                Begin = 0
                Pos = 0
                while ',' in ArgsMM[Begin:len(ArgsMM)]:
                    Pos = ArgsMM.find(',',Begin)
                    MatList.append(ArgsMM[Begin:Pos])
                    Begin = Pos + 1
                MatList.append(ArgsMM[Begin:len(ArgsMM)])
                #print(MatList)
                #关于矩阵乘法参数获取，建立参数二叉树

                temp = MatList.copy()
                #print(MatList)
                ArgsList.append(["MatMul",temp])
                MatList.clear()
            elif "tf.nn.lrn" in Lines[i] or "tf.nn.local_response_normalization" in Lines[i]:
                OpSeq.append("LRN" + (str)(i))
                ArgsPos = Lines[i].find('(')
                ArgsLRN = Lines[i][ArgsPos + 1]
                ArgsLRN = Lines[i][ArgsPos + 1: len(Lines[i]) - 2]
                ArgsLRN = ArgsLRN.strip()
                ArgsLRN = ArgsLRN.replace(' ','')
                Begin = 0
                Pos = 0
                while ',' in ArgsLRN[Begin:len(ArgsLRN)]:
                    Pos = ArgsLRN.find(',',Begin)
                    OpArgs.append(ArgsLRN[Begin:Pos])
                    Begin = Pos + 1
                OpArgs.append(ArgsLRN[Begin:len(ArgsLRN)])
                if '=' in OpArgs[2]:
                    Pos = OpArgs[2].find('=')
                    OpArgs[2] = OpArgs[2][Pos + 1:len(OpArgs[2])]
                if '=' in OpArgs[4]:
                    Pos = OpArgs[4].find('=')
                    OpArgs[4] = OpArgs[4][Pos + 1:len(OpArgs[4])]
                temp = []
                temp.append(OpArgs[0])
                temp.append(OpArgs[1])
                temp.append(OpArgs[4])
                LrnList.append([(float)(OpArgs[1]),(float)(OpArgs[4])])
                ArgsList.append(["LRN",temp])
                OpArgs.clear()
                
            elif "tf.nn.softmax" in Lines[i]:
                ArgsList.append(["Softmax",[]])
                #print("Softmax")
        #第一次循环，提取直接赋值的参数
        #矩阵乘法待修改
        ParaToBeFound = []
        for i in range(0, len(ArgsList)):
            if i == 0:
                for j in range(0, len(ArgsList[i][1])):
                    if ArgsList[i][1][j] < "0" or ArgsList[i][1][j] > "9":
                        ParaToBeFound.append(ArgsList[i][1][j])
            else:
                for j in range(1, len(ArgsList[i][1])):
                    if ArgsList[i][1][j] < "0" or ArgsList[i][1][j] > "9":
                        ParaToBeFound.append(ArgsList[i][1][j])
        #print(ParaToBeFound)
        #得到待查找参数列表
        #其中第一个参数(image,也就是输入的尺寸)单独查找
        for i in range(0, len(Lines)):
            Lines[i] = Lines[i].replace(' ','')
            if Lines[i].startswith(ParaToBeFound[0]):
                #print(i)
                PosLeft = Lines[i].find('[')
                PosRight = Lines[i].find(']')
                input_size = Lines[i][PosLeft + 1:PosRight]
                Begin = 0
        #print(input_size)
                for j in range(0, 3):
                    Pos = input_size.find(',',Begin)
                    if input_size[Begin:Pos].isdigit():
                        InputList.append((int)(input_size[Begin:Pos]))
                    else:
                        for k in range(0,len(Lines)):
                            Lines[k] = Lines[k].replace(' ','')
                            if Lines[k].startswith(input_size[Begin:Pos]):
                                PosL=Lines[k].find("=")
                                InputList.append(int(Lines[k][PosL+1:]))
                                break
                    Begin = Pos + 1
                #InputList.append((int)(input_size[Begin:len(input_size)]))
                InputList.append((int)(input_size[Begin:len(input_size)]))

                break
    #print(InputList)
    batch_size,in_Height,in_Width,in_channel = InputList[0],InputList[1],InputList[2],InputList[3]
    #获得第一个输入的尺寸[batch_size, in_height, in_width, in_channel]
    BeginPtr = 0
    EndPtr = 0
    j = 1
    for i in range(0, len(Lines)):
        if Lines[i].strip().startswith(ParaToBeFound[j]):
            KernelOpList.append(Lines[i].strip())
    #获得卷积核的定义语句
    Kernel = []
    for i in range(0, len(KernelOpList)):
        PosLeft = KernelOpList[i].find('[')
        PosRight = KernelOpList[i].find(']')
        Begin = PosLeft + 1
        for j in range(0, 3):
            Pos = KernelOpList[i].find(',',Begin)
            Kernel.append((int)(KernelOpList[i][Begin:Pos]))
            Begin = Pos + 1
        Kernel.append((int)(KernelOpList[i][Begin:PosRight]))
        temp = Kernel.copy()
        KernelList.append(temp)
        Kernel.clear()
    #至此，提取出kernel全部参数
    
    
    
    #下一步将确定输出到文件的参数和格式
    NextShape = [in_Height,in_Width,in_channel,batch_size]
    AftFc = []
    k = 0
    s = 0
    ks = 0
    ll = 0
    for i in range(0, len(ArgsList)):
        if ArgsList[i][0] == "Conv2d":
            OutputList.extend(["Conv2d\n",(str)(NextShape[0]) + "\n",(str)(NextShape[1]) + "\n"])
            for j in range(0, 4):
                OutputList.append((str)(KernelList[k][j]) + "\n")
            OutputList.append((str)(StrideList[s]) + "\n")
            OutputList.append((str)(batch_size) + "\n")
            batch_size = batch_size
            in_Height = math.ceil(in_Height/StrideList[s])
            in_Width = math.ceil(in_Width/StrideList[s])
            in_channel = KernelList[k][3]
            NextShape = [in_Height,in_Width,in_channel,batch_size]
            OutputList.extend(["BiasAdd\n",(str)(in_Height) + "\n",str(in_Width)+"\n"])
            k += 1
            s += 1
        elif ArgsList[i][0] == "MaxPool":
            OutputList.extend(["MaxPool\n",(str)(NextShape[0]) + "\n",(str)(NextShape[1]) + "\n"])
            OutputList.append((str)(ksizeList[ks]) + "\n")
            OutputList.append((str)(StrideList[s]) + "\n")
            in_Height = (in_Height - ksizeList[ks])/StrideList[s]
            in_Width = (in_Width - ksizeList[ks])/StrideList[s]
            if (in_Height - ksizeList[ks])%StrideList[s] == 0:
                in_Width += 1
                in_Height += 1
            else:
                in_Height = math.ceil(in_Height)
                in_Width = math.ceil(in_Width)
            NextShape[0] = int(in_Height)
            NextShape[1] = int(in_Width)
            s += 1
            ks += 1
        elif ArgsList[i][0] == "AvgPool":
            OutputList.extend(["AvgPool\n",(str)(NextShape[0]) + "\n",(str)(NextShape[1]) + "\n"])
            OutputList.append((str)(ksizeList[ks]))
            OutputList.append((str)(StrideList[s]) + "\n")
            in_Height = math.ceil((in_Height - ksizeList[ks])/StrideList[s]) + 1
            in_Width = math.ceil((in_Width - ksizeList[ks])/StrideList[s]) + 1
            NextShape[0] = in_Height
            NextShape[1] = in_Width
            s += 1
            ks += 1
        elif ArgsList[i][0] == "LRN":
            OutputList.extend(["LRN\n",(str)(LrnList[ll][0]) + "\n",(str)(LrnList[ll][1]) + "\n"])
            OutputList.extend([(str)(in_Height) + "\n",(str)(in_Width) + "\n",(str)(in_channel) + "\n",(str)(batch_size) + "\n"])
        elif ArgsList[i][0] == "MatMul":
            OutputList.extend(["MatMul\n"])
            #第一层
            #print(MatMulList)
            ArgsTree = MatTree()
            Args0 = MatTree(ArgsList[i][1][0])
            Args1 = MatTree(ArgsList[i][1][1])
            ArgsTree.Lappend(Args0)
            ArgsTree.Rappend(Args1)
            Append(Args0,Lines,0,len(Lines))
            Append(Args1,Lines,0,len(Lines))
            ArgsTree.Traverse()
                #左节点为第一个参数，右节点第二个参数
                #分别找到参数的赋值语句，获得参数的shape
                #参数语句中的值建立左右子节点，直到获得全部参数
                #左到右遍历二叉树的叶子节点，得到参数列表
                #仅适用于第一次矩阵乘法
            if NextFc == []:
                #print(MatMulList)
                for j in range(0, len(MatMulList)):
                    if MatMulList[j] == '-1':
                        MatMulList[j] = (str)(NextShape[3])
                        continue
                    if MatMulList[j][0] < '0' or MatMulList[j][0] > '9':
                        MatMulList[j] = (str)(NextShape[0] * NextShape[1] * NextShape[2])
                    NextFc.extend([NextShape[3],MatMulList[3]])
            else:
                MatMulList.pop(0)
                MatMulList.insert(0, NextFc[1])
                MatMulList.insert(0, NextFc[0])
            #print(MatMulList)
            OutputList.extend([(str)(MatMulList[0])+"\n", (str)(MatMulList[1])+"\n",(str)(MatMulList[2])+"\n"])
            OutputList.append((str)(MatMulList[3]) + "\n")
            OutputList.extend(["BiasAdd\n",(str)(MatMulList[0])+"\n",(str)(MatMulList[3]) + "\n"])
            MatMulList.clear()
        #SoftMax
        elif ArgsList[i][0] == "Softmax":
            OutputList.append("Softmax\n")
            OutputList.append(str(OutputList[len(OutputList)-3]))
            OutputList.append(str(OutputList[len(OutputList)-3]))
    #print(NextFc,"(NextFc)")
    #print(OutputList,"(OutputList)")
    with open("./OperationList.txt",'w+') as f:
        f.writelines(OutputList)
    return

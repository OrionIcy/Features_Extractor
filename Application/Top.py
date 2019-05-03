import TextAnalyser
import ReadFileModule as RFM

filename = input("Please enter your file name of CNN model: \n>>>")
TextAnalyser.TextAnalyser(filename)
OL = RFM.InitOperationList()
RFM.ReadOperationFile(OL)
#print(RFM.AMCalculate(OL))
RFM.Network.PrintTotal()
RFM.Network.PrintEveryClassOfLayer()
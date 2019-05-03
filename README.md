# Features_Extractor
A feature extraction tool for machine learning algorithms.  
* Using convolution as an example, we propose a design of features extractor for machine learning to acquire the compute feature of every single layer, which can be an instruction of dedicated hardware accelerators designing. Evaluation indicates our method is suitable for diverse algorithms.  
* The model file that is analyzed by the program should using general expression which was already defined in TensorFlow(such as tf.nn.conv2d(input,kernel,strides,padding)).  
* The output will be a list including the times of addtion and multiplication executed in every layer.  
  
Here are the supported operators:  
Conv2D  
MaxPool  
AvgPool  
BiasAdd  
LRN  
MatMul  
Softmax

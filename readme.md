# My PyTorch Learning Notes

> **_Note:_** This is a notebook for me to learn PyTorch. And about my own understanding about it.

## Basic 

### How Machine Learning(ML) achieves

1. Import dataset: get dataset from providers like [Kaggle](https://www.kaggle.com/). This is a huge project, which needs human to do, including collecting data, classifying data and make these files in a list. It also needs a `.csv` file, which likes tabular of all the data, provides the information of the given data. Some include the actual answer of the result, which is used to be compared with the result of module predicted.
2. Determine the Model: Select a model which is used for this project, like CNN.
3. Train the model: use the data in dataset. For this project, loading the pictures get from Kaggle, using their `.csv` file to train the model. The one called `xxx_train.csv` is the one used to train the model, which contains corresponding labels(actual answer). The one called `xxx_test.csv` is the one used to evaluate the performance of the trained model.

    In the training process, model get the input data, and process the data with the selected learning method, like CNN. Program will use the method to do predict, and compare the result with the true answer, a forward propagation will be applied if it is right. Or, a backward propagation will be invoked to modify the model. In the backward propagation, there is one thing called bias and weight. They are the matrix that created automatically by CNN. In the process of prediction, there are lots of CNN Kernel used for understanding pictures, each of them extracts a feature for the image which make the model understand what the image is. However, cause there are lots of feature, different feature contributes differently to getting to a correct conclusion. Weight and bias are two parameters that may be able to make the result close to right answer. 

### Terminology

- Batch_size: The size of samples in each iteration. For example, my dataset has 1000 images, 100 of them are input as a group and be used to train the model. 
- l_r: Learn Rate, it is like step of each iteration.
- Epoch: Number of training sessions. Each epoch means that all images in the dataset are traversed once.
- Iteration: The rate of renew the value of weight and bias. Everytime, when model give prediction is an iteration(after a batch). If it is different from the actual answer, a backward propagation will be invoked to optimise the value of weight and bias.
- gradient: a vector of partial derivatives of the loss function with respect to the network's parameter(weight & bias)
- forward propagation: a feedback to output the prediction, which is same with the true answer. It can be interpreted as positive feedback
- backward propagation: a feedback to let the model know it is a wrong predict. Then it will change the value of weight & bias.
- weight: 
- bias:
- label: it is a value in `.csv` files, which means the answer of the dataset. When do classification, a same label means the two objects have a same kind.
- Segmentation:
- CNN layer:
- pooling layer: 
- ReLU:
- loss:
- optimizer:
- tensor:
- Kernel:
- 
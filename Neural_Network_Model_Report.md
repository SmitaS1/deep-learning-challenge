# The performance of the deep learning model for Alphabet Soup.

## Overview 


The goal of this project is to develop a machine learning algorithm, specifically a binary classifier using neural networks, to predict the likelihood of success for applicants if funded by the fictional non-profit foundation, Alphabet Soup. Alphabet Soup aims to leverage data-driven approaches to select applicants with the highest potential for success in their ventures. By utilizing historical data and relevant features, the algorithm will classify applicants into two categories: successful and unsuccessful.

## Results

### Data Preprocessing

What variable(s) are the target(s) for your model?

In this code snippet, the target variable for the model is IS_SUCCESSFUL, which is extracted from the df_dummies DataFrame and stored in the variable y. Therefore, IS_SUCCESSFUL is the target variable for the model.

What variable(s) are the feature(s) for your model?

The features for the model are all the columns in the df_dummies DataFrame except for the target variable IS_SUCCESSFUL. These features are stored in the variable X. Therefore, the features for the model include all columns of df_dummies except IS_SUCCESSFUL.


What variable(s) should be removed from the input data because they are neither targets nor features?
Dropped EIN and NAME non-beneficial columns.


Choosing a cutoff point count of  528 for APPLICATION_TYPE and 1883 count FOR CLASSIFICATIONto bin rare categorical values together into a new value called "Other",
using pd.get_dummies() to convert categorical data to numeric,
dividing the data into a target array (IS_SUCCESSFUL) and features arrays,
applying the train_test_split to create a testing and a training dataset,
and finally, using StandardScaler to scale the training and testing sets

 
Compiling, Training, and Evaluating the Model

The model was required to achieve a target predictive accuracy higher than 75%. I made four official attempts using machine learning and neural networks. They all resulted in the same accuracy rate – right around 73%, so a little short of the required target accuracy.

Results from each model attempt are detailed below:

Attempt 1.

The hyperparameters used were:

layers = 2
layer1 = 8 neurons : activation function = relu
layer2 = 5 neurons : activation function = relu
output layer : activation function = sigmoid
epochs = 100
Loss: 0.5503498315811157, Accuracy: 0.7310787439346313


Attempt 2.
layers = 2
layer1 = 11 neurons : activation function = LeakyReLU
layer2 =  9 neurons : activation function = LeakyReLU
epochs = 50
Loss: 0.5826610922813416, Accuracy: 0.7239649891853333


Attempt 3.
layers = 3
layer1 = 8 neurons : activation function = relu
layer2 = 5 neurons : activation function = relu
layer2 = 6 neurons : activation function = relu
output layer : activation function = sigmoid
epochs = 60
Loss: 0.5550488233566284, Accuracy: 0.727580189704895


Attempt 4
layers = 3
layer1 = 8 neurons : activation function = tanh
layer2 = 5 neurons : activation function = tanh
layer2 = 6 neurons : activation function = tanh
output layer : activation function = sigmoid
epochs = 100
Loss: 0.5565226078033447, Accuracy: 0.7258309125900269


What steps did you take in your attempts to increase model performance?
I added more layers, modify the number of nurons, switched up the activation functions associated with each layer in an attempt to achieve higher model accuracy.

## Summary of Deep Learning Model Results:

The deep learning models attempted to predict the success of applicants if funded by Alphabet Soup, aiming for a target predictive accuracy exceeding 75%. However, all four official attempts using various neural network architectures and hyperparameters resulted in consistent accuracy rates around 73%, falling short of the desired threshold.

Attempt 1:
Utilized a two-layer neural network with ReLU activation function.
Achieved an accuracy of 73.11% after training for 100 epochs.

Attempt 2:
Employed a similar two-layer architecture with LeakyReLU activation function.
Yielded an accuracy of 72.40% after 50 epochs.

Attempt 3:
Expanded to a three-layer architecture with ReLU activation function.
Attained an accuracy of 72.76% after 60 epochs.

Attempt 4:
Utilized a three-layer architecture with tanh activation function.
Resulted in an accuracy of 72.58% after 100 epochs.
Despite variations in architecture and activation functions, all attempts failed to significantly improve predictive accuracy beyond 73%.

Recommendation for an Alternative Model:

Considering the limitations encountered with deep learning models, a recommendation is to explore the implementation of ensemble learning techniques such as Random Forest or Gradient Boosting Machine (GBM) for solving this classification problem.

Explanation of Recommendation:

Random Forest:

Random Forest is an ensemble learning method that aggregates predictions from multiple decision trees.
It is robust against overfitting and can handle high-dimensional data well.
Random Forest provides feature importance rankings, aiding in understanding the factors contributing to success.
Its ability to handle complex relationships between features might help capture patterns that neural networks missed.

Gradient Boosting Machine (GBM):
GBM is another ensemble learning technique that sequentially builds weak learners to reduce overall error.
It is less prone to overfitting compared to deep neural networks.
GBM effectively captures complex relationships in data and iteratively improves predictive performance.
By focusing on areas where previous models failed, GBM may provide better predictive accuracy for this classification problem.
By exploring these alternative models, we can leverage their strengths in handling complex data relationships, reducing overfitting, and providing interpretability, potentially leading to improved predictive accuracy for determining the success of applicants if funded by Alphabet Soup.


## Note: AlphabetSoupCharity_Optimization 
In this file I have used steps to increase model performance to predict accuracy exceed 75%

I added column 'NAME', added more layers, modify the number of nurons, switched up the activation functions associated with each layer in an attempt to achieve higher model accuracy.
 
layers = 3
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 8
hidden_nodes_layer3= 6
epochs = 30
Loss: 0.4892241656780243, Accuracy: 0.7560349702835083
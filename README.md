# Online Fraud Detection 

# <h3>Abstract</h3>
This project aims to tackle the problem of online payment fraud by applying machine learning techniques to accurately classify transactions as either fraudulent or legitimate. Using a dataset of historical transaction records, including details such as transaction type, amount, and account balances, this study trains and evaluates predictive models capable of detecting fraudulent behavior. The goal is to contribute to more secure digital financial systems in Africa by providing a data-driven fraud detection mechanism that operates efficiently and in real time. <br>

# <h3>Problem Statement</h3>
Online payment systems have revolutionized financial transactions across the globe, offering speed and convenience, but they also pose a growing risk of fraud. In Africa, this threat is especially pressing as digital adoption surges. According to the Nigeria Inter-Bank Settlement System (NIBSS), over ₦12.7 billion (approximately $27 million) was lost to electronic payment fraud in 2023 alone, a sharp indicator of the region’s vulnerability to cybercrime. 

# <h3>The Dataset</h3>
This dataset, sourced from Kaggle, contains historical transaction data aimed at identifying fraudulent online payments. It includes detailed information about each transaction, such as the transaction type, amount, and account balances before and after the transaction for both sender and recipient. A key feature is the isFraud column, which labels whether a transaction is fraudulent (1) or not (0). This labeled data makes it suitable for training supervised machine learning models for fraud classification.

Key features:

step: Time unit (1 step = 1 hour)

type: Nature of the transaction (e.g., transfer, payment)

amount: Transaction amount

nameOrig / nameDest: Sender and recipient identifiers

oldbalanceOrg / newbalanceOrig: Sender's balance before and after

oldbalanceDest / newbalanceDest: Recipient's balance before and after

isFraud: Target variable (1 = fraud, 0 = legitimate)

Dataset Source:KAGGLE 

Data can be found https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection

# <h3>Summary Table</h3>

## Model Training Instances

| **Model** | **Optimizer** | **Regularization** | **Early Stopping** | **Dropout** | **Learning Rate** | **Architecture**       | **Batch Size** | **Precision** | **Recall** | **F1-Score** | **AUC** |
|-----------|---------------|--------------------|---------------------|-------------|-------------------|-------------------------|----------------|---------------|------------|--------------|---------|
| Model 1   | Adam          | L2 (0.01)          | Yes              | 0.2         | 0.001             | [64, 32]                | 256            | 0.51          | 0.95       | 0.49         | —       |
| Model 2   | RMSprop       | L1 (0.01)          | No               | 0.2         | 0.001             | [64, 32]                | 512            | 0.94          | 0.94       | 0.94         | —       |
| Model 3   | SGD           | L1 (0.01)          | Yes              | 0.3         | 0.001             | [128, 64, 32]           | 512            | 0.86          | 0.86       | 0.86         | —       |
| Model 4   | AdamW         | L2 (0.01)          | Yes              | 0.3         | 0.001             | [256, 128, 64]          | 512            | 0.93          | 0.92       | 0.92         | —       |
| Model 5   | Nadam         | None               | Yes              | 0.4         | 0.001             | [256, 128, 64, 32]      | 256            | 0.96          | 0.96       | 0.96         | 0.98    |
 0.90      | 0.95    |


## Logistic Regression Model Summary

| C Value | Penalty | Solver  | Max Iterations | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------------|----------|-----------|--------|----------|
| 0.5     | L2      | lbfgs   | 100           | 0.9486  | 0.9509   | 0.9461 | 0.948   |


# <h3>Summary</h3>
The training process was quite hectic,having overfitting models and try to adjust parameters to improve accuracy. The process involved using neural network and traditional machine Learning algorithm.

In training the neural network, there were four(4) instances using different optimizers, varied number of layers including Dropout layers  to build complex model architecture to effectively learn patterns in data and regularisers as shown in the table above. In all the instances; Instance 2 which used Stochastic Gradient Descent(SGD), L1 regulariser, Dropout and binary cross entropy was my  best model with an accuracy of 0.8767.This model recorded the lowest test loss of 0.3795 among all the four(4) instances. From the confusion matrix, the model was also able to correctly predict the classes  as shown here; [149  22   15   114 ] This model was initially overfitting but with the introduction of the l1 regualriser; it helped ignore less important features and keeping the important ones to prevent the model from overfitting. The use of the SGD optimizer enabled faster convergence by applying gradient updates efficiently to able to make accurate predictions.

The traditional Machine Learning algorithm I used was Logistic Regression making use of the hyperparameters as shown the table above. The accuracy was low because the Logistic regression model was not able to learn the patterns very well as it is only learning the linear relations. Comparing the ML algorithm and the neural network , the Neural network performed way better because it is able to learn the non-linear relationships to correctly make predictions and with the use of optimizers and regularisers ; preventing feature selection and avoiding overfitting.

# Video Presentation


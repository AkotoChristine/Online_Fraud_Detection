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

| **Model** | **Optimizer** | **Regularization** | **Early Stopping** | **Dropout** | **Learning Rate** | **Layers** | **Precision** | **Recall** | **F1-Score** | **AUC**  |
|-----------|---------------|--------------------|---------------------|-------------|-------------------|------------|---------------|------------|--------------|---------|
| Model 1   | Adam          | L2 (0.01)          | Yes                 | 0.2         | 0.001             | 2          | 0.93          | 0.98       | 0.95         | 0.9919  |
| Model 2(Instance1)  | RMSprop       | L1 (0.01)          | No                  | 0.2         | 0.001             | 2          | 0.94          | 0.94       | 0.94         | 0.9916  |
| Model 3   | SGD           | L1 (0.01)          | Yes                 | 0.3         | 0.001             | 3          | 0.86          | 0.86       | 0.86         | 0.9375  |
| Model 4   | AdamW         | L2 (0.01)          | Yes                 | 0.3         | 0.001             | 3          | 0.93          | 0.92       | 0.92         | 0.9867  |
| Model 5   | Nadam         | None               | Yes                 | 0.4         | 0.0005            | 4          | 0.96          | 0.96       | 0.96         | 0.9951  |



## Logistic Regression Model Summary

| C Value | Penalty | Solver  | Max Iterations | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------------|----------|-----------|--------|----------|
| 0.5     | L2      | lbfgs   | 100           | 0.9486  | 0.9509   | 0.9461 | 0.948   |


# <h3>Summary</h3>
The training process was quite challenging due to the highly imbalanced nature of the dataset. Initially, this imbalance negatively impacted model performance, making it difficult to accurately detect the minority class. To address this, I applied class weighting techniques and ensured that both X_test and y_test(A ratio of 0.5 to 200) were balanced, allowing for a fair evaluation of the model's effectiveness. Throughout the process, I also encountered overfitting in some models, which required careful tuning of parameters such as regularization, dropout, and learning rate to improve generalization and overall accuracy. The project involved both deep learning approaches using neural networks and a traditional machine learning baseline with Logistic Regression, allowing for a comprehensive comparison between the two.

In training the neural network, five (5) different model instances were developed, each using a different optimizer, varied layer depths, dropout rates, and regularization techniques to identify the most optimal configuration for accurate predictions. These experiments were guided by the goal of building a robust model that can generalize well on imbalanced data. As presented in the summary table above, Model 5, which used the Nadam optimizer, emerged as the best-performing model, achieving an accuracy of approximately 95.1%, a precision and recall of 96%, and the highest AUC score of 0.9951, indicating near-perfect class separation.

This model leveraged a deep architecture with 4[256, 128, 64, 32]   layers and a dropout rate of 0.4 to prevent overfitting,  without explicit L1 or L2 regularization. The lower learning rate of 0.0005 ensured smoother convergence and stability during training. While earlier models like Model 1 (Adam) and Model 2 (RMSprop) showed strong metrics, they fell short in F1-score and AUC compared to Nadam. From the confusion matrix, Model 5 was also able to correctly classify both majority and minority classes more consistently, which is critical given the class imbalance in the dataset.

The combination of the Nadam optimizer and early stopping allowed the model to learn more efficiently and avoid overtraining. Compared to traditional machine learning methods such as Logistic Regression, this neural network architecture was able to learn non-linear patterns in the data more effectively and make significantly more accurate predictions, thanks to the use of deep layers, dropout, and advanced optimization strategies.

# Video Presentation


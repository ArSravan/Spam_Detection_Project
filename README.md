
# Spam Detection Project

## Project Overview
This project aims to detect spam messages using machine learning techniques, with a Multinomial Naive Bayes  achieving an accuracy of 98%. The objective is to automatically identify and filter out spam messages from a dataset, enhancing communication efficiency.

## Dataset
- **Source**: The dataset is sourced from [Kaggle - Spam Detection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **Description**: 
  - The dataset consists of messages with the following columns:
    - **`message`**: The text content of the message
    - **`label`**: Target variable indicating whether the message is spam (1) or not spam (0)

## Approach
- **Data Preprocessing**:
  - **Data Cleaning**: Removed duplicates and irrelevant data from the dataset.
  - **Text Processing**: Processed the text by:
    - Tokenization: Splitting the message into individual words.
    - Removing stop words: Eliminated common words that do not contribute to the meaning (e.g., "and," "the").
    - Lowercasing: Converted all text to lowercase to ensure uniformity.
    - Stemming: Reduced words to their root form (e.g., "running" to "run").
  - **Feature Extraction**: Utilized the TF-IDF (Term Frequency-Inverse Document Frequency) method to convert text data into numerical features, which helps in evaluating the importance of words in the context of the dataset.
  
- **Model**:
  - **Machine Learning Models Used**: Initially tried Logistic Regression and Support Vector Machine (SVM) but ultimately chose the Multinomial Naive Bayes  due to its high accuracy and ability to handle complex datasets.

- **Evaluation**:
  - The following evaluation metrics were used to assess model performance:
    - **Accuracy**: 98%
    - **Precision**: 96%
    - **Recall**: 95%
    - **F1-score**: 99%
  - The dataset was split into 80% training data and 20% testing data to validate the model.

## Results
- Multinomial Naive Bayes  demonstrated exceptional performance, achieving high accuracy and precision. The model effectively identified spam messages with minimal false positives, indicating its reliability. A confusion matrix visualized the results, confirming the model's proficiency in distinguishing between spam and non-spam messages.

## Challenges
- **Handling Imbalanced Data**: The initial dataset exhibited class imbalance, with more non-spam messages than spam. Addressed this by applying oversampling techniques (e.g., SMOTE) to balance the classes effectively.
- **Model Overfitting**: Experienced issues with overfitting during training, which were mitigated by employing cross-validation and tuning hyperparameters.

## Future Improvements
- Explore deep learning algorithms such as LSTM (Long Short-Term Memory) networks for potentially better accuracy.
- Incorporate additional features, such as user metadata, to enhance model predictions.
- Improve the dataset by collecting more labeled examples to ensure robustness.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArSravan/Spam_Detection_Project.git
   cd Spam_Detection_Project
   pip install pandas scikit-learn nltk

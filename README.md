Email Spam Detection Project
This project focuses on building a machine learning model to classify emails as spam or ham (not spam). It utilizes two datasets containing email text and their corresponding labels.

Project Goal
The primary goal is to develop an accurate and reliable spam detection system using natural language processing techniques and machine learning algorithms.

Datasets Used
Two datasets were used in this project:
emails.csv: Contains a collection of emails with word counts and a 'Prediction' column indicating spam (1) or ham (0).
spam_ham_dataset.csv: Contains emails with 'text', 'label' (spam/ham), and 'label_num' (1/0) columns.
These datasets were combined and preprocessed to create a unified dataset for model training and evaluation.

Project Steps
The project followed these key steps:
Data Loading: Loaded the two datasets into pandas DataFrames.
Data Cleaning and Transformation:
Handled missing values (none were found in this case).
Standardized column names ('label_num' in spam_ham_dataset.csv was renamed to 'Prediction' to match emails.csv).
Selected relevant columns ('text' and 'Prediction') from the combined dataset.
Exploratory Data Analysis (EDA): Visualized word frequencies in spam and ham emails using WordClouds to understand the common terms in each category.
Preprocessing and Vectorization:
Applied text preprocessing techniques (removing non-alphanumeric characters, converting to lowercase, tokenization, removing stop words, and stemming) to the email text.
Converted the cleaned text data into numerical features using TF-IDF vectorization.
Data Splitting: Split the vectorized data into training and testing sets.
Handling Data Imbalance (Oversampling): Used Random Oversampling to address the class imbalance in the training data, creating a resampled training set with an equal number of spam and ham examples.
Model Training: Trained a Multinomial Naive Bayes model on both the original and the resampled training data.
Model Evaluation: Evaluated the performance of both models using accuracy, classification reports, and confusion matrices.
Confusion Matrix Visualization: Visualized the confusion matrices to better understand the models' performance in terms of true positives, true negatives, false positives, and false negatives.

Results
The Multinomial Naive Bayes model trained on the original data achieved an accuracy of 0.9585.
The model trained on the oversampled data also achieved an accuracy of 0.9585. However, the classification report and confusion matrix for the resampled model showed a slight improvement in recalling spam emails (higher recall for class 1), indicating that oversampling helped the model better identify the minority class.

Evaluation Metrics (Resampled Model):
Accuracy: 0.9585
Classification Report:
Ham (0): Precision: 0.99, Recall: 0.95, F1-score: 0.97
Spam (1): Precision: 0.89, Recall: 0.98, F1-score: 0.93
Confusion Matrix:
True Positives (Spam correctly identified): 287
True Negatives (Ham correctly identified): 705
False Positives (Ham incorrectly identified as Spam): 37
False Negatives (Spam incorrectly identified as Ham): 6

Conclusion
The project successfully built and evaluated a spam detection model. While both models performed well, the model trained on resampled data showed improved performance in identifying spam emails, which is crucial for a spam detection system.

How to Run the Code
Clone the repository.
Install the required libraries (e.g., pandas, numpy, matplotlib, wordcloud, scikit-learn, imblearn).
Ensure you have the emails.csv and spam_ham_dataset.csv files in the appropriate directory or modify the code to load them from your desired location.
Run the Jupyter notebook or Python script to execute the code step by step.

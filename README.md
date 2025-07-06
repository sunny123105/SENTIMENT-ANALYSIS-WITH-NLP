COMPANY : CODTECH IT SOLUTIONS

NAME : PATEL SUNNY KANAKKUMAR

INTERN ID : CT04DG881

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEK

MENTOR : NEELA SANTOSH

DESCRIPTION OF TASK :This project focuses on implementing a sentiment analysis model using a traditional machine learning approach. The primary goal is to classify text reviews as either positive or negative by analyzing the sentiment expressed in the text. The approach involves converting raw text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) and then training a Logistic Regression classifier to make predictions.

Dataset Overview
The dataset used is a CSV file named reviews_balanced.csv. It contains two main columns:

text: The actual review content written by users.

label: The sentiment associated with the review, usually binary (e.g., positive or negative).

This dataset must be preprocessed before being fed into the machine learning model since raw text cannot be directly interpreted by algorithms.

Methodology
The project is implemented in Python using key libraries such as pandas, scikit-learn, and matplotlib. Below are the major steps followed in the process:

1. Data Loading and Inspection
The dataset is loaded using pandas.read_csv(). The first few records are printed to understand the structure and verify data quality.

2. Text Vectorization
The raw text is transformed into numeric format using TfidfVectorizer from Scikit-learn. TF-IDF helps in identifying the most relevant words by balancing the frequency of terms in individual documents with their occurrence across all documents. Stop words (commonly used words like “the”, “is”, “and”) are removed using stop_words='english'.

vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)

3. Train-Test Split
To evaluate the model fairly, the dataset is split into training and testing sets using an 80-20 ratio. This ensures that the model is trained on one portion and tested on unseen data.

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

4. Model Training
A Logistic Regression model is chosen due to its efficiency and suitability for binary classification. The model is trained using the .fit() method on the vectorized training data.

model = LogisticRegression()
model.fit(X_train, y_train)

5. Model Evaluation
Predictions are made on the test data, and the performance is evaluated using a classification report. This report provides important metrics like accuracy, precision, recall, and F1-score for each class.

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# Sample data
data = pd.read_csv(r'C:\Users\HP\Downloads\reviews_balanced.csv')  # Must contain 'text' and 'label' columns
print(data.head())
X = data['text']
y = data['label']

# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X_vect = vectorizer.fit_transform(X)
 

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

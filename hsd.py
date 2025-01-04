import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.util import pr
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
df = pd.read_csv("twitter_data.csv")
print(df.head())
df['labels'] = df['class'].map({0:"hatespeech detected" , 1:"offensive language detected",3:"neither hate nor offensive"})
df.head()
df = df[['tweet' ,'labels']]
df.head()
import re
import string  # Import the string module

def clean(text):
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+' , '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+' , '', text)
    
    # Remove punctuation using string.punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove newlines
    text = re.sub('\n', '', text)
    
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    
    return text

# Apply the clean function to the 'tweet' column of the DataFrame
df["tweet"] = df["tweet"].apply(clean)

# Check the result
print(df.head())


# Sample Data (Replace with actual data)
df = np.array([["Tweet 1", 0], ["Tweet 2", 1], ["Tweet 3", 0]])  # Replace with actual data
df = pd.DataFrame(df, columns=['tweet', 'labels'])

# Features (X) and Labels (y)
X = df['tweet']
y = df['labels']

# Convert the 'X' data into a sparse matrix of features using CountVectorizer
cv = CountVectorizer()
X_transformed = cv.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.33, random_state=10)

# Initialize and train the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=20)  # Ensures reproducibility
clf.fit(X_train, y_train)

# Predicting the test data
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")  # Display accuracy as a percentage

# Displaying more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# You can test prediction on a new sample
test_data = ["I hate you", "dawg "]
test_data_transformed = cv.transform(test_data)  # Apply the same transformation to the test data
test_predictions = clf.predict(test_data_transformed)
print("\nPredictions on new test data:", test_predictions)


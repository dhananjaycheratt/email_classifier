import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle
import streamlit as st

# Load the Spambase dataset
data = pd.read_csv("spam_ham_dataset.csv")

# Separate features and labels
X = data.drop('label_num', axis=1)  # Features
y = data['label_num']  # Labels

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Preprocess labels using label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Print the shape of preprocessed data
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Download stopwords if not already present
nltk.download('stopwords')

# Download the Punkt tokenizer if not already present
nltk.download('punkt')

# Download the Porter stemmer if not already present
nltk.download('punkt')

# Define a function for text preprocessing
def preprocess_text(text):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Convert to lowercase
    filtered_tokens = [token.lower() for token in filtered_tokens]
    
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

# Apply preprocessing to the text column of the data DataFrame
data['preprocessed_text'] = data['text'].apply(preprocess_text)

# Print the preprocessed text
print(data['preprocessed_text'])

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the preprocessed text data
X = vectorizer.fit_transform(data['preprocessed_text'])

# Convert the sparse matrix to a dense array
X = X.toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['label_num'], test_size=0.2, random_state=42)

# Create an instance of the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Save the model to a file
pickle.dump(model, open("spam_detection_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

def predict_email(email_text):
    # Preprocess the email text
    preprocessed_email = preprocess_text(email_text)
    
    # Transform the preprocessed email text into a feature vector
    email_vector = vectorizer.transform([preprocessed_email])
    
    # Make a prediction on the feature vector
    prediction = model.predict(email_vector)
    
    # Interpret the prediction
    if prediction == 0:
        return "The email is not spam."
    else:
        return "The email is spam."
    
# Set the page title
st.title("Spam Email Detection")

# Add a text input field for the user to enter an email
email = st.text_input("Enter an email")

# Add a button to predict if the email is spam or not
if st.button("Predict"):
    prediction = predict_email(email)
    st.write(prediction)




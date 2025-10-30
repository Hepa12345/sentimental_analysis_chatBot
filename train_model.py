import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
data = pd.read_csv(url)[['label', 'tweet']]
data['label'] = data['label'].map({0: 'Negative', 1: 'Positive'})
data.dropna(inplace=True)

# Step 2: Balance the dataset (undersample)
positive = data[data['label'] == 'Positive']
negative = data[data['label'] == 'Negative'].sample(len(positive), random_state=42)
balanced_data = pd.concat([positive, negative])

# Step 3: Debug - show new label distribution
print("Balanced Label distribution:\n", balanced_data['label'].value_counts())

# Add custom examples to improve short-word detection
extra = pd.DataFrame({
    'tweet': [
        'happy', 'joyful', 'great', 'wonderful day', 'awesome', 'love it',
        'worst', 'terrible', 'hate', 'bad', 'awful', 'disgusting'
    ],
    'label': [
        'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive',
        'Negative', 'Negative', 'Negative', 'Negative', 'Negative', 'Negative'
    ]
})
balanced_data = pd.concat([balanced_data, extra])


# Step 4: Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(balanced_data['tweet'])
y = balanced_data['label']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Balanced Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model training completed and saved.")

from textblob import TextBlob

def predict_sentiment(user_input):
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Test manually
if __name__ == "__main__":
    while True:
        text = input("You: ")
        print("Predicted Sentiment:", predict_sentiment(text))

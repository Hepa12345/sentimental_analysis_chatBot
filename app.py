from flask import Flask, request, render_template
from chatbot_model import predict_sentiment


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form['message']
    sentiment = predict_sentiment(user_input)

    if sentiment == "Positive":
        response = "😊 I'm glad to hear that!"
    elif sentiment == "Negative":
        response = "😔 I'm here for you. Want to talk about it?"
    else:
        response = "😐 I see. Can you tell me more?"

    return render_template("index.html", user_input=user_input, sentiment=sentiment, response=response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)








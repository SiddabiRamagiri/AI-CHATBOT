print("✅ Script started!")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Knowledge base
corpus = [
    "hi", "hello", "hey", "hii", "heyy",
    "how are you", "what is your name", 
    "tell me a joke", "goodbye", "bye", "see you","what are you doing","where are you"
]

responses = [
    "Hello there! 😊", 
    "Hi! How can I assist you?", 
    "Hi! What’s up?", 
    "Hello there! 😊", 
    "Hi! What’s up?", 
    "I'm doing great, thanks for asking!", 
    "I'm ChatBot, your assistant!", 
    "Why don't scientists trust atoms? Because they make up everything! 😂", 
    "Goodbye! Have a great day!", 
    "Bye-bye! 👋",
    "See you soon! 👋"
    "I am good and you",
    "nothing and you","i am in your system"
]


# TF-IDF setup
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    confidence = similarity[0][index]
    if confidence > 0.3:
        return responses[index]
    else:
        return "Sorry, I didn't quite understand that."

# ✅ This part is crucial for running in terminal
if __name__ == "__main__":
    print("🤖 ChatBot is online! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        response = get_response(user_input)
        print("Bot:", response)

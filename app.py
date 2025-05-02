import streamlit as st
from chatbot import get_response

st.title("🤖 AI Chatbot")
st.write("Type something and the chatbot will respond!")

user_input = st.text_input("You:", "")

if user_input:
    reply = get_response(user_input)
    st.write("Bot:", reply)

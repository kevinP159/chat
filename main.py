import streamlit as st
from app import get_few_shot_db_chain
import time

st.title("BrandStore Assistant👕")

question = st.text_input("Question: ")

if question:
    with st.spinner('Thinking...'):
        try:
            chain = get_few_shot_db_chain()
            response = chain.run(question)
            
            st.header("Answer")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with a different question.")

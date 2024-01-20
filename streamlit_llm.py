import streamlit as st
import random
import time
import sqlite3

import google.generativeai as genai

st.title("Simple chat")

GOOGLE_API_KEY = "AIzaSyCaaQOjdmVYJpE7u21QqTUL9LyGX1EhwGEw"
genai.configure(api_key=GOOGLE_API_KEY)



## Function To Load Google Gemini Model and provide queries as response

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

## Fucntion To retrieve query from the database

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

prompt_eng=[
    """
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output
    """
]


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        resp = message["content"]
        print(f"heyyyyyy : {resp}")

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response= get_gemini_response(prompt,prompt_eng)
        response_final=read_sql_query(response,"student.db")
        print(f"response_final is {type(response_final)}") 
        assistant_response = response_final
        # Simulate stream of response with milliseconds delay
        # for chunk in assistant_response:
        #     # full_response += chunk + " "
        #     time.sleep(0.05)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(chunk + "▌")
        message_placeholder.markdown(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from few_shots import few_shots
import os
from dotenv import load_dotenv
import streamlit as st
import sqlite3

load_dotenv()

def init_db():
    """Initialize the database if it doesn't exist"""
    if not os.path.exists("BranszStore.db"):
        try:
            conn = sqlite3.connect('BranszStore.db')
            with open('create_db.sql', 'r') as f:
                conn.executescript(f.read())
            conn.commit()
            conn.close()
            st.success("Database created successfully!")
        except Exception as e:
            st.error(f"Error creating database: {str(e)}")

def get_few_shot_db_chain():
    # Initialize database first
    init_db()
    
    # Create database connection
    db = SQLDatabase.from_uri("sqlite:///BranszStore.db", sample_rows_in_table_info=3)

    # Get the API key from environment variables (for Hugging Face Spaces)
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        # Fallback to streamlit secrets (for local testing)
        try:
            google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
        except:
            st.error("GOOGLE_API_KEY not found in environment variables or secrets")
            return None

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.6
    )

    # Set up embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # SQLite prompt template
    sqlite_prompt = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use `date('now')` function to get the current date, if the question involves "today".

    Use the following format:

    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=sqlite_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )
    
    try:
        chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        return None

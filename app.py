import streamlit as st
import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(page_title="DataSight AI", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for a Premium Vibe ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Chat bubbles */
    .stChatMessage {
        background-color: #1E2127;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8F9FA !important;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("✨ DataSight AI")
st.markdown("*Your intelligent, conversational database assistant.*")
st.divider()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render any dataframes in history
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True, hide_index=True)

# Connect to database
@st.cache_resource
def get_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chinook.db")
    return db_path, SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

# Initialize AI model
@st.cache_resource
def get_model():
    return ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

db_path, db = get_db()
model = get_model()

# Accept user input
if prompt := st.chat_input("Ask a question (e.g., 'What are the top 5 best-selling artists?'):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your database..."):
            try:
                # 1. Get Schema and Generate SQL Query
                schema = db.get_table_info()
                sql_prompt = f"""You are a SQLite data analyst expert. Given the database schema below, write a syntactically correct SQLite query to answer the user's question.
IMPORTANT RULES:
1. ONLY use tables and columns that explicitly exist in the schema below. Do NOT make up or hallucinate column names (e.g., if there is no Salary column, do not use it).
2. If the user asks a question that cannot be answered with the available columns, write a query that returns nothing (e.g. SELECT 'Data not available' AS Error) instead of guessing columns.
3. Unless the user specifies a number of examples, limit your query to at most 10 results.
4. Return ONLY the SQL query, without any markdown formatting or explanation.

Schema:
{schema}

Question: {prompt}
SQL Query:"""

                sql_response = model.invoke(sql_prompt)
                sql_query = sql_response.content.strip()
                
                # Clean markdown formatting if present
                if "```sql" in sql_query:
                    sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql_query:
                    sql_query = sql_query.split("```")[1].strip()
                
                # Show the SQL query in a nice expander
                with st.expander("🛠️ View Generated SQL"):
                    st.code(sql_query, language="sql")
                
                # 2. Execute SQL and get Dataframe for a beautiful table
                df = pd.DataFrame()
                try:
                    conn = sqlite3.connect(db_path)
                    df = pd.read_sql_query(sql_query, conn)
                    conn.close()
                except Exception as db_e:
                    st.warning(f"⚠️ The AI generated an invalid query (often because the data you asked for doesn't exist in the database).\n\n**Details:** `{db_e}`")
                
                # Display the beautiful dataframe
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    result_str = df.head(10).to_string()
                else:
                    if "Data not available" in sql_query:
                        st.info("The database doesn't contain the necessary information to answer that.")
                    result_str = "No results returned."
                
                # 3. Formulate the final natural language answer
                final_prompt = f"Given the user question, the SQL query, and the query result table, provide a concise and engaging answer to the user.\n\nQuestion: {prompt}\nSQL: {sql_query}\nResult:\n{result_str}\n\nAnswer:"
                
                answer_msg = model.invoke(final_prompt)
                answer = answer_msg.content if hasattr(answer_msg, 'content') else str(answer_msg)
                
                # Show the final answer
                st.markdown(f"**{answer}**")
                
                # Add assistant response to chat history
                history_content = f"**{answer}**"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": history_content,
                    "dataframe": df if not df.empty else None
                })
                
            except Exception as e:
                error_msg = f"Sorry, I ran into a system error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

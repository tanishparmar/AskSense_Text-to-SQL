import streamlit as st
import os
import re
from schema_extractor import csvs_to_sqlite, extract_schema
from predict import SQLPredictor
from sentence_model import SentenceModelPredictor
import sqlite3
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AskSense",
    page_icon="image.png",  # ← your logo as the favicon
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styles */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Chat Message Styles */
    .chat-message {
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .user-message {
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
    }
    
    /* Code Block Styles */
    .sql-code {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 0.375rem;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        margin: 0.75rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Schema Info Styles */
    .schema-info {
        background-color: #212529;
        color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0d6efd;
        margin: 1.25rem 0;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .schema-info strong {
        color: #6ea8fe;
        font-weight: 600;
    }
    .schema-info br {
        margin-bottom: 0.625rem;
    }
    
    /* Button and Input Styles */
    .stButton>button {
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTextInput>div>div>input {
        border-radius: 0.375rem;
    }
    
    /* Dataframe Styles */
    .dataframe {
        border-radius: 0.375rem;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    
    /* Sidebar Styles */
    .css-1d391kg, .css-163ttbj {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def create_detailed_schema_info(schema):
    """Create detailed schema information for the model"""
    schema_parts = []
    for table_name, columns in schema.items():
        column_info = []
        for col_name, col_type in columns:
            column_info.append(f"{col_name} ({col_type})")
        table_info = f"Table '{table_name}' with columns: {', '.join(column_info)}"
        schema_parts.append(table_info)
    return "Database Schema:\n" + "\n".join(schema_parts)

def extract_table_names_from_sql(sql):
    """Extract table names from SQL query"""
    sql_lower = sql.lower()
    sql_clean = re.sub(r'--.*$', '', sql_lower, flags=re.MULTILINE)
    sql_clean = re.sub(r"'.*?'", "''", sql_clean)
    sql_clean = re.sub(r'".*?"', '""', sql_clean)
    table_patterns = [
        r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'insert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'delete\s+from\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    tables = set()
    for pattern in table_patterns:
        matches = re.findall(pattern, sql_clean)
        tables.update(matches)
    return list(tables)

def validate_sql_tables(sql, available_tables):
    """Validate that all tables in SQL exist in the schema"""
    sql_tables = extract_table_names_from_sql(sql)
    missing_tables = [table for table in sql_tables if table not in available_tables]
    if missing_tables:
        return False, f"Tables not found in database: {', '.join(missing_tables)}. Available tables: {', '.join(available_tables)}"
    return True, ""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'sqlite_path' not in st.session_state:
    st.session_state.sqlite_path = None
if 'schema' not in st.session_state:
    st.session_state.schema = None
if 'predictors' not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.predictors = {
            'model_m3': SQLPredictor(model_path="models/model-m3"),
            'model_m3_v2': SQLPredictor(model_path="models/model-m3-v2"),
            'model_m3_v4_anti_hallucination': SQLPredictor(model_path="models/model-m3-v4-anti-hallucination")
        }
        st.session_state.sentence_model = SentenceModelPredictor()
        try:
            from huggingface_model import HuggingFacePredictor
            st.session_state.huggingface_model = HuggingFacePredictor()
        except Exception as e:
            st.warning(f"Failed to load Hugging Face model: {str(e)}")
            st.session_state.huggingface_model = None

# # Header
# st.title("AskMyData")
# st.markdown("Transform natural language questions into SQL queries and get instant insights from your data.")

# Header – centered logo + caption using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("image.png", width=400)


# Sidebar for file upload and schema display
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_files = st.file_uploader(
        "Choose CSV file(s)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files. Each file becomes a table."
    )
    if uploaded_files:
        try:
            with st.spinner("Processing files..."):
                sqlite_path = csvs_to_sqlite(uploaded_files)
                schema = extract_schema(sqlite_path)
                st.session_state.sqlite_path = sqlite_path
                st.session_state.schema = schema
                st.session_state.chat_history = []
            st.success(f"Loaded {len(schema)} table(s)")
            st.subheader("Database Schema")
            for table_name, columns in schema.items():
                with st.expander(f"{table_name}"):
                    for col_name, col_type in columns:
                        st.write(f"• `{col_name}` ({col_type})")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            st.session_state.sqlite_path = None
            st.session_state.schema = None
    else:
        st.info("Upload CSV files to get started")

# Main chat interface
st.header("Chat with Your Data")
if not st.session_state.sqlite_path or not st.session_state.schema:
    st.info("Please upload CSV files in the sidebar to begin chatting.")
    st.stop()

detailed_schema_info = create_detailed_schema_info(st.session_state.schema)
available_tables = list(st.session_state.schema.keys())

with st.expander("Current Database Schema", expanded=False):
    st.markdown(f"""
    <div class="schema-info">
        <strong>Available Tables:</strong> {', '.join(available_tables)}
        <br><br>
        <strong>Detailed Schema:</strong><br>
        {detailed_schema_info.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("**Generated SQL:**")
            st.code(message.get("sql", ""), language="sql")
            if message.get("error"):
                st.error(f"**Error:** {message['error']}")
            elif message.get("result") is not None:
                st.write("**Results:**")
                st.dataframe(message["result"], use_container_width=True)
            else:
                st.info("No results to display")

if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                if hasattr(st.session_state, 'sentence_model'):
                    enhanced_prompt = st.session_state.sentence_model.enhance_sql_generation(
                        question=prompt,
                        schema_info=detailed_schema_info,
                        db_id="user_uploaded"
                    )
                else:
                    enhanced_prompt = f"""
Database Schema:
{detailed_schema_info}

Available tables: {', '.join(available_tables)}

Question: {prompt}

IMPORTANT: Only use the tables and columns listed above. Do not create or reference tables that don't exist.
"""

                model_results = {}
                best_model = None
                best_result = None
                best_sql = None
                best_error = None
                all_models_failed = True

                # Custom models
                for model_name, predictor in st.session_state.predictors.items():
                    try:
                        sql = predictor.predict_sql(
                            question=enhanced_prompt,
                            db_info=detailed_schema_info,
                            db_id="user_uploaded",
                            method="beam",
                            verbose=False
                        )
                        is_valid, validation_error = validate_sql_tables(sql, available_tables)
                        if is_valid:
                            try:
                                conn = sqlite3.connect(st.session_state.sqlite_path)
                                result_df = pd.read_sql_query(sql, conn)
                                conn.close()
                                model_results[model_name] = {'sql': sql, 'result': result_df, 'error': None, 'valid': True, 'executable': True}
                                if best_model is None or (not result_df.empty and best_result is None):
                                    best_model, best_sql, best_result, best_error, all_models_failed = model_name, sql, result_df, None, False
                            except Exception as e:
                                model_results[model_name] = {'sql': sql, 'result': None, 'error': str(e), 'valid': True, 'executable': False}
                                if best_model is None:
                                    best_model, best_sql, best_error = model_name, sql, str(e)
                        else:
                            model_results[model_name] = {'sql': sql, 'result': None, 'error': validation_error, 'valid': False, 'executable': False}
                            if best_model is None:
                                best_model, best_sql, best_error = model_name, sql, validation_error
                    except Exception as e:
                        model_results[model_name] = {'sql': "-- Error generating SQL", 'result': None, 'error': str(e), 'valid': False, 'executable': False}
                        if best_model is None:
                            best_model, best_sql, best_error = model_name, "-- Error generating SQL", str(e)

                # fallback
                if (all_models_failed or best_result is None or best_error is not None) and st.session_state.huggingface_model:
                    try:
                        sql = st.session_state.huggingface_model.predict_sql(
                            question=enhanced_prompt,
                            db_info=detailed_schema_info,
                            db_id="user_uploaded",
                            method="beam",
                            verbose=False
                        )
                        is_valid, validation_error = validate_sql_tables(sql, available_tables)
                        if is_valid:
                            try:
                                conn = sqlite3.connect(st.session_state.sqlite_path)
                                result_df = pd.read_sql_query(sql, conn)
                                conn.close()
                                model_results['fallback'] = {'sql': sql, 'result': result_df, 'error': None, 'valid': True, 'executable': True}
                                if best_result is None or best_error is not None or (not result_df.empty and (best_result is None or best_result.empty)):
                                    best_model, best_sql, best_result, best_error = 'fallback', sql, result_df, None
                            except Exception as e:
                                model_results['fallback'] = {'sql': sql, 'result': None, 'error': str(e), 'valid': True, 'executable': False}
                                if best_result is None:
                                    best_model, best_sql, best_error = 'fallback', sql, str(e)
                        else:
                            model_results['fallback'] = {'sql': sql, 'result': None, 'error': validation_error, 'valid': False, 'executable': False}
                    except Exception as e:
                        model_results['fallback'] = {'sql': "-- Error generating SQL", 'result': None, 'error': str(e), 'valid': False, 'executable': False}
                        if best_model is None:
                            best_model, best_sql, best_error = 'fallback', "-- Error generating SQL", str(e)

                # Display
                st.write("**Generated SQL:**")
                st.code(best_sql, language="sql")
                if best_error:
                    st.error(f"**Error:** {best_error}")
                elif best_result is not None:
                    if best_result.empty:
                        st.info("Query returned no results")
                    else:
                        st.write("**Results:**")
                        st.dataframe(best_result, use_container_width=True)
                else:
                    st.info("No results to display")

                sql, result, error = best_sql, best_result, best_error
            except Exception as e:
                st.error(f"**Model Error:** {str(e)}")
                sql, error, result = "-- Error generating SQL", str(e), None

        # Save history
        st.session_state.chat_history.append({
            "role": "assistant", "content": "Generated SQL and results",
            "sql": sql, "result": result, "error": error, "best_model": best_model
        })



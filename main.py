import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Set Streamlit page layout
st.set_page_config(page_title="Excel Data Visualization with LLM",  layout="wide")

MODEL_NAME = "gpt-4o"

def extract_code_from_markdown(md_text):
    code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)
    return "\n".join([block[1].strip() for block in code_blocks])

def execute_openai_code(response_text: str, df: pd.DataFrame):
    code = extract_code_from_markdown(response_text)
    if code:
        try:
            # Execute the generated code in the global context
            exec(code, globals())
            # Collect and display each generated figure as an image
            for num in plt.get_fignums():
                fig = plt.figure(num)
                fig.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                st.image(buf, use_container_width=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
            plt.close('all')
        except Exception as e:
            error_message = str(e)
            st.error(f"📟 Apologies, failed to execute the code due to the error: {error_message}")
            st.warning(
                """
                📟 Check the error message and the code executed above to investigate further.
                
                Pro tips:
                - Tweak your prompts to overcome the error
                - Use the words 'Plot' or 'Subplot'
                - Use simpler, concise words
                - Remember, I'm specialized in displaying charts not in conveying information about the dataset
                """
            )
    else:
        st.write(response_text)

def handle_openai_query(df, column_names, selected_llm, custom_instruction=""):
    default_query = (
        "First, analyze the DataFrame 'df' by performing the following steps:\n"
        "- Compute basic summary statistics for each column.\n"
        "- Identify numeric and categorical columns.\n"
        "- Handle missing values:\n"
        "  - Fill missing values in numeric columns with their mean.\n"
        "  - Fill missing values in categorical columns with 'Unknown'.\n\n"
        "Then, generate visualizations using Pandas and Matplotlib:\n"
        "1️⃣ **Bar Charts**: Create bar charts for each categorical column, showing value counts.\n"
        "2️⃣ **Pie Charts**: Create pie charts for columns with ≤10 unique values.\n"
        "3️⃣ **Histograms**: Show distributions of numeric columns.\n"
        "4️⃣ **Correlation**: If at least two numeric columns exist, show a correlation heatmap.\n\n"
        "⚠️ **Guidelines**:\n"
        "- Use a valid Matplotlib style from the following list: " + str(plt.style.available) + ".\n"
        "- MAKE SURE that the plots are clear, visually appealing, and use legible fonts with legends where needed.\n"
        "- Wrap all code in a single Markdown code block.\n"
        "- Create separate figures for each plot, calling `plt.show()` after each.\n"
        "- DO NOT provide explanations or comments—only return code.\n"
        "- DO NOT reapeat ant of the charts"
        "- Do not generate charts or plots for large text inputs labels, as these will not render properly. Instead, analyze such text using a dedicated, alternative visualization."
    )
    
    # Build the final prompt by prepending any custom instruction provided by the user
    if custom_instruction.strip():
        custom_prompt = custom_instruction.strip() + "\n\n"
    else:
        custom_prompt = ""
    
    final_prompt = custom_prompt + f"The DataFrame has columns: {column_names}\n{default_query}"
    
    messages = [
        {"role": "system", "content": "Provide a single code block for data analysis and visualization."},
        {"role": "user", "content": final_prompt}
    ]
    
    with st.spinner("Generating visualizations..."):
        if selected_llm == "Google Gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            result = genai.GenerativeModel('gemini-2.0-flash').generate_content(final_prompt).text
        else:
            result = openai.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7).choices[0].message.content
    
    st.toast("Visualization generated successfully!", icon="✅")
    execute_openai_code(result, df)

# New: Define call_llm function to answer column analysis questions
def call_llm(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in data analysis."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7)
    return response.choices[0].message.content

def display_app_header():
    st.title("Data Visualization with LLM 📊")
    st.markdown("**Prompt about your data, and see it visualized ✨**")

display_app_header()

with st.expander("ℹ️ App Overview", expanded=False):
    st.markdown("""
    **Key Features**:
    - Upload your file.
    - Choose an LLM (GPT-4 or Google Gemini).
    - Optionally, provide specific chart instructions.
    - Get instant visualizations.
    """)

def key_check():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OpenAI API key. Set it in your .env file.")
        st.stop()

key_check()

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
selected_llm = st.sidebar.radio("Choose AI Model", ["GPT-4", "Google Gemini"])

# Sidebar: Additional Instructions (Optional)
user_instruction = st.sidebar.text_area(
    "Optional: Provide specific chart instructions or custom prompt",
    placeholder="e.g., Show me a histogram for column X or a line chart for trend analysis."
)

# Load data only from file upload
from utils import get_data
df = get_data()

if df is not None:
    st.session_state.df = df  # Store the dataframe in session_state for later use
    with st.sidebar.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(5))
    
    # Create two tabs: one for LLM Visualizations and another for Column Analysis
    main_tabs = st.tabs(["Visualization", "Column Analysis"])
    
    # --- Visualization Tab ---
    with main_tabs[0]:
        st.header("Visualization")
        if not df.empty:
            handle_openai_query(df, ", ".join(df.columns), selected_llm, custom_instruction=user_instruction)
        else:
            st.warning("Empty dataset provided.")
    
    # --- Column Analysis Tab ---
    with main_tabs[1]:
        st.title("Column Analysis")
        st.write("Select a column from your uploaded dataset to view basic statistics and a visualization.")
        if st.session_state.df is not None:
            columns = list(st.session_state.df.columns)
            selected_column = st.selectbox("Select column for analysis", columns)
            if selected_column:
                st.subheader(f"Statistics for {selected_column}")
                column_data = st.session_state.df[selected_column]
                data_type = column_data.dtype
                
                # Compute statistics and show visualizations
                if pd.api.types.is_numeric_dtype(data_type):
                    stats = {
                        "Count": column_data.count(),
                        "Missing": column_data.isna().sum(),
                        "Mean": column_data.mean(),
                        "Median": column_data.median(),
                        "Min": column_data.min(),
                        "Max": column_data.max(),
                        "Std Dev": column_data.std()
                    }
                    stats_df = pd.DataFrame(stats.items(), columns=["Statistic", "Value"])
                    st.dataframe(stats_df, use_container_width=True)
                    
                    fig = px.histogram(
                        st.session_state.df, x=selected_column, 
                        title=f"Distribution of {selected_column}",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = column_data.value_counts().reset_index()
                    value_counts.columns = ['Value', 'Count']
                    stats = {
                        "Count": column_data.count(),
                        "Missing": column_data.isna().sum(),
                        "Unique Values": column_data.nunique(),
                        "Most Common": column_data.mode()[0] if not column_data.mode().empty else "N/A"
                    }
                    stats_df = pd.DataFrame(stats.items(), columns=["Statistic", "Value"])
                    st.dataframe(stats_df, use_container_width=True)
                    
                    st.subheader("Top Values")
                    st.dataframe(value_counts.head(10), use_container_width=True)
                    if len(value_counts) <= 20:
                        fig = px.bar(
                            value_counts.head(10), x='Value', y='Count', 
                            title=f"Top 10 values in {selected_column}",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Convert stats to serializable types (convert numpy.int64/float64 to native types)
                stats_serializable = {
                    k: int(v) if isinstance(v, (np.integer)) 
                    else float(v) if isinstance(v, (np.floating)) 
                    else v 
                    for k, v in stats.items()
                }
                
                # Additional LLM analysis: allow user to ask questions about the column data
                with st.expander("LLM Column Analysis"):
                    st.write("Ask the LLM for further insights or commentary on this column data.")
                    user_question = st.text_input("Ask LLM about this column:", key=f"llm_question_{selected_column}")
                    if st.button("Submit Question", key=f"ask_llm_{selected_column}"):
                        # Use a sample of the column data (first 50 values) and the computed stats
                        sample_data = column_data.head(50).to_list()
                        analysis_prompt = f"""You are analyzing the data for the column "{selected_column}".
Column statistics: {json.dumps(stats_serializable)}
A sample of the column data (first 50 values): {sample_data}.
Please provide additional insights, interpretation, and any recommendations based on this analysis.
User question: {user_question}
"""
                        response = call_llm(analysis_prompt)
                        st.markdown("### LLM Response")
                        st.write(response)
        else:
            st.info("Please upload an Excel file in the sidebar")
else:
    st.info("Awaiting file upload...")

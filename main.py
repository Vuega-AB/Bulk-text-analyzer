import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import re
import openai
import matplotlib.pyplot as plt
from io import BytesIO
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Set Streamlit page layout
st.set_page_config(page_title="Excel Data Visualization with LLM")

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
    with st.sidebar.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(5))
    if not df.empty:
        handle_openai_query(df, ", ".join(df.columns), selected_llm, custom_instruction=user_instruction)
    else:
        st.warning("Empty dataset provided.")

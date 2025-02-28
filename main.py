import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'suggestions_str' not in st.session_state:
    st.session_state.suggestions_str = ""
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {}
if 'selected_options' not in st.session_state:
    st.session_state.selected_options = []
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = {}  # Cache for visualization data

def extract_json(response_text):
    # Remove markdown code block markers if present
    response_text = re.sub(r"```json", "", response_text)
    response_text = re.sub(r"```", "", response_text)
    return response_text.strip()

def call_llm(prompt):
    # Configure and call the Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Set page configuration and title
st.set_page_config(page_title="Excel Data Visualization", layout="wide")
st.title("Excel Data Visualization with LLM")

# Sidebar: File upload and visualization selection
with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
    
    # Visualization options selection will appear after suggestions are generated.
    if st.session_state.suggestions:
        available_options = list(st.session_state.suggestions.keys())
        st.session_state.selected_options = st.multiselect(
            "Select visualization options", available_options, default=st.session_state.selected_options
        )

# Main App Area
st.write("This application uses a Google Generative AI model to suggest and generate visualizations based on your Excel data.")
if st.session_state.df is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())

    # When generating visualization suggestions, send the entire Excel data
    # and instruct the AI to use the exact column names from the file.
    if not st.session_state.suggestions:
        columns_list = list(st.session_state.df.columns)
        all_data = st.session_state.df.to_dict(orient='list')
        st.write(all_data)
        prompt = (
            f"Analyze the following Excel file with columns: {columns_list} and data: {all_data}. "
            "Suggest appropriate visualization options for this data in JSON format with keys such as 'bar_chart', 'pie_chart', "
            "'word_cloud', 'histogram', and 'line_chart'. Each key should map to a list of suggestions. "
            "Each suggestion should be a JSON object that includes a 'title', the required data mappings (for example, 'x_axis' and 'y_axis' for charts, "
            "or 'labels' and 'values' for pie charts), and a 'description'. "
            f"IMPORTANT: Return the exact column names as provided in the Excel file for the X-axis and the Y-axis: {columns_list}."
        )
        suggestions_str = call_llm(prompt)
        st.session_state.suggestions_str = suggestions_str
        clean_json = extract_json(suggestions_str)
        try:
            st.session_state.suggestions = json.loads(clean_json)
        except Exception as e:
            st.error("Error parsing JSON from LLM response: " + str(e))
            st.session_state.suggestions = {}

        # Filter out line_chart suggestions for feedback trends if there is only one year
        if "Year" in st.session_state.df.columns:
            unique_years = st.session_state.df["Year"].nunique()
            if unique_years <= 1 and "line_chart" in st.session_state.suggestions:
                filtered = [
                    s for s in st.session_state.suggestions["line_chart"]
                    if "feedback" not in s.get("title", "").lower() and "trend" not in s.get("title", "").lower()
                ]
                st.session_state.suggestions["line_chart"] = filtered

    with st.expander("LLM Visualization Suggestions (raw JSON)", expanded=False):
        st.code(st.session_state.suggestions_str, language="json")
    
    # If no selection has been made yet, let the user choose from the available options in the sidebar.
    if not st.session_state.selected_options:
        available_options = list(st.session_state.suggestions.keys())
        st.session_state.selected_options = st.sidebar.multiselect("Select visualization options", available_options)

    # Create a tab for each selected visualization option
    if st.session_state.selected_options:
        tabs = st.tabs(st.session_state.selected_options)
        for idx, option in enumerate(st.session_state.selected_options):
            with tabs[idx]:
                st.markdown(f"## {option.replace('_', ' ').title()}")
                for viz in st.session_state.suggestions.get(option, []):
                    st.markdown(f"### {viz.get('title', '')}")
                    st.write(viz.get("description", ""))
                    
                    # Determine which columns are required for this visualization
                    if option in ['bar_chart', 'histogram', 'line_chart']:
                        required_cols = [viz.get('x_axis', ''), viz.get('y_axis', '')]
                    elif option == 'pie_chart':
                        required_cols = [viz.get('labels', ''), viz.get('values', '')]
                    elif option == 'word_cloud':
                        required_cols = [viz.get('text', '')]
                    else:
                        required_cols = list(st.session_state.df.columns)
                    
                    # Filter out empty column names
                    required_cols = [col for col in required_cols if col]
                    if not required_cols:
                        st.error("No required columns specified for this visualization.")
                        continue

                    # Ensure the required columns exist in the dataframe.
                    missing = [col for col in required_cols if col not in st.session_state.df.columns]
                    if missing:
                        st.error(f"Required column(s) {missing} not found in the Excel file. Please verify the column names.")
                        continue

                    # Send only the needed columns to the AI for visualization data
                    data_sample = st.session_state.df[required_cols].to_dict(orient='list')
                    
                    # Build the mapping prompt using the necessary columns only and instructing to use exact column names.
                    if option in ['bar_chart', 'histogram', 'line_chart']:
                        mapping_text = (
                            f"Generate data for a {option.replace('_', ' ')} titled '{viz.get('title','')}' using the following data sample (only the columns {required_cols}): {data_sample}. "
                            "Return a JSON object with keys 'x' (a list of x values) and 'y' (a list of y values). "
                            f"Ensure that the column names match exactly as: {required_cols}."
                        )
                    elif option == 'pie_chart':
                        mapping_text = (
                            f"Generate data for a pie chart titled '{viz.get('title','')}' using the following data sample (only the columns {required_cols}): {data_sample}. "
                            "Return a JSON object with keys 'labels' and 'values'. "
                            f"Ensure that the column names match exactly as: {required_cols}."
                        )
                    elif option == 'word_cloud':
                        mapping_text = (
                            f"Generate word frequency data for a word cloud titled '{viz.get('title','')}' using the following data sample (only the column {required_cols}): {data_sample}. "
                            "Return a JSON object where keys are words and values are frequencies. "
                            f"Ensure that the column name matches exactly as: {required_cols}."
                        )
                    else:
                        mapping_text = (
                            f"Generate data for visualization '{viz.get('title','')}' using the following data sample (only the columns {required_cols}): {data_sample}. "
                            "Return a JSON object accordingly."
                        )
                    
                    # Cache the LLM call for visualization data to avoid repeated calls
                    if option not in st.session_state.viz_data:
                        st.session_state.viz_data[option] = {}
                    viz_title = viz.get("title", "Untitled")
                    if viz_title not in st.session_state.viz_data[option]:
                        viz_data_str = call_llm(mapping_text)
                        st.session_state.viz_data[option][viz_title] = viz_data_str
                    else:
                        viz_data_str = st.session_state.viz_data[option][viz_title]
                    
                    st.write("LLM Generated Data:")
                    st.code(viz_data_str, language="json")
                    
                    clean_viz_data = extract_json(viz_data_str)
                    try:
                        viz_data = json.loads(clean_viz_data)
                    except Exception as e:
                        st.error(f"Error parsing visualization data for {viz_title}: {e}")
                        continue
                    
                    # Render visualizations based on option type
                    if option in ['bar_chart', 'histogram']:
                        try:
                            df_chart = pd.DataFrame({
                                "x": viz_data.get("x", []),
                                "y": viz_data.get("y", [])
                            })
                            fig = px.bar(
                                df_chart, x="x", y="y", title=viz.get("title", ""),
                                labels={"x": viz.get("x_axis", "X-axis"), "y": viz.get("y_axis", "Y-axis")},
                                color_discrete_sequence=["#4C78A8"]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering {option} for {viz_title}: {e}")
                    elif option == 'pie_chart':
                        try:
                            fig = px.pie(
                                names=viz_data.get("labels", []), 
                                values=viz_data.get("values", []),
                                title=viz.get("title", ""),
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering pie chart for {viz_title}: {e}")
                    elif option == 'word_cloud':
                        try:
                            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
                            wc.generate_from_frequencies(viz_data)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wc, interpolation='bilinear')
                            plt.axis('off')
                            plt.title(viz.get("title", ""))
                            st.pyplot(plt)
                        except Exception as e:
                            st.error(f"Error rendering word cloud for {viz_title}: {e}")
                    elif option == 'line_chart':
                        try:
                            df_chart = pd.DataFrame({
                                "x": viz_data.get("x", []),
                                "y": viz_data.get("y", [])
                            })
                            fig = px.line(
                                df_chart, x="x", y="y", title=viz.get("title", ""),
                                labels={"x": viz.get("x_axis", "X-axis"), "y": viz.get("y_axis", "Y-axis")}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering line chart for {viz_title}: {e}")
                    else:
                        st.info("Visualization type not supported yet.")

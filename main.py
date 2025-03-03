import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import re
import json
import os
import time  # Added for delay
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import requests
from collections import Counter

# Set page config
st.set_page_config(page_title="Excel Data Visualization", layout="wide")
load_dotenv()

# API Key Handling
api_key = os.environ.get("GOOGLE_API_KEY") or st.error("Missing Google API key")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI") or st.warning("Missing OpenAI key")
together_api_key = os.environ.get("TOGETHER_API_KEY") or st.warning("Missing Together key")

# Session State Initialization
session_defaults = {
    "df": None,
    "suggestions_str": "",
    "suggestions": {},
    "selected_options": [],
    "selected_llm": "Google Gemini",
    "config": {"system_prompt": "You are a helpful assistant."}
}
for key, val in session_defaults.items():
    st.session_state.setdefault(key, val)

def generate_response_together(prompt, context, model, temp, top_p):
    headers = {"Authorization": f"Bearer {together_api_key}"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": st.session_state.config["system_prompt"]},
            {"role": "user", "content": f"{context}\n{prompt}"}
        ],
        "temperature": temp,
        "top_p": top_p
    }
    try:
        response = requests.post("https://api.together.xyz/chat/completions", headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"API Error: {str(e)}"

# Sidebar Components
with st.sidebar:
    st.header("Upload & Preview")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
    if uploaded_file:
        try:
            st.session_state.df = pd.read_excel(uploaded_file)
            st.session_state.suggestions = {}  # reset suggestions on new upload
        except Exception as e:
            st.error(f"Read error: {e}")
    
    st.session_state.selected_llm = st.selectbox(
        "LLM Model", ["Google Gemini", "OpenAI GPT-4", "Together Llama"]
    )
    
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        st.caption(f"Rows: {len(st.session_state.df)} | Columns: {list(st.session_state.df.columns)}")

# LLM Call Functions
def extract_json(response):
    return re.sub(r"```json|```", "", response).strip()

def call_llm(prompt):
    selected_llm = st.session_state.selected_llm
    if selected_llm == "Google Gemini":
        genai.configure(api_key=api_key)
        result = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt).text
    elif selected_llm == "OpenAI GPT-4":
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content
    elif selected_llm == "Together Llama":
        result = generate_response_together(
            prompt=prompt, context="", 
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            temp=0.7, top_p=1.0
        )
    time.sleep(2)  # 5 second delay between API calls
    return result

# Main Application
st.title("Excel Data Visualization with LLM")
st.write("Upload an Excel file to generate AI-powered visualizations.")

if st.session_state.df is not None:
    # First, analyze the dataset and get suggestions from the model.
    if not st.session_state.suggestions:
        with st.spinner("Analyzing data and generating suggestions..."):
            sample_data = st.session_state.df.head(100).to_csv(index=False)  # use first 100 rows as sample
            columns = list(st.session_state.df.columns)
            # Updated suggestion prompt: now only for bar_chart, pie_chart, histogram.
            prompt = f"""Analyze this dataset with columns: {columns} and sample data:
{sample_data}
Suggest visualizations in JSON format with keys: bar_chart, pie_chart, histogram.
Each key should contain a list of suggestions with:
- title (specific and descriptive)
- For bar_chart, histogram: "x_axis" and "y_axis" (exact names from: {columns})
- For pie_chart: "labels" and "values" (exact names from: {columns})
- A brief description.

Format example:
{{
  "bar_chart": [
    {{
      "title": "Sales by Region",
      "x_axis": "Region",
      "y_axis": "Total Sales",
      "description": "Compares sales figures across different regions"
    }}
  ]
}}

Return ONLY valid JSON with exact column names as provided (do not add extra words like 'count of' or 'number of')."""
            try:
                response = call_llm(prompt)
                st.session_state.suggestions = json.loads(extract_json(response))
                # Remove scatter_chart suggestions if any exist
                if "scatter_chart" in st.session_state.suggestions:
                    del st.session_state.suggestions["scatter_chart"]
            except Exception as e:
                st.error(f"Failed to parse suggestions: {str(e)}")
    
    if st.session_state.suggestions:
        st.markdown("### Analysis Complete: Visualization Suggestions")
        with st.expander("View Raw Suggestions"):
            st.code(json.dumps(st.session_state.suggestions, indent=2), language="json")
        
        # Filter out empty suggestion lists
        available_viz = [key for key, sugg in st.session_state.suggestions.items() if sugg]
        selected = st.multiselect("Choose visualizations to draw", available_viz, key="selected_viz")
        
        # Create tabs for each selected visualization type
        if selected:
            tabs = st.tabs(selected)
            for i, viz_type in enumerate(selected):
                with tabs[i]:
                    st.subheader(f"{viz_type.replace('_', ' ').title()} Visualizations")
                    for viz in st.session_state.suggestions.get(viz_type, []):
                        try:
                            # Determine required columns for this visualization type
                            if viz_type == 'bar_chart':
                                x_col = viz.get('x_axis', '')
                                y_col = viz.get('y_axis', '')
                                cols = [x_col, y_col]
                            elif viz_type == 'histogram':
                                x_col = viz.get('x_axis', '')
                                cols = [x_col]
                            elif viz_type == 'pie_chart':
                                labels = viz.get('labels', '')
                                values = viz.get('values', '')
                                cols = [labels, values]
                            else:
                                continue

                            # Check if required columns exist in dataframe
                            missing = [c for c in cols if c not in st.session_state.df.columns or not c]
                            if missing:
                                st.error(f"Missing columns: {missing}")
                                continue

                            # Build mapping prompt to call the LLM for visualization data
                            data_sample = st.session_state.df[cols].to_dict(orient='list')
                            if viz_type in ['bar_chart', 'histogram']:
                                mapping_text = (
                                    f"Generate data for a {viz_type.replace('_', ' ')} titled '{viz.get('title','')}' using the following data sample from columns {cols}: {data_sample}. "
                                    "Return ONLY a valid JSON object with exactly two keys: 'x' (a list of x values) and 'y' (a list of y values). "
                                    f"Ensure that the keys 'x' and 'y' correspond exactly to the provided column names: {cols}."
                                )
                            elif viz_type == 'pie_chart':
                                mapping_text = (
                                    f"Generate data for a pie chart titled '{viz.get('title','')}' using the following data sample from columns {cols}: {data_sample}. "
                                    "Return ONLY a valid JSON object with exactly two keys: 'labels' (a list of labels) and 'values' (a list of numeric values). "
                                    f"Ensure that the keys correspond exactly to the provided column names: {cols}."
                                )
                            else:
                                mapping_text = (
                                    f"Generate data for visualization '{viz.get('title','')}' using the following data sample from columns {cols}: {data_sample}. "
                                    "Return ONLY a valid JSON object."
                                )
                            
                            # Call the LLM to get the visualization data
                            viz_data_str = call_llm(mapping_text)
                            st.write("LLM Generated Data:")
                            st.code(viz_data_str, language="json")
                            
                            clean_viz_data = extract_json(viz_data_str)
                            try:
                                viz_data = json.loads(clean_viz_data)
                            except Exception as e:
                                st.error(f"Error parsing visualization data for {viz.get('title','Untitled')}: {e}")
                                continue
                            
                            # Render visualization using the LLM-provided data
                            if viz_type == 'bar_chart':
                                try:
                                    df_chart = pd.DataFrame({
                                        "x": viz_data.get("x", []),
                                        "y": viz_data.get("y", [])
                                    })
                                    fig = px.bar(
                                        df_chart, x="x", y="y", title=viz.get("title", ""),
                                        labels={"x": viz.get("x_axis", "X-axis"), "y": viz.get("y_axis", "Y-axis")},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error rendering bar chart for {viz.get('title','Untitled')}: {e}")
                            elif viz_type == 'histogram':
                                try:
                                    df_chart = pd.DataFrame({
                                        "x": viz_data.get("x", []),
                                        "y": viz_data.get("y", [])
                                    })
                                    fig = px.histogram(
                                        df_chart, x="x", y="y", title=viz.get("title", ""),
                                        labels={"x": viz.get("x_axis", "X-axis"), "y": "Frequency"},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error rendering histogram for {viz.get('title','Untitled')}: {e}")
                            elif viz_type == 'pie_chart':
                                try:
                                    fig = px.pie(
                                        names=viz_data.get("labels", []),
                                        values=viz_data.get("values", []),
                                        title=viz.get("title", ""),
                                        color_discrete_sequence=px.colors.qualitative.Pastel,
                                        hole=0.3
                                    )
                                    fig.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error rendering pie chart for {viz.get('title','Untitled')}: {e}")
                            
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error generating {viz_type} visualization: {e}")
else:
    st.info("Please upload an Excel file using the sidebar")
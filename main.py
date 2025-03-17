import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import os
import time  
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import requests
import numpy as np

# Set page config
st.set_page_config(page_title="Excel Data Visualization", layout="wide")
load_dotenv()

# API Key Handling
api_key = os.environ.get("GOOGLE_API_KEY") or st.error("Missing Google API key")
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.warning("Missing OpenAI key")
together_api_key = os.environ.get("TOGETHER_API_KEY") or st.warning("Missing Together key")

# Session State Initialization
session_defaults = {
    "df": None,
    "suggestions": {},
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
    time.sleep(2)  # Delay between API calls
    return result

# ─── SIDEBAR ─────────────────────────────────────────────

st.sidebar.title("Excel Data Analysis")

# File Upload remains in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
if uploaded_file:
    try:
        st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.suggestions = {}  # Reset suggestions on new upload
    except Exception as e:
        st.error(f"Read error: {e}")

# LLM Model selection
st.session_state.selected_llm = st.sidebar.selectbox(
    "LLM Model", ["Google Gemini", "OpenAI GPT-4", "Together Llama"]
)

# Show a small data preview if file is uploaded
if st.session_state.df is not None:
    st.sidebar.subheader("Data Preview")
    st.sidebar.dataframe(st.session_state.df.head(), use_container_width=True)
    st.sidebar.caption(f"Rows: {len(st.session_state.df)} | Columns: {list(st.session_state.df.columns)}")

# ─── MAIN PAGE TABS ─────────────────────────────────────────

main_tabs = st.tabs(["Visualization", "Column Analysis"])

# Visualization Tab
with main_tabs[0]:
    st.title("Excel Data Visualization with LLM")
    st.write("Upload an Excel file using the sidebar to generate AI-powered visualizations.")
    
    if st.session_state.df is not None:
        # Generate suggestions if not already available
        if not st.session_state.suggestions:
            with st.spinner("Analyzing data and generating suggestions..."):
                sample_data = st.session_state.df.head(100).to_csv(index=False)
                columns = list(st.session_state.df.columns)
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

Return ONLY valid JSON with exact column names as provided."""
                try:
                    response = call_llm(prompt)
                    st.session_state.suggestions = json.loads(extract_json(response))
                    st.session_state.suggestions.pop("scatter_chart", None)
                except Exception as e:
                    st.error(f"Failed to parse suggestions: {str(e)}")
        
        if st.session_state.suggestions:
            st.markdown("### Analysis Complete: Visualization Suggestions")
            with st.expander("View Raw Suggestions"):
                st.code(json.dumps(st.session_state.suggestions, indent=2), language="json")
            
            available_viz = [key for key, sugg in st.session_state.suggestions.items() if sugg]
            selected = st.multiselect("Choose visualizations to draw", available_viz, key="selected_viz")
            
            if selected:
                viz_tabs = st.tabs(selected)
                for i, viz_type in enumerate(selected):
                    with viz_tabs[i]:
                        st.subheader(f"{viz_type.replace('_', ' ').title()} Visualizations")
                        for viz in st.session_state.suggestions.get(viz_type, []):
                            try:
                                # Determine the required columns based on viz type
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
                                
                                # Check for missing columns
                                missing = [c for c in cols if c not in st.session_state.df.columns or not c]
                                if missing:
                                    st.error(f"Missing columns: {missing}")
                                    continue

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
                                
                                # Render visualization based on type
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
            
            st.markdown("### Custom Visualization Request")
            with st.expander("Custom Visualization"):
                custom_request = st.text_area("Enter your custom visualization request:", 
                                              placeholder="E.g., Plot a line chart showing monthly revenue trends")
                custom_chart_type = st.selectbox("Select custom chart type:", ["bar_chart", "line_chart", "pie_chart", "histogram"])
                if st.button("Generate Custom Visualization"):
                    if custom_request:
                        sample_data = st.session_state.df.head(100).to_csv(index=False)
                        columns = list(st.session_state.df.columns)
                        if custom_chart_type in ["bar_chart", "line_chart", "histogram"]:
                            key_info = "'x' (list of x values) and 'y' (list of y values)"
                        elif custom_chart_type == "pie_chart":
                            key_info = "'labels' (list of labels) and 'values' (list of numeric values)"
                        mapping_text = (
                            f"Generate data for a {custom_chart_type.replace('_', ' ')} titled 'Custom Visualization' using the following custom instructions: '{custom_request}'. "
                            f"Use the following sample data from the dataset with columns {columns}: {sample_data}. "
                            f"Return ONLY a valid JSON object with exactly two keys: {key_info}. "
                            "Ensure the keys correspond exactly to the provided column names where applicable."
                        )
                        custom_viz_data_str = call_llm(mapping_text)
                        st.write("LLM Generated Data for Custom Visualization:")
                        st.code(custom_viz_data_str, language="json")
                        clean_custom_data = extract_json(custom_viz_data_str)
                        try:
                            custom_viz_data = json.loads(clean_custom_data)
                        except Exception as e:
                            st.error(f"Error parsing custom visualization data: {e}")
                        
                        if custom_chart_type in ["bar_chart", "line_chart", "histogram"]:
                            try:
                                df_custom = pd.DataFrame({
                                    "x": custom_viz_data.get("x", []),
                                    "y": custom_viz_data.get("y", [])
                                })
                                if custom_chart_type == "bar_chart":
                                    fig_custom = px.bar(
                                        df_custom, x="x", y="y", title="Custom Visualization",
                                        labels={"x": "X-axis", "y": "Y-axis"},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                elif custom_chart_type == "line_chart":
                                    fig_custom = px.line(
                                        df_custom, x="x", y="y", title="Custom Visualization",
                                        labels={"x": "X-axis", "y": "Y-axis"},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                elif custom_chart_type == "histogram":
                                    fig_custom = px.histogram(
                                        df_custom, x="x", y="y", title="Custom Visualization",
                                        labels={"x": "X-axis", "y": "Frequency"},
                                        color_discrete_sequence=px.colors.qualitative.Pastel
                                    )
                                st.plotly_chart(fig_custom, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error rendering custom chart: {e}")
                        elif custom_chart_type == "pie_chart":
                            try:
                                fig_custom = px.pie(
                                    names=custom_viz_data.get("labels", []),
                                    values=custom_viz_data.get("values", []),
                                    title="Custom Visualization",
                                    color_discrete_sequence=px.colors.qualitative.Pastel,
                                    hole=0.3
                                )
                                fig_custom.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig_custom, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error rendering custom pie chart: {e}")
                    else:
                        st.warning("Please enter a custom visualization request.")
    else:
        st.info("Please upload an Excel file using the sidebar")
        
# Column Analysis Tab
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
                # Use st.dataframe with a set height to make the table larger
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


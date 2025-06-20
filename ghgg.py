import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from collections import defaultdict

# MCP server endpoint
MCP_URL = "http://localhost:8000/tool/analyze-data"

st.set_page_config(page_title="📊 MCP JSON Analyzer Client", layout="wide")
st.title("📊 JSON Column Analyzer via MCP Server")

# --- Helper Functions ---
def extract_numeric_values(data: Any) -> Dict[str, List[float]]:
    result = defaultdict(list)
    
    def process_value(value: Any):
        if isinstance(value, dict):
            if 'components' in value:
                components = value['components']
                if isinstance(components, list):
                    for comp in components:
                        if isinstance(comp, dict):
                            if 'data' in comp:
                                for data_item in comp['data']:
                                    if isinstance(data_item, dict):
                                        for k, v in data_item.items():
                                            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(',', '').replace('.', '').isdigit()):
                                                try:
                                                    result[k].append(float(str(v).replace(',', '')))
                                                except (ValueError, TypeError):
                                                    pass
                            for k, v in comp.items():
                                if k != 'data' and (isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(',', '').replace('.', '').isdigit())):
                                    try:
                                        result[k].append(float(str(v).replace(',', '')))
                                    except (ValueError, TypeError):
                                        pass
            for k, v in value.items():
                if k != 'components':
                    process_value(v)
        elif isinstance(value, list):
            for v in value:
                process_value(v)
        else:
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace(',', '').replace('.', '').isdigit()):
                try:
                    result['value'].append(float(str(value).replace(',', '')))
                except (ValueError, TypeError):
                    pass
    
    process_value(data)
    return dict(result)

def create_dataframe_from_numeric_values(all_numeric_values: Dict[str, List[float]]) -> pd.DataFrame:
    max_length = max(len(values) for values in all_numeric_values.values())
    padded_values = {k: v + [None] * (max_length - len(v)) for k, v in all_numeric_values.items()}
    return pd.DataFrame(padded_values)

def prepare_data_for_server(df: pd.DataFrame) -> List[Dict]:
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            value = row[col]
            record[col] = None if pd.isna(value) else value
        records.append(record)
    return records

# --- Main UI ---
uploaded_file = st.file_uploader("📁 Upload a JSON file", type=["json"])

if uploaded_file:
    try:
        json_data = json.load(uploaded_file)
        all_numeric_values = extract_numeric_values(json_data)
        
        st.subheader("🔍 Extracted Columns")
        for col, values in all_numeric_values.items():
            st.write(f"- {col}: {len(values)} values")
        
        df = create_dataframe_from_numeric_values(all_numeric_values)
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if numeric_columns:
            st.subheader("📊 Available Numeric Columns")
            summary = pd.DataFrame({
                "Column": numeric_columns,
                "Count": [df[col].count() for col in numeric_columns],
                "Mean": [df[col].mean() for col in numeric_columns],
                "Min": [df[col].min() for col in numeric_columns],
                "Max": [df[col].max() for col in numeric_columns],
            })
            st.dataframe(summary, use_container_width=True)

            selected_column = st.selectbox("🔢 Select a numeric column to analyze", numeric_columns)
            operation = st.selectbox("⚙️ Select operation", ["sum", "mean", "average", "median", "min", "max", "count"])

            if st.button("🚀 Run Analysis"):
                try:
                    server_data = prepare_data_for_server(df)
                    response = requests.post(MCP_URL, json={
                        "data": server_data,
                        "column": selected_column,
                        "operation": operation
                    })
                    result = response.json()
                    if result["status"] == "success":
                        st.success(f"✅ {operation.title()} of '{selected_column}': {result['value']:.2f}")
                        clean_series = df[selected_column].dropna()
                        if not clean_series.empty:
                            st.subheader("📈 Histogram")
                            st.bar_chart(clean_series.value_counts().sort_index())
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
        else:
            st.warning("No numeric columns found in uploaded data.")
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file format.")
else:
    st.info("📤 Please upload a JSON file to begin analysis.")

# health_graph_generator.py
# Separate file for all graph-related functionality

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class HealthGraphGenerator:
    """Graph generation capabilities for health data"""
    
    def __init__(self):
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71', 
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#9b59b6',
            'success': '#27ae60'
        }
        logger.info("🎨 HealthGraphGenerator initialized")
    
    def detect_graph_request(self, user_query: str) -> Dict[str, Any]:
        """Detect if user is requesting a graph/visualization"""
        graph_keywords = [
            'graph', 'chart', 'plot', 'visualize', 'visualization', 'show me a',
            'timeline', 'trend', 'over time', 'histogram', 'bar chart', 'line chart',
            'scatter plot', 'pie chart', 'heatmap', 'dashboard', 'visual'
        ]
        
        query_lower = user_query.lower()
        has_graph_keyword = any(keyword in query_lower for keyword in graph_keywords)
        
        if not has_graph_keyword:
            return {"is_graph_request": False}
        
        # Determine graph type
        graph_type = "timeline"  # default
        
        if any(word in query_lower for word in ['timeline', 'over time', 'chronological']):
            graph_type = "timeline"
        elif any(word in query_lower for word in ['bar chart', 'bar graph', 'count']):
            graph_type = "bar"
        elif any(word in query_lower for word in ['pie chart', 'pie graph', 'distribution']):
            graph_type = "pie"
        elif any(word in query_lower for word in ['risk', 'factors', 'assessment', 'dashboard']):
            graph_type = "risk_dashboard"
        elif any(word in query_lower for word in ['medication', 'drug', 'pharmacy']):
            graph_type = "medication_timeline"
        elif any(word in query_lower for word in ['diagnosis', 'condition', 'medical']):
            graph_type = "diagnosis_timeline"
        
        return {
            "is_graph_request": True,
            "graph_type": graph_type,
            "query": user_query
        }
    
    def generate_medication_timeline(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication timeline visualization"""
        try:
            pharmacy_extraction = chat_context.get('pharmacy_extraction', {})
            ndc_records = pharmacy_extraction.get('ndc_records', [])
            
            if not ndc_records:
                return "No pharmacy data available for visualization."
            
            # Prepare data for timeline
            med_data = []
            for record in ndc_records:
                rx_date = record.get('rx_filled_dt')
                med_name = record.get('lbl_nm', 'Unknown Medication')
                ndc_code = record.get('ndc', 'Unknown')
                
                if rx_date:
                    try:
                        # Parse date (handle different formats)
                        if isinstance(rx_date, str):
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                                try:
                                    parsed_date = datetime.strptime(rx_date, fmt)
                                    break
                                except:
                                    continue
                            else:
                                parsed_date = datetime.now()
                        else:
                            parsed_date = datetime.now()
                        
                        med_data.append({
                            'date': parsed_date,
                            'medication': med_name,
                            'ndc': ndc_code
                        })
                    except:
                        continue
            
            if not med_data:
                return "No valid medication timeline data found."
            
            # Create DataFrame
            df = pd.DataFrame(med_data)
            df = df.sort_values('date')
            
            # Create scatter plot timeline
            fig = px.scatter(
                df,
                x='date',
                y='medication',
                title='💊 Medication Timeline',
                labels={'date': 'Date', 'medication': 'Medication'},
                hover_data=['ndc']
            )
            
            fig.update_traces(marker=dict(size=12, symbol='circle'))
            fig.update_layout(
                height=600,
                xaxis_title="Date",
                yaxis_title="Medications",
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            return f"""
## 💊 Medication Timeline

{html_str}

**Summary:** Found {len(med_data)} medication records spanning from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}.

**Key Medications:**
{chr(10).join([f"• {med}" for med in df['medication'].unique()[:10]])}
"""
            
        except Exception as e:
            logger.error(f"Error generating medication timeline: {e}")
            return f"Error generating medication timeline: {str(e)}"
    
    def generate_diagnosis_timeline(self, chat_context: Dict[str, Any]) -> str:
        """Generate diagnosis timeline visualization"""
        try:
            medical_extraction = chat_context.get('medical_extraction', {})
            hlth_srvc_records = medical_extraction.get('hlth_srvc_records', [])
            
            if not hlth_srvc_records:
                return "No medical diagnosis data available for visualization."
            
            # Prepare data for timeline
            diag_data = []
            for record in hlth_srvc_records:
                claim_date = record.get('clm_rcvd_dt')
                service_code = record.get('hlth_srvc_cd')
                diagnosis_codes = record.get('diagnosis_codes', [])
                
                if claim_date and diagnosis_codes:
                    try:
                        # Parse date
                        if isinstance(claim_date, str):
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                                try:
                                    parsed_date = datetime.strptime(claim_date, fmt)
                                    break
                                except:
                                    continue
                            else:
                                parsed_date = datetime.now()
                        else:
                            parsed_date = datetime.now()
                        
                        for diag in diagnosis_codes:
                            diag_code = diag.get('code', 'Unknown')
                            diag_data.append({
                                'date': parsed_date,
                                'diagnosis_code': diag_code,
                                'service_code': service_code
                            })
                    except:
                        continue
            
            if not diag_data:
                return "No valid diagnosis timeline data found."
            
            # Create DataFrame
            df = pd.DataFrame(diag_data)
            df = df.sort_values('date')
            
            # Count diagnoses by month
            df['month'] = df['date'].dt.to_period('M')
            monthly_counts = df.groupby('month').size().reset_index(name='count')
            monthly_counts['month_str'] = monthly_counts['month'].astype(str)
            
            # Create bar chart
            fig = px.bar(
                monthly_counts,
                x='month_str',
                y='count',
                title='🏥 Medical Diagnoses Over Time',
                labels={'month_str': 'Month', 'count': 'Number of Diagnoses'},
                color='count',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Month",
                yaxis_title="Number of Diagnoses",
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            # Top diagnoses
            top_diagnoses = df['diagnosis_code'].value_counts().head(10)
            
            return f"""
## 🏥 Medical Diagnosis Timeline

{html_str}

**Summary:** Found {len(diag_data)} diagnosis records spanning from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}.

**Top Diagnosis Codes:**
{chr(10).join([f"• {code}: {count} occurrences" for code, count in top_diagnoses.items()])}
"""
            
        except Exception as e:
            logger.error(f"Error generating diagnosis timeline: {e}")
            return f"Error generating diagnosis timeline: {str(e)}"
    
    def generate_risk_dashboard(self, chat_context: Dict[str, Any]) -> str:
        """Generate comprehensive risk assessment dashboard"""
        try:
            entity_extraction = chat_context.get('entity_extraction', {})
            heart_attack_prediction = chat_context.get('heart_attack_prediction', {})
            
            # Prepare risk data
            risk_factors = {
                'Diabetes': 1 if entity_extraction.get('diabetics', 'no') == 'yes' else 0,
                'High Blood Pressure': 1 if entity_extraction.get('blood_pressure', 'unknown') in ['managed', 'diagnosed'] else 0,
                'Smoking': 1 if entity_extraction.get('smoking', 'no') == 'yes' else 0,
                'Alcohol Use': 1 if entity_extraction.get('alcohol', 'no') == 'yes' else 0
            }
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk Factors Distribution', 'Heart Attack Risk Score', 'Risk Factor Breakdown', 'Health Conditions'),
                specs=[[{"type": "pie"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Risk factors pie chart
            risk_labels = list(risk_factors.keys())
            risk_values = list(risk_factors.values())
            
            fig.add_trace(
                go.Pie(labels=risk_labels, values=risk_values, name="Risk Factors"),
                row=1, col=1
            )
            
            # Heart attack risk gauge
            heart_risk_score = heart_attack_prediction.get('raw_risk_score', 0.0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=heart_risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Heart Attack Risk %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 25], 'color': "lightgreen"},
                               {'range': [25, 50], 'color': "yellow"},
                               {'range': [50, 75], 'color': "orange"},
                               {'range': [75, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 75}}
                ),
                row=1, col=2
            )
            
            # Risk factor breakdown
            colors = ['red' if val == 1 else 'lightgray' for val in risk_values]
            fig.add_trace(
                go.Bar(x=risk_labels, y=risk_values, name="Risk Factors", marker_color=colors),
                row=2, col=1
            )
            
            # Medical conditions count
            medical_conditions = entity_extraction.get('medical_conditions', [])
            condition_counts = {'Identified': len(medical_conditions), 'Not Identified': max(0, 5 - len(medical_conditions))}
            
            fig.add_trace(
                go.Bar(x=list(condition_counts.keys()), y=list(condition_counts.values()), 
                       name="Conditions", marker_color=['blue', 'lightblue']),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="📊 Comprehensive Health Risk Dashboard")
            
            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            return f"""
## 📊 Health Risk Assessment Dashboard

{html_str}

**Risk Summary:**
- **Heart Attack Risk:** {heart_attack_prediction.get('risk_display', 'Not calculated')}
- **Active Risk Factors:** {sum(risk_values)} out of {len(risk_values)}
- **Medical Conditions:** {len(medical_conditions)} identified conditions

**Risk Factors Present:**
{chr(10).join([f"• {factor}" for factor, present in risk_factors.items() if present])}

**Recommendations:**
- Regular monitoring of identified risk factors
- Lifestyle modifications based on risk profile
- Continued medical follow-up as appropriate
"""
            
        except Exception as e:
            logger.error(f"Error generating risk dashboard: {e}")
            return f"Error generating risk dashboard: {str(e)}"
    
    def generate_medication_distribution(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication distribution pie chart"""
        try:
            pharmacy_extraction = chat_context.get('pharmacy_extraction', {})
            ndc_records = pharmacy_extraction.get('ndc_records', [])
            
            if not ndc_records:
                return "No pharmacy data available for medication distribution."
            
            # Count medications
            med_counts = {}
            for record in ndc_records:
                med_name = record.get('lbl_nm', 'Unknown Medication')
                med_counts[med_name] = med_counts.get(med_name, 0) + 1
            
            # Create pie chart
            fig = px.pie(
                values=list(med_counts.values()),
                names=list(med_counts.keys()),
                title='💊 Medication Distribution'
            )
            
            fig.update_layout(height=600)
            
            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            return f"""
## 💊 Medication Distribution

{html_str}

**Summary:** Total of {len(med_counts)} unique medications with {sum(med_counts.values())} total prescriptions.

**Top Medications:**
{chr(10).join([f"• {med}: {count} prescriptions" for med, count in sorted(med_counts.items(), key=lambda x: x[1], reverse=True)[:10]])}
"""
            
        except Exception as e:
            logger.error(f"Error generating medication distribution: {e}")
            return f"Error generating medication distribution: {str(e)}"
    
    def generate_comprehensive_health_overview(self, chat_context: Dict[str, Any]) -> str:
        """Generate a comprehensive health overview with multiple charts"""
        try:
            # Create a comprehensive dashboard with multiple visualizations
            entity_extraction = chat_context.get('entity_extraction', {})
            medical_extraction = chat_context.get('medical_extraction', {})
            pharmacy_extraction = chat_context.get('pharmacy_extraction', {})
            
            # Create subplots for comprehensive overview
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Health Risk Factors', 'Medication Count Over Time',
                    'Top Medical Conditions', 'Pharmacy vs Medical Claims',
                    'Health Score Overview', 'Risk Assessment Summary'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "indicator"}, {"type": "bar"}]
                ]
            )
            
            # Health risk factors
            risk_factors = {
                'Diabetes': 1 if entity_extraction.get('diabetics', 'no') == 'yes' else 0,
                'High BP': 1 if entity_extraction.get('blood_pressure', 'unknown') in ['managed', 'diagnosed'] else 0,
                'Smoking': 1 if entity_extraction.get('smoking', 'no') == 'yes' else 0,
                'Alcohol': 1 if entity_extraction.get('alcohol', 'no') == 'yes' else 0
            }
            
            fig.add_trace(
                go.Bar(x=list(risk_factors.keys()), y=list(risk_factors.values()), 
                       name="Risk Factors", marker_color='red'),
                row=1, col=1
            )
            
            # Medication timeline (simplified)
            ndc_records = pharmacy_extraction.get('ndc_records', [])
            if ndc_records:
                med_dates = []
                for record in ndc_records[:10]:  # Limit for performance
                    rx_date = record.get('rx_filled_dt')
                    if rx_date:
                        try:
                            parsed_date = datetime.strptime(rx_date, '%Y-%m-%d')
                            med_dates.append(parsed_date)
                        except:
                            continue
                
                if med_dates:
                    df_meds = pd.DataFrame({'dates': med_dates})
                    df_meds['month'] = df_meds['dates'].dt.to_period('M')
                    monthly_med_counts = df_meds.groupby('month').size().reset_index(name='count')
                    
                    fig.add_trace(
                        go.Scatter(x=monthly_med_counts['month'].astype(str), 
                                 y=monthly_med_counts['count'],
                                 mode='lines+markers', name="Medications"),
                        row=1, col=2
                    )
            
            # Top medical conditions
            medical_conditions = entity_extraction.get('medical_conditions', [])
            if medical_conditions:
                condition_names = [cond.split('(')[0].strip() for cond in medical_conditions[:5]]
                condition_counts = [1] * len(condition_names)
                
                fig.add_trace(
                    go.Bar(x=condition_names, y=condition_counts,
                           name="Conditions", marker_color='blue'),
                    row=2, col=1
                )
            
            # Claims distribution
            medical_count = len(medical_extraction.get('hlth_srvc_records', []))
            pharmacy_count = len(pharmacy_extraction.get('ndc_records', []))
            
            fig.add_trace(
                go.Pie(labels=['Medical Claims', 'Pharmacy Claims'], 
                       values=[medical_count, pharmacy_count]),
                row=2, col=2
            )
            
            # Overall health score
            health_score = 85 - (sum(risk_factors.values()) * 10)  # Simple calculation
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    title={'text': "Health Score"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "green"},
                           'steps': [
                               {'range': [0, 50], 'color': "red"},
                               {'range': [50, 75], 'color': "yellow"},
                               {'range': [75, 100], 'color': "green"}]}
                ),
                row=3, col=1
            )
            
            # Risk summary
            risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_counts = [2, 1, sum(risk_factors.values())]
            
            fig.add_trace(
                go.Bar(x=risk_categories, y=risk_counts,
                       marker_color=['green', 'orange', 'red']),
                row=3, col=2
            )
            
            fig.update_layout(height=1200, title_text="🏥 Comprehensive Health Overview Dashboard")
            
            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            return f"""
## 🏥 Comprehensive Health Overview

{html_str}

**Key Insights:**
- **Health Score:** {health_score}/100
- **Active Risk Factors:** {sum(risk_factors.values())}/4
- **Medical Claims:** {medical_count} records
- **Pharmacy Claims:** {pharmacy_count} records
- **Conditions Identified:** {len(medical_conditions)}

**Available Specific Visualizations:**
- `show medication timeline` - Detailed medication history
- `create diagnosis chart` - Medical diagnosis trends
- `generate risk dashboard` - Focused risk assessment
"""
            
        except Exception as e:
            logger.error(f"Error generating comprehensive overview: {e}")
            return f"Error generating comprehensive health overview: {str(e)}"

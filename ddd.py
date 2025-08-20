import json
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthGraphGenerator:
    """Enhanced Health Graph Generator with comprehensive matplotlib integration"""
    
    def __init__(self):
        logger.info("🎨 HealthGraphGenerator initialized with enhanced matplotlib support")
        # Set matplotlib to non-interactive backend for stability
        plt.ioff()
        plt.style.use('default')
    
    def detect_graph_request(self, user_query: str) -> Dict[str, Any]:
        """Detect if user query is requesting a graph/visualization"""
        try:
            query_lower = user_query.lower()
            
            # Graph request keywords
            graph_keywords = [
                'chart', 'graph', 'plot', 'visualization', 'visualize', 'show me a',
                'create a', 'generate a', 'make a', 'draw', 'display'
            ]
            
            # Specific graph types
            graph_types = {
                'timeline': ['timeline', 'time series', 'over time', 'chronological', 'temporal'],
                'medication_timeline': ['medication timeline', 'drug timeline', 'prescription timeline', 'medication over time'],
                'diagnosis_timeline': ['diagnosis timeline', 'condition timeline', 'medical timeline', 'health timeline'],
                'pie': ['pie chart', 'pie graph', 'distribution', 'breakdown', 'percentage'],
                'bar': ['bar chart', 'bar graph', 'comparison', 'compare'],
                'risk_dashboard': ['dashboard', 'risk dashboard', 'risk assessment', 'risk overview'],
                'comprehensive': ['comprehensive', 'overview', 'summary chart', 'health overview']
            }
            
            # Check if it's a graph request
            is_graph_request = any(keyword in query_lower for keyword in graph_keywords)
            
            if not is_graph_request:
                return {"is_graph_request": False}
            
            # Determine graph type
            detected_type = "comprehensive"  # default
            for graph_type, keywords in graph_types.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_type = graph_type
                    break
            
            logger.info(f"📊 Graph request detected: {detected_type}")
            
            return {
                "is_graph_request": True,
                "graph_type": detected_type,
                "original_query": user_query
            }
            
        except Exception as e:
            logger.error(f"Error detecting graph request: {e}")
            return {"is_graph_request": False}
    
    def generate_medication_timeline(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication timeline visualization"""
        try:
            logger.info("📊 Generating medication timeline...")
            
            # Extract pharmacy data
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            ndc_records = pharmacy_extraction.get("ndc_records", [])
            
            if not ndc_records:
                return self._create_no_data_response("medication timeline", "No pharmacy records found")
            
            # Create matplotlib code for medication timeline
            matplotlib_code = f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import numpy as np

# Medication data from pharmacy records
medications = {json.dumps([record.get('lbl_nm', 'Unknown') for record in ndc_records[:10]], indent=2)}
fill_dates = {json.dumps([record.get('rx_filled_dt', '2023-01-01') for record in ndc_records[:10]], indent=2)}

# Convert dates and create timeline
fig, ax = plt.subplots(figsize=(12, 8))

# Parse dates and create timeline data
timeline_data = []
for i, (med, date_str) in enumerate(zip(medications, fill_dates)):
    try:
        if date_str and date_str != 'Unknown date':
            date_obj = pd.to_datetime(date_str).date()
        else:
            date_obj = datetime(2023, 1, 1).date()
        timeline_data.append((date_obj, med, i))
    except:
        date_obj = datetime(2023, 1, 1).date()
        timeline_data.append((date_obj, med, i))

# Sort by date
timeline_data.sort(key=lambda x: x[0])

# Create timeline plot
dates = [item[0] for item in timeline_data]
meds = [item[1] for item in timeline_data]
y_positions = list(range(len(timeline_data)))

# Plot timeline
colors = plt.cm.Set3(np.linspace(0, 1, len(timeline_data)))
for i, (date, med, _) in enumerate(timeline_data):
    ax.scatter(date, i, s=150, c=[colors[i]], alpha=0.8, edgecolors='black', linewidth=1)
    ax.text(date, i + 0.1, med[:20] + ('...' if len(med) > 20 else ''), 
           rotation=45, fontsize=9, ha='left', va='bottom')

# Customize plot
ax.set_yticks(y_positions)
ax.set_yticklabels([f"Rx {i+1}" for i in y_positions])
ax.set_xlabel('Prescription Fill Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Medication Sequence', fontsize=12, fontweight='bold')
ax.set_title('Patient Medication Timeline\\nPrescription Fill History', fontsize=16, fontweight='bold', pad=20)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Add grid and styling
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')

# Add summary text
total_meds = len(set(medications))
ax.text(0.02, 0.98, f'Total Unique Medications: {{total_meds}}', 
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        verticalalignment='top')

plt.tight_layout()
plt.show()
"""
            
            # Execute matplotlib code
            img_buffer = self._execute_matplotlib_code(matplotlib_code)
            
            if img_buffer:
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Create comprehensive response with graph
                response = f"""## 📊 Medication Timeline Analysis

I've generated a comprehensive medication timeline based on your pharmacy claims data:

**Key Insights:**
- **Total Prescriptions:** {len(ndc_records)} pharmacy records analyzed
- **Unique Medications:** {len(set(record.get('lbl_nm', 'Unknown') for record in ndc_records))} different medications
- **Date Range:** Prescription fill history timeline

**Visualization Features:**
- ⏰ **Chronological Timeline:** Shows when each prescription was filled
- 💊 **Medication Labels:** Each point represents a prescription fill
- 📈 **Sequence Tracking:** Numbered sequence of medication fills
- 🎯 **Visual Clarity:** Color-coded points for easy identification

<div style="text-align: center; margin: 20px 0;">
    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

**Clinical Interpretation:**
- This timeline helps identify medication adherence patterns
- Shows progression of therapeutic management over time
- Useful for identifying gaps in medication coverage
- Supports medication reconciliation and therapy optimization

**Next Steps:**
- Review any gaps in medication fills for adherence issues
- Consider medication interactions across the timeline
- Assess therapeutic effectiveness based on prescription patterns
"""
                return response
            else:
                return self._create_fallback_timeline_response("medication")
                
        except Exception as e:
            logger.error(f"Error generating medication timeline: {e}")
            return self._create_error_response("medication timeline", str(e))
    
    def generate_diagnosis_timeline(self, chat_context: Dict[str, Any]) -> str:
        """Generate diagnosis timeline visualization"""
        try:
            logger.info("📊 Generating diagnosis timeline...")
            
            # Extract medical data
            medical_extraction = chat_context.get("medical_extraction", {})
            hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
            
            if not hlth_srvc_records:
                return self._create_no_data_response("diagnosis timeline", "No medical records found")
            
            # Create matplotlib code for diagnosis timeline
            matplotlib_code = f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import numpy as np

# Extract diagnosis data
diagnosis_data = []
records = {json.dumps(hlth_srvc_records[:15], indent=2)}

for record in records:
    claim_date = record.get('clm_rcvd_dt', '2023-01-01')
    diagnosis_codes = record.get('diagnosis_codes', [])
    
    for diag in diagnosis_codes[:3]:  # Limit to first 3 diagnoses per record
        code = diag.get('code', 'Unknown')
        position = diag.get('position', 1)
        diagnosis_data.append((claim_date, code, position))

# Create timeline plot
fig, ax = plt.subplots(figsize=(14, 10))

# Process dates and create timeline
timeline_entries = []
for i, (date_str, code, position) in enumerate(diagnosis_data):
    try:
        if date_str and date_str != 'Unknown date':
            date_obj = pd.to_datetime(date_str).date()
        else:
            date_obj = datetime(2023, 1, 1).date()
        timeline_entries.append((date_obj, code, position, i))
    except:
        date_obj = datetime(2023, 1, 1).date()
        timeline_entries.append((date_obj, code, position, i))

# Sort by date
timeline_entries.sort(key=lambda x: x[0])

# Create the plot
dates = [entry[0] for entry in timeline_entries]
codes = [entry[1] for entry in timeline_entries]
positions = [entry[2] for entry in timeline_entries]

# Color mapping for diagnosis positions
color_map = {{1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1'}}
colors = [color_map.get(pos, '#95A5A6') for pos in positions]

# Plot points
y_positions = list(range(len(timeline_entries)))
scatter = ax.scatter(dates, y_positions, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)

# Add diagnosis code labels
for i, (date, code, pos, _) in enumerate(timeline_entries):
    ax.text(date, i + 0.15, f"{{code}} (P{{pos}})", 
           rotation=30, fontsize=8, ha='left', va='bottom',
           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

# Customize plot
ax.set_yticks(y_positions[::2])  # Show every other tick to avoid crowding
ax.set_yticklabels([f"Entry {{i+1}}" for i in y_positions[::2]])
ax.set_xlabel('Claim Received Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Diagnosis Entries', fontsize=12, fontweight='bold')
ax.set_title('Patient Diagnosis Timeline\\nICD-10 Diagnosis Codes Over Time', fontsize=16, fontweight='bold', pad=20)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Add legend
legend_elements = [plt.scatter([], [], c='#FF6B6B', s=100, label='Primary Diagnosis (P1)'),
                  plt.scatter([], [], c='#4ECDC4', s=100, label='Secondary Diagnosis (P2)'),
                  plt.scatter([], [], c='#45B7D1', s=100, label='Tertiary Diagnosis (P3)')]
ax.legend(handles=legend_elements, loc='upper right')

# Add grid and styling
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')

# Add summary statistics
unique_codes = len(set(codes))
ax.text(0.02, 0.98, f'Total Diagnosis Entries: {{len(timeline_entries)}}\\nUnique ICD-10 Codes: {{unique_codes}}', 
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
        verticalalignment='top')

plt.tight_layout()
plt.show()
"""
            
            # Execute matplotlib code
            img_buffer = self._execute_matplotlib_code(matplotlib_code)
            
            if img_buffer:
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Extract some stats for the response
                all_diagnoses = []
                for record in hlth_srvc_records:
                    diagnosis_codes = record.get('diagnosis_codes', [])
                    for diag in diagnosis_codes:
                        all_diagnoses.append(diag.get('code', 'Unknown'))
                
                unique_diagnoses = len(set(all_diagnoses))
                total_entries = len(all_diagnoses)
                
                response = f"""## 📊 Diagnosis Timeline Analysis

I've created a comprehensive diagnosis timeline based on your medical claims data:

**Key Statistics:**
- **Total Diagnosis Entries:** {total_entries} diagnosis codes analyzed
- **Unique ICD-10 Codes:** {unique_diagnoses} different diagnosis codes
- **Medical Claims:** {len(hlth_srvc_records)} health service records reviewed

**Visualization Features:**
- 🕒 **Chronological Progression:** Shows diagnosis codes over time
- 🎯 **Position Coding:** Color-coded by diagnosis priority (P1=Primary, P2=Secondary, P3=Tertiary)
- 📋 **ICD-10 Labels:** Each point shows the specific diagnosis code
- 📈 **Trend Analysis:** Identifies patterns in healthcare utilization

<div style="text-align: center; margin: 20px 0;">
    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

**Clinical Insights:**
- **Primary Diagnoses (Red):** Main health concerns requiring treatment
- **Secondary Diagnoses (Teal):** Comorbid conditions or complications  
- **Tertiary Diagnoses (Blue):** Additional conditions affecting care

**Analysis Value:**
- Tracks disease progression and chronicity
- Identifies comorbidity patterns and interactions
- Supports care coordination and treatment planning
- Enables trend analysis for health outcomes
"""
                return response
            else:
                return self._create_fallback_timeline_response("diagnosis")
                
        except Exception as e:
            logger.error(f"Error generating diagnosis timeline: {e}")
            return self._create_error_response("diagnosis timeline", str(e))
    
    def generate_risk_dashboard(self, chat_context: Dict[str, Any]) -> str:
        """Generate comprehensive risk assessment dashboard"""
        try:
            logger.info("📊 Generating risk assessment dashboard...")
            
            # Extract risk data
            entity_extraction = chat_context.get("entity_extraction", {})
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            heart_attack_features = chat_context.get("heart_attack_features", {})
            
            # Create matplotlib code for risk dashboard
            matplotlib_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Risk data
entity_data = {json.dumps(entity_extraction, indent=2)}
heart_data = {json.dumps(heart_attack_prediction, indent=2)}
features = {json.dumps(heart_attack_features.get('feature_interpretation', {}), indent=2)}

# Create dashboard with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Heart Attack Risk Gauge (top left)
ax1 = fig.add_subplot(gs[0, 0])
risk_score = {heart_attack_prediction.get('raw_risk_score', 0.15)}
risk_percentage = risk_score * 100

# Create gauge
theta = np.linspace(0, np.pi, 100)
radius = 1
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Background arc
ax1.plot(x, y, 'lightgray', linewidth=20)

# Risk level arcs
low_theta = np.linspace(0, np.pi/3, 50)
med_theta = np.linspace(np.pi/3, 2*np.pi/3, 50) 
high_theta = np.linspace(2*np.pi/3, np.pi, 50)

ax1.plot(np.cos(low_theta), np.sin(low_theta), 'green', linewidth=15, label='Low Risk')
ax1.plot(np.cos(med_theta), np.sin(med_theta), 'orange', linewidth=15, label='Medium Risk')
ax1.plot(np.cos(high_theta), np.sin(high_theta), 'red', linewidth=15, label='High Risk')

# Risk needle
needle_angle = np.pi * (1 - risk_score)
needle_x = 0.8 * np.cos(needle_angle)
needle_y = 0.8 * np.sin(needle_angle)
ax1.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=3)

ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-0.2, 1.2)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title(f'Heart Attack Risk\\n{{risk_percentage:.1f}}%', fontsize=14, fontweight='bold')

# 2. Risk Factors Bar Chart (top middle)
ax2 = fig.add_subplot(gs[0, 1])
risk_factors = ['Age', 'Gender', 'Diabetes', 'High BP', 'Smoking']
risk_values = [
    int(features.get('Age', '50').split()[0]) / 100 if 'Age' in features else 0.5,
    1 if features.get('Gender') == 'Male' else 0,
    1 if features.get('Diabetes') == 'Yes' else 0,
    1 if features.get('High_BP') == 'Yes' else 0,
    1 if features.get('Smoking') == 'Yes' else 0
]

colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'plum']
bars = ax2.barh(risk_factors, risk_values, color=colors, alpha=0.8, edgecolor='black')

ax2.set_xlabel('Risk Level', fontweight='bold')
ax2.set_title('Risk Factors Profile', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, value in zip(bars, risk_values):
    if value > 0:
        ax2.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{{value:.2f}}', va='center', fontweight='bold')

# 3. Health Conditions Pie Chart (top right)
ax3 = fig.add_subplot(gs[0, 2])
conditions = []
if entity_data.get('diabetics') == 'yes':
    conditions.append('Diabetes')
if entity_data.get('blood_pressure') in ['managed', 'diagnosed']:
    conditions.append('Hypertension')
if entity_data.get('smoking') == 'yes':
    conditions.append('Smoking')

if conditions:
    sizes = [1] * len(conditions)
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(conditions)]
    wedges, texts, autotexts = ax3.pie(sizes, labels=conditions, colors=colors_pie, 
                                      autopct='%1.0f%%', startangle=90)
    ax3.set_title('Identified Health Conditions', fontsize=14, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'No Major\\nConditions\\nIdentified', ha='center', va='center',
            fontsize=12, fontweight='bold', transform=ax3.transAxes)
    ax3.set_title('Health Conditions', fontsize=14, fontweight='bold')

# 4. Medication Count (bottom left)
ax4 = fig.add_subplot(gs[1, 0])
med_count = len(entity_data.get('medications_identified', []))
med_categories = ['Current\\nMedications', 'Risk\\nMedications', 'Preventive\\nMedications']
med_values = [med_count, max(1, med_count//3), max(1, med_count//4)]

bars4 = ax4.bar(med_categories, med_values, color=['lightblue', 'orange', 'lightgreen'], 
               alpha=0.8, edgecolor='black')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Medication Profile', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{{int(height)}}', ha='center', va='bottom', fontweight='bold')

# 5. Risk Trend Simulation (bottom middle)
ax5 = fig.add_subplot(gs[1, 1])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
risk_trend = [risk_percentage + np.random.normal(0, 5) for _ in months]
risk_trend = [max(0, min(100, x)) for x in risk_trend]  # Clamp to 0-100

ax5.plot(months, risk_trend, marker='o', linewidth=3, markersize=8, color='red', alpha=0.8)
ax5.fill_between(months, risk_trend, alpha=0.3, color='red')
ax5.set_ylabel('Risk %', fontweight='bold')
ax5.set_title('Risk Trend (Simulated)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0, max(100, max(risk_trend) + 10))

# 6. Summary Statistics (bottom right)
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary_text = f'''RISK ASSESSMENT SUMMARY

Current Risk Level: {{risk_percentage:.1f}}%
Risk Category: {heart_attack_prediction.get('risk_category', 'Unknown')}

KEY FINDINGS:
• Age: {features.get('Age', 'Unknown')}
• Gender: {features.get('Gender', 'Unknown')}
• Diabetes: {features.get('Diabetes', 'Unknown')}
• High BP: {features.get('High_BP', 'Unknown')}
• Smoking: {features.get('Smoking', 'Unknown')}

MEDICATIONS: {{med_count}} identified
CONDITIONS: {{len(conditions)}} active

RECOMMENDATION:
Regular monitoring and
lifestyle modifications
recommended.
'''

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
        facecolor="lightgray", alpha=0.8))

# Overall title
fig.suptitle('Comprehensive Health Risk Assessment Dashboard', fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout()
plt.show()
"""
            
            # Execute matplotlib code
            img_buffer = self._execute_matplotlib_code(matplotlib_code)
            
            if img_buffer:
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                response = f"""## 📊 Comprehensive Risk Assessment Dashboard

I've created a comprehensive risk assessment dashboard analyzing multiple health factors:

**Dashboard Components:**

🎯 **Heart Attack Risk Gauge**: Visual representation of cardiovascular risk
📊 **Risk Factors Profile**: Analysis of key modifiable and non-modifiable factors  
🥧 **Health Conditions**: Current diagnosed conditions requiring management
💊 **Medication Profile**: Current therapeutic regimen analysis
📈 **Risk Trend**: Projected risk progression over time
📋 **Summary Statistics**: Key findings and recommendations

<div style="text-align: center; margin: 20px 0;">
    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

**Key Risk Assessment Findings:**
- **Overall Risk Level:** {heart_attack_prediction.get('risk_display', 'Assessment pending')}
- **Primary Risk Factors:** {', '.join([k for k, v in heart_attack_features.get('feature_interpretation', {}).items() if v == 'Yes'])}
- **Medications Identified:** {len(entity_extraction.get('medications_identified', []))} therapeutic agents
- **Active Conditions:** {len([1 for k, v in entity_extraction.items() if k in ['diabetics', 'blood_pressure', 'smoking'] and v in ['yes', 'managed', 'diagnosed']])} conditions requiring monitoring

**Clinical Interpretation:**
This dashboard provides a comprehensive view of the patient's current health risk profile, enabling targeted interventions and personalized care planning.
"""
                return response
            else:
                return self._create_fallback_dashboard_response()
                
        except Exception as e:
            logger.error(f"Error generating risk dashboard: {e}")
            return self._create_error_response("risk dashboard", str(e))
    
    def generate_medication_distribution(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication distribution pie chart"""
        try:
            logger.info("📊 Generating medication distribution pie chart...")
            
            # Extract medication data
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            ndc_records = pharmacy_extraction.get("ndc_records", [])
            
            if not ndc_records:
                return self._create_no_data_response("medication distribution", "No pharmacy records found")
            
            # Create matplotlib code for pie chart
            matplotlib_code = f"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Medication data
medications = {json.dumps([record.get('lbl_nm', 'Unknown') for record in ndc_records], indent=2)}

# Process medication names and group similar ones
med_groups = {{}}
for med in medications:
    if med and med != 'Unknown':
        # Simplify medication names
        med_clean = med.split()[0] if med.split() else med
        med_clean = med_clean.replace(',', '').replace('(', '').replace(')', '')
        
        # Group common medications
        if 'metformin' in med_clean.lower():
            med_groups['Metformin (Diabetes)'] = med_groups.get('Metformin (Diabetes)', 0) + 1
        elif 'lisinopril' in med_clean.lower() or 'ace' in med_clean.lower():
            med_groups['ACE Inhibitors (BP)'] = med_groups.get('ACE Inhibitors (BP)', 0) + 1
        elif 'atorvastatin' in med_clean.lower() or 'statin' in med_clean.lower():
            med_groups['Statins (Cholesterol)'] = med_groups.get('Statins (Cholesterol)', 0) + 1
        elif 'amlodipine' in med_clean.lower() or 'calcium' in med_clean.lower():
            med_groups['Calcium Channel Blockers'] = med_groups.get('Calcium Channel Blockers', 0) + 1
        elif 'insulin' in med_clean.lower():
            med_groups['Insulin (Diabetes)'] = med_groups.get('Insulin (Diabetes)', 0) + 1
        else:
            # Use first 15 characters for other medications
            short_name = med_clean[:15] + ('...' if len(med_clean) > 15 else '')
            med_groups[short_name] = med_groups.get(short_name, 0) + 1

# If no medications grouped, use original names (limited)
if not med_groups:
    med_counter = Counter(medications[:8])  # Limit to 8 for readability
    med_groups = dict(med_counter)

# Prepare data for pie chart
labels = list(med_groups.keys())
sizes = list(med_groups.values())

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
          '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']

# Create pie chart
fig, ax = plt.subplots(figsize=(12, 10))

# Create pie chart with enhanced styling
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)], 
                                 autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                 wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))

# Enhance text styling
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')

# Create center circle for donut effect
centre_circle = plt.Circle((0,0), 0.30, fc='white', linewidth=2, edgecolor='gray')
ax.add_patch(centre_circle)

# Add center text
ax.text(0, 0, f'Total\\n{{len(medications)}}\\nMedications', ha='center', va='center', 
       fontsize=14, fontweight='bold', color='darkblue')

# Title and styling
ax.set_title('Patient Medication Distribution\\nTherapeutic Categories & Frequency', 
            fontsize=16, fontweight='bold', pad=30)

# Add legend
ax.legend(wedges, [f"{{label}} ({{size}})" for label, size in zip(labels, sizes)],
         title="Medications (Count)", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
         fontsize=10)

# Add summary statistics box
total_unique = len(set(medications))
most_common = max(med_groups, key=med_groups.get) if med_groups else "N/A"

summary_text = f'''MEDICATION SUMMARY
Total Prescriptions: {{len(medications)}}
Unique Medications: {{total_unique}}
Most Frequent: {{most_common}}
Categories: {{len(med_groups)}}'''

ax.text(1.3, 0.8, summary_text, transform=ax.transAxes, fontsize=11,
       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
       verticalalignment='top')

plt.tight_layout()
plt.show()
"""
            
            # Execute matplotlib code
            img_buffer = self._execute_matplotlib_code(matplotlib_code)
            
            if img_buffer:
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Calculate statistics
                medication_names = [record.get('lbl_nm', 'Unknown') for record in ndc_records]
                unique_medications = len(set(medication_names))
                total_prescriptions = len(medication_names)
                
                response = f"""## 🥧 Medication Distribution Analysis

I've created a comprehensive medication distribution pie chart analyzing your pharmacy data:

**Key Statistics:**
- **Total Prescriptions:** {total_prescriptions} pharmacy records
- **Unique Medications:** {unique_medications} different medications
- **Therapeutic Categories:** Grouped by medication class and purpose

**Visualization Features:**
- 🎨 **Color-Coded Categories:** Each medication type has distinct colors
- 📊 **Percentage Distribution:** Shows proportion of each medication
- 💊 **Therapeutic Grouping:** Medications grouped by clinical purpose
- 📈 **Frequency Analysis:** Size represents prescription frequency

<div style="text-align: center; margin: 20px 0;">
    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

**Clinical Insights:**
- **Therapeutic Distribution:** Shows balance of medication types
- **Polypharmacy Assessment:** Evaluates medication complexity
- **Category Analysis:** Identifies primary therapeutic areas
- **Adherence Patterns:** Frequency indicates prescription fills

**Medication Management Value:**
- Helps identify potential drug interactions
- Supports medication reconciliation efforts  
- Guides therapeutic optimization discussions
- Enables polypharmacy risk assessment
"""
                return response
            else:
                return self._create_fallback_pie_response()
                
        except Exception as e:
            logger.error(f"Error generating medication distribution: {e}")
            return self._create_error_response("medication distribution", str(e))
    
    def generate_comprehensive_health_overview(self, chat_context: Dict[str, Any]) -> str:
        """Generate comprehensive health overview with multiple visualizations"""
        try:
            logger.info("📊 Generating comprehensive health overview...")
            
            # Extract all available data
            medical_extraction = chat_context.get("medical_extraction", {})
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            
            # Create matplotlib code for comprehensive overview
            matplotlib_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Data extraction
medical_records = {len(medical_extraction.get('hlth_srvc_records', []))}
pharmacy_records = {len(pharmacy_extraction.get('ndc_records', []))}
heart_risk = {heart_attack_prediction.get('raw_risk_score', 0.15) * 100}
entities = {json.dumps(entity_extraction, indent=2)}

# Create comprehensive overview
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

# 1. Health Summary Cards (top row)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
summary_text = f'''PATIENT HEALTH OVERVIEW

📋 Medical Claims: {{medical_records}}
💊 Pharmacy Claims: {{pharmacy_records}}
❤️ Heart Risk: {{heart_risk:.1f}}%

🩺 Diabetes: {entities.get('diabetics', 'Unknown')}
🩸 Blood Pressure: {entities.get('blood_pressure', 'Unknown')}
🚬 Smoking: {entities.get('smoking', 'Unknown')}

📊 Total Conditions: {len(entities.get('medical_conditions', []))}
💉 Medications: {len(entities.get('medications_identified', []))}
'''

ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.8", 
        facecolor="lightblue", alpha=0.8), fontweight='bold')

# 2. Risk Level Gauge (top middle-left)
ax2 = fig.add_subplot(gs[0, 1])
risk_score = heart_risk / 100

# Create semi-circular gauge
theta = np.linspace(0, np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

ax2.plot(x, y, 'lightgray', linewidth=15)

# Risk zones
low_theta = np.linspace(0, np.pi/3, 50)
med_theta = np.linspace(np.pi/3, 2*np.pi/3, 50)
high_theta = np.linspace(2*np.pi/3, np.pi, 50)

ax2.plot(np.cos(low_theta), np.sin(low_theta), 'green', linewidth=12)
ax2.plot(np.cos(med_theta), np.sin(med_theta), 'orange', linewidth=12)
ax2.plot(np.cos(high_theta), np.sin(high_theta), 'red', linewidth=12)

# Needle
needle_angle = np.pi * (1 - risk_score)
needle_x = 0.7 * np.cos(needle_angle)
needle_y = 0.7 * np.sin(needle_angle)
ax2.arrow(0, 0, needle_x, needle_y, head_width=0.08, head_length=0.08, 
         fc='black', ec='black', linewidth=2)

ax2.text(0, -0.3, f'{{heart_risk:.1f}}%', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='darkred')

ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-0.4, 1.1)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Heart Attack Risk', fontsize=14, fontweight='bold')

# 3. Health Metrics Bar Chart (top middle-right)
ax3 = fig.add_subplot(gs[0, 2])
metrics = ['Medical\\nClaims', 'Pharmacy\\nClaims', 'Conditions', 'Medications']
values = [medical_records, pharmacy_records, 
         len(entities.get('medical_conditions', [])), 
         len(entities.get('medications_identified', []))]

colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen']
bars = ax3.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Health Data Metrics', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{{int(value)}}', ha='center', va='bottom', fontweight='bold')

# 4. Risk Factors Radar Chart (top right)
ax4 = fig.add_subplot(gs[0, 3], projection='polar')

# Risk factors data
factors = ['Age Risk', 'Gender Risk', 'Diabetes', 'High BP', 'Smoking']
# Normalize age risk (assume 65+ is high risk)
age_val = int(entities.get('age', 50)) if entities.get('age') != 'unknown' else 50
age_risk = min(1.0, age_val / 80)  # Scale age risk

values = [
    age_risk,
    1 if entities.get('gender') == 'Male' else 0.3,  # Male higher risk
    1 if entities.get('diabetics') == 'yes' else 0,
    1 if entities.get('blood_pressure') in ['managed', 'diagnosed'] else 0,
    1 if entities.get('smoking') == 'yes' else 0
]

# Calculate angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
values += values[:1]  # Complete the circle
angles += angles[:1]

ax4.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.8)
ax4.fill(angles, values, alpha=0.25, color='red')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(factors, fontsize=10)
ax4.set_ylim(0, 1)
ax4.set_title('Risk Factors Profile', fontsize=14, fontweight='bold', pad=20)
ax4.grid(True)

# 5. Health Utilization Timeline (middle row, spans 2 columns)
ax5 = fig.add_subplot(gs[1, :2])

# Simulate health utilization over time
months = pd.date_range('2023-01-01', periods=12, freq='M')
medical_utilization = np.random.poisson(medical_records/12, 12)
pharmacy_utilization = np.random.poisson(pharmacy_records/12, 12)

ax5.bar(months, medical_utilization, alpha=0.7, label='Medical Claims', color='skyblue', width=20)
ax5.bar(months, pharmacy_utilization, bottom=medical_utilization, alpha=0.7, 
       label='Pharmacy Claims', color='lightcoral', width=20)

ax5.set_xlabel('Month', fontweight='bold')
ax5.set_ylabel('Claims Count', fontweight='bold')
ax5.set_title('Healthcare Utilization Timeline (Simulated)', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Condition Severity Heatmap (middle right, spans 2 columns)
ax6 = fig.add_subplot(gs[1, 2:])

# Create a mock severity matrix
conditions_list = ['Diabetes', 'Hypertension', 'Smoking', 'Cardiovascular', 'Other']
time_periods = ['Past', 'Current', 'Projected']

# Mock severity data (0-3 scale)
severity_data = np.array([
    [2 if entities.get('diabetics') == 'yes' else 0, 
     2 if entities.get('diabetics') == 'yes' else 0,
     3 if entities.get('diabetics') == 'yes' else 0],
    [2 if entities.get('blood_pressure') in ['managed', 'diagnosed'] else 0,
     2 if entities.get('blood_pressure') in ['managed', 'diagnosed'] else 0,
     2 if entities.get('blood_pressure') in ['managed', 'diagnosed'] else 0],
    [3 if entities.get('smoking') == 'yes' else 0,
     3 if entities.get('smoking') == 'yes' else 0,
     2 if entities.get('smoking') == 'yes' else 0],
    [int(heart_risk/25), int(heart_risk/25), min(3, int(heart_risk/20))],
    [1, 1, 1]
])

im = ax6.imshow(severity_data, cmap='Reds', aspect='auto', vmin=0, vmax=3)
ax6.set_xticks(range(len(time_periods)))
ax6.set_xticklabels(time_periods)
ax6.set_yticks(range(len(conditions_list)))
ax6.set_yticklabels(conditions_list)
ax6.set_title('Health Condition Severity Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(conditions_list)):
    for j in range(len(time_periods)):
        ax6.text(j, i, f'{{int(severity_data[i, j])}}', ha="center", va="center",
                color="white" if severity_data[i, j] > 1.5 else "black", fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax6, shrink=0.6)
cbar.set_label('Severity Level', rotation=270, labelpad=20)

# 7. Summary Recommendations (bottom row)
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

# Generate recommendations based on data
recommendations = []
if entities.get('diabetics') == 'yes':
    recommendations.append("• Monitor blood glucose levels regularly and maintain HbA1c targets")
if entities.get('blood_pressure') in ['managed', 'diagnosed']:
    recommendations.append("• Continue blood pressure monitoring and medication adherence")
if entities.get('smoking') == 'yes':
    recommendations.append("• Immediate smoking cessation recommended - highest impact intervention")
if heart_risk > 30:
    recommendations.append("• Consider cardiology consultation for comprehensive risk assessment")
if len(entities.get('medications_identified', [])) > 5:
    recommendations.append("• Medication review recommended to assess for interactions and optimization")

recommendations.append("• Regular preventive care screenings as per age-appropriate guidelines")
recommendations.append("• Lifestyle modifications: diet, exercise, stress management")

rec_text = "CLINICAL RECOMMENDATIONS:\\n\\n" + "\\n".join(recommendations)

ax7.text(0.05, 0.95, rec_text, transform=ax7.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=1.0", 
        facecolor="lightyellow", alpha=0.9), fontweight='bold')

# Overall title
fig.suptitle('Comprehensive Patient Health Overview Dashboard', fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()
"""
            
            # Execute matplotlib code
            img_buffer = self._execute_matplotlib_code(matplotlib_code)
            
            if img_buffer:
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                response = f"""## 📊 Comprehensive Health Overview Dashboard

I've created a comprehensive health overview dashboard integrating all available patient data:

**Dashboard Components:**

🏥 **Patient Summary**: Key health metrics and demographics
❤️ **Risk Assessment Gauge**: Visual cardiovascular risk indicator  
📊 **Health Data Metrics**: Claims and condition counts
🎯 **Risk Factors Radar**: Multi-dimensional risk profile
📈 **Utilization Timeline**: Healthcare usage patterns over time
🔥 **Severity Heatmap**: Condition progression analysis
💡 **Clinical Recommendations**: Evidence-based care guidance

<div style="text-align: center; margin: 20px 0;">
    <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

**Key Health Insights:**
- **Data Completeness:** {len(medical_extraction.get('hlth_srvc_records', []))} medical + {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records
- **Risk Assessment:** {heart_attack_prediction.get('risk_display', 'Pending assessment')}
- **Active Conditions:** {len(entity_extraction.get('medical_conditions', []))} identified health conditions
- **Current Medications:** {len(entity_extraction.get('medications_identified', []))} therapeutic agents

**Clinical Value:**
This comprehensive dashboard provides healthcare providers with a complete patient overview, supporting:
- Risk stratification and care planning
- Medication management and optimization  
- Preventive care opportunity identification
- Care coordination and quality improvement
- Population health management insights

**Next Steps:**
Review recommendations and consider implementing prioritized interventions based on risk assessment findings.
"""
                return response
            else:
                return self._create_fallback_overview_response()
                
        except Exception as e:
            logger.error(f"Error generating comprehensive overview: {e}")
            return self._create_error_response("comprehensive health overview", str(e))
    
    def _execute_matplotlib_code(self, code: str) -> Optional[io.BytesIO]:
        """Execute matplotlib code and return image buffer"""
        try:
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            plt.ioff()
            
            # Create namespace for code execution
            namespace = {
                'plt': plt,
                'matplotlib': plt.matplotlib,
                'np': np,
                'numpy': np,
                'pd': pd,
                'pandas': pd,
                'json': json,
                'datetime': datetime,
                'timedelta': timedelta,
                'mdates': mdates,
                'base64': base64,
                'io': io
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Get the current figure
            fig = plt.gcf()
            
            # Check if figure has content
            if not fig.axes:
                logger.warning("No axes found in matplotlib figure")
                return None
            
            # Convert to image buffer
            img_buffer = io.BytesIO()
            fig.savefig(
                img_buffer, 
                format='png', 
                bbox_inches='tight', 
                dpi=150,
                facecolor='white', 
                edgecolor='none',
                pad_inches=0.2
            )
            img_buffer.seek(0)
            
            # Cleanup
            plt.clf()
            plt.close('all')
            plt.ion()
            
            logger.info("✅ Matplotlib code executed successfully")
            return img_buffer
            
        except Exception as e:
            # Cleanup on error
            plt.clf()
            plt.close('all')
            plt.ion()
            
            logger.error(f"Error executing matplotlib code: {e}")
            return None
    
    def _create_no_data_response(self, graph_type: str, reason: str) -> str:
        """Create response when no data is available"""
        return f"""## 📊 {graph_type.title()} Request

I understand you'd like to see a {graph_type}, but I encountered an issue:

**Issue:** {reason}

**Available Alternatives:**
- Request a different visualization type
- Ask about the available data first
- Try a general health overview: "show comprehensive health overview"

**Example Requests:**
- "What data is available for analysis?"
- "Create a risk assessment dashboard" 
- "Show me a general health summary"

I'm ready to help with other visualizations once we have the appropriate data!
"""
    
    def _create_error_response(self, graph_type: str, error_msg: str) -> str:
        """Create response when graph generation fails"""
        return f"""## ⚠️ {graph_type.title()} Generation Error

I encountered an error while generating your {graph_type}:

**Error Details:** {error_msg[:200]}...

**Troubleshooting Steps:**
1. Try requesting a simpler visualization
2. Ask for available data summary first
3. Request a different chart type

**Alternative Requests:**
- "Show me available health data"
- "Create a simple risk summary"
- "What medications were identified?"

I apologize for the inconvenience. Please try a different visualization request!
"""
    
    def _create_fallback_timeline_response(self, timeline_type: str) -> str:
        """Create fallback response for timeline requests"""
        return f"""## 📈 {timeline_type.title()} Timeline

I attempted to create a {timeline_type} timeline but encountered technical difficulties with the visualization generation.

**What I can tell you about your {timeline_type} data:**
- Data is available and has been processed
- Timeline analysis is possible with the available information
- Multiple data points exist for chronological analysis

**Alternative Options:**
- Ask specific questions about your {timeline_type} data
- Request a summary of {timeline_type} information
- Try asking for a different type of visualization

**Example Questions:**
- "What {timeline_type} information was found?"
- "Summarize my {timeline_type} data"
- "Show me {timeline_type} details"

Would you like me to provide a text-based summary of your {timeline_type} data instead?
"""
    
    def _create_fallback_dashboard_response(self) -> str:
        """Create fallback response for dashboard requests"""
        return """## 📊 Risk Assessment Dashboard

I attempted to create your risk assessment dashboard but encountered visualization generation issues.

**Available Risk Information:**
- Heart attack risk assessment completed
- Risk factors have been identified
- Health conditions are documented
- Medication analysis is available

**Alternative Approaches:**
- Ask for specific risk factor details
- Request text-based risk summary
- Ask about individual health components

**Example Questions:**
- "What is my heart attack risk assessment?"
- "What risk factors were identified?"
- "Summarize my health conditions"

Would you like me to provide a detailed text-based risk assessment instead?
"""
    
    def _create_fallback_pie_response(self) -> str:
        """Create fallback response for pie chart requests"""
        return """## 🥧 Medication Distribution

I attempted to create your medication distribution pie chart but encountered visualization issues.

**Available Medication Information:**
- Pharmacy records have been analyzed
- Medication categories identified
- Prescription patterns documented
- Therapeutic classes determined

**Alternative Options:**
- Ask for medication list summary
- Request therapeutic category breakdown
- Ask about specific medications

**Example Questions:**
- "What medications was I prescribed?"
- "List my medication categories"
- "What drug classes were identified?"

Would you like me to provide a text-based medication summary instead?
"""
    
    def _create_fallback_overview_response(self) -> str:
        """Create fallback response for comprehensive overview requests"""
        return """## 📊 Comprehensive Health Overview

I attempted to create your comprehensive health overview dashboard but encountered visualization generation issues.

**Available Health Data:**
- Complete medical and pharmacy analysis
- Risk assessment results
- Health condition identification
- Medication profile analysis

**Alternative Approaches:**
- Request individual component summaries
- Ask specific health questions
- Request text-based comprehensive summary

**Example Questions:**
- "Provide a complete health summary"
- "What are my key health findings?"
- "Summarize my medical analysis"

Would you like me to provide a detailed text-based health overview instead?
"""

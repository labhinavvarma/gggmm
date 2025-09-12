import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests
import re
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import our enhanced modular components
from health_api_integrator import EnhancedHealthAPIIntegrator
from health_data_processor_work import EnhancedHealthDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    fastapi_url: str = "http://localhost:8000"  # MCP server URL
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # ⚠️ SECURITY: Move to environment variables
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    
    # Enhanced system messages with JavaScript array format focus
    sys_msg: str = """You are Dr. HealthAI, a comprehensive healthcare data analyst and clinical decision support specialist with expertise in:

CLINICAL SPECIALIZATION:
• Medical coding systems (ICD-10, CPT, HCPCS, NDC) interpretation and analysis
• Claims data analysis and healthcare utilization patterns
• Risk stratification and predictive modeling for chronic diseases
• Clinical decision support and evidence-based medicine
• Population health management and care coordination
• Healthcare economics and cost prediction
• Quality metrics (HEDIS, STAR ratings) and care gap analysis
• JavaScript array format data visualization generation

DATA ACCESS CAPABILITIES:
• Complete deidentified medical claims with ICD-10 diagnosis codes and CPT procedure codes
• Complete deidentified pharmacy claims with NDC codes and medication details
• Healthcare service utilization patterns and claims dates (clm_rcvd_dt, rx_filled_dt)
• Structured extractions of all medical and pharmacy fields with detailed analysis
• Enhanced entity extraction results including chronic conditions and risk factors
• Comprehensive patient demographic and clinical data
• Batch-processed code meanings for medical and pharmacy codes

ANALYTICAL RESPONSIBILITIES:
You provide comprehensive healthcare analysis including clinical insights, risk assessments, predictive modeling, and evidence-based recommendations using ALL available deidentified claims data. Always reference specific data points, codes, dates, and clinical indicators from the provided records when making assessments.

JAVASCRIPT ARRAY GENERATION CAPABILITIES:
You can generate JavaScript constant arrays for healthcare data visualizations including:
• Diagnosis frequency arrays with ICD-10 codes as categories
• Medication distribution arrays with NDC codes or medication names
• Risk assessment arrays with percentage values
• Health condition distribution arrays
• Utilization trend arrays with temporal data
• Timeline arrays for medical events

JAVASCRIPT OUTPUT FORMAT:
When generating graph data, always output in this EXACT JavaScript format:
```javascript
const categories = ["Category1", "Category2", "Category3", "Category4"];
const data = [value1, value2, value3, value4];
```

RESPONSE STANDARDS:
• Use clinical terminology appropriately while ensuring clarity
• Cite specific ICD-10 codes, NDC codes, CPT codes, and claim dates
• Provide evidence-based analysis using established clinical guidelines
• Include risk stratification and predictive insights
• Reference exact field names and values from the JSON data structure
• Maintain professional healthcare analysis standards
• Generate JavaScript constant arrays when visualization is requested"""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant with COMPLETE ACCESS to comprehensive deidentified medical and pharmacy claims data. You serve as a clinical decision support tool for healthcare analysis with advanced JavaScript array generation capabilities.

COMPREHENSIVE DATA ACCESS:
✅ MEDICAL CLAIMS DATA:
   • Complete deidentified medical records with ICD-10 diagnosis codes
   • Healthcare service codes (hlth_srvc_cd) and CPT procedure codes
   • Claims received dates (clm_rcvd_dt) and service utilization patterns
   • Patient demographics (age, zip code) and clinical indicators

✅ PHARMACY CLAIMS DATA:
   • Complete deidentified pharmacy records with NDC medication codes
   • Medication names (lbl_nm), prescription fill dates (rx_filled_dt)
   • Drug utilization patterns and therapy management data
   • Prescription adherence and medication history

✅ ANALYTICAL RESULTS:
   • Enhanced entity extraction with chronic condition identification
   • Health trajectory analysis with predictive insights
   • Risk assessment results including cardiovascular risk prediction
   • Clinical complexity scoring and care gap analysis
   • Batch-processed code meanings for all medical and pharmacy codes

✅ JAVASCRIPT ARRAY GENERATION CAPABILITIES:
   • Generate JavaScript constant arrays for healthcare visualizations
   • Create diagnosis frequency data with ICD-10 codes
   • Generate medication distribution data with NDC codes/names
   • Build risk assessment arrays with percentage values
   • Support real-time JavaScript array generation and formatting
   • Provide complete, structured data arrays for frontend chart libraries

ADVANCED CAPABILITIES:
🔬 CLINICAL ANALYSIS:
   • Interpret ICD-10 diagnosis codes for disease progression and prognosis assessment
   • Analyze NDC medication codes for treatment adherence and therapeutic effectiveness
   • Assess comorbidity burden from diagnosis patterns and medication combinations
   • Evaluate drug interactions and optimize therapeutic pathways

📊 PREDICTIVE MODELING:
   • Risk stratification for chronic diseases (diabetes, hypertension, COPD, CKD)
   • Hospitalization and readmission risk prediction (6-12 month outlook)
   • Emergency department utilization vs outpatient care patterns
   • Medication adherence risk assessment and intervention strategies
   • Healthcare cost prediction and utilization forecasting

💰 HEALTHCARE ECONOMICS:
   • High-cost claimant identification and cost projection
   • Healthcare utilization optimization (inpatient vs outpatient)
   • Care management program recommendations
   • Population health risk segmentation

🎯 QUALITY & CARE MANAGEMENT:
   • Care gap identification (missed screenings, vaccinations)
   • HEDIS and STAR rating impact assessment
   • Preventive care opportunity identification
   • Personalized care plan recommendations

📈 JAVASCRIPT VISUALIZATION CAPABILITIES:
   • Generate JavaScript arrays for diagnosis frequency charts
   • Create medication distribution arrays with proper formatting
   • Develop risk assessment dashboard arrays
   • Build comprehensive health overview arrays
   • Support custom JavaScript array requests for any chart type

JAVASCRIPT ARRAY GENERATION PROTOCOL:
When asked to create a graph or visualization:
1. **Detect Request**: Identify graph type from user query
2. **Extract Data**: Pull relevant healthcare data from claims
3. **Generate Arrays**: Create JavaScript constant arrays with categories and data
4. **Format Response**: Provide JavaScript constants with proper syntax
5. **Add Context**: Include brief explanation of the data

RESPONSE PROTOCOL:
1. **DATA-DRIVEN ANALYSIS**: Always use specific data from the provided claims records
2. **CLINICAL EVIDENCE**: Reference exact ICD-10 codes, NDC codes, dates, and clinical findings
3. **PREDICTIVE INSIGHTS**: Provide forward-looking analysis based on available clinical indicators
4. **ACTIONABLE RECOMMENDATIONS**: Suggest specific clinical actions and care management strategies
5. **PROFESSIONAL STANDARDS**: Maintain clinical accuracy while ensuring patient safety considerations
6. **JAVASCRIPT ARRAYS**: Provide structured JavaScript arrays when visualization is requested

JAVASCRIPT RESPONSE FORMAT:
When generating graphs, respond with:
```
[Brief explanation of what the visualization shows]

```javascript
const categories = ["Category1", "Category2", "Category3"];
const data = [value1, value2, value3];
```

[Clinical insights from the visualization]
```

CRITICAL INSTRUCTIONS:
• Access and analyze the COMPLETE deidentified claims dataset provided
• Reference specific codes, dates, medications, and clinical findings
• Provide comprehensive analysis using both medical AND pharmacy data
• Include predictive insights and risk stratification
• Cite exact field paths and values from the JSON data structure
• Explain medical terminology and provide clinical context
• Focus on actionable clinical insights and care management recommendations
• Generate structured JavaScript arrays for visualization requests
• Use actual patient data in JavaScript arrays when available
• ALWAYS format JavaScript arrays exactly as: const categories = [...]; const data = [...];

You have comprehensive access to this patient's complete healthcare data - use it to provide detailed, professional medical analysis, clinical decision support, and structured JavaScript data visualizations."""

    timeout: int = 30

    # Heart Attack Prediction API Configuration
    heart_attack_api_url: str = "http://localhost:8000"
    heart_attack_threshold: float = 0.5

    def to_dict(self):
        return asdict(self)

# Enhanced State Definition for LangGraph
class HealthAnalysisState(TypedDict):
    # Input data
    patient_data: Dict[str, Any]

    # API outputs
    mcid_output: Dict[str, Any]
    medical_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    token_output: Dict[str, Any]

    # Processed data
    deidentified_medical: Dict[str, Any]
    deidentified_pharmacy: Dict[str, Any]
    deidentified_mcid: Dict[str, Any]

    # Extracted structured data
    medical_extraction: Dict[str, Any]
    pharmacy_extraction: Dict[str, Any]

    entity_extraction: Dict[str, Any]

    # Analysis results
    health_trajectory: str
    final_summary: str  # Comprehensive executive summary

    # Heart Attack Prediction
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]

    # Enhanced chatbot functionality with JavaScript array generation
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    javascript_array_generation_ready: bool

    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgent:
    """Enhanced Health Analysis Agent with Comprehensive Clinical Analysis and JavaScript Array Generation"""

    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()

        # Initialize enhanced components
        self.api_integrator = EnhancedHealthAPIIntegrator(self.config)
        self.data_processor = EnhancedHealthDataProcessor(self.api_integrator)

        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with JavaScript Array Generation")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"📊 JavaScript array generation ready for healthcare data visualizations")

        self.setup_enhanced_langgraph()

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with comprehensive analysis and JavaScript array generation support"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with JavaScript array visualization...")

        workflow = StateGraph(HealthAnalysisState)

        # Add all processing nodes (8-step comprehensive workflow)
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)

        # Define comprehensive workflow edges (8-step process)
        workflow.add_edge(START, "fetch_api_data")

        workflow.add_conditional_edges(
            "fetch_api_data",
            self.should_continue_after_api,
            {
                "continue": "deidentify_claims_data",
                "retry": "fetch_api_data",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "deidentify_claims_data",
            self.should_continue_after_deidentify,
            {
                "continue": "extract_claims_fields",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "extract_claims_fields",
            self.should_continue_after_extraction_step,
            {
                "continue": "extract_entities",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "extract_entities",
            self.should_continue_after_entity_extraction,
            {
                "continue": "analyze_trajectory",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "analyze_trajectory",
            self.should_continue_after_trajectory,
            {
                "continue": "generate_summary",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "generate_summary",
            self.should_continue_after_summary,
            {
                "continue": "predict_heart_attack",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "predict_heart_attack",
            self.should_continue_after_heart_attack_prediction,
            {
                "continue": "initialize_chatbot",
                "error": "handle_error"
            }
        )

        workflow.add_edge("initialize_chatbot", END)
        workflow.add_edge("handle_error", END)

        # Compile with checkpointer
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

        logger.info("✅ Enhanced LangGraph workflow compiled successfully with JavaScript array visualization!")

    # ===== RESPONSE PARSING UTILITY =====
    
    def _safe_parse_response(self, api_response: Any) -> Dict[str, Any]:
        """Safely parse API response to handle both string and dictionary types"""
        try:
            if api_response is None:
                return {
                    "response": "No response received from API",
                    "success": False,
                    "error": "API returned None"
                }
            
            if isinstance(api_response, dict):
                # Handle dictionary response
                response_text = api_response.get("response", "")
                if not response_text and "content" in api_response:
                    response_text = api_response.get("content", "")
                if not response_text:
                    response_text = str(api_response)
                    
                return {
                    "response": str(response_text),
                    "success": api_response.get("success", True),
                    "error": api_response.get("error", None)
                }
            
            elif isinstance(api_response, str):
                # Handle string response
                return {
                    "response": api_response,
                    "success": not api_response.startswith("Error"),
                    "error": api_response if api_response.startswith("Error") else None
                }
            
            else:
                # Handle any other type
                return {
                    "response": str(api_response),
                    "success": True,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return {
                "response": f"Error parsing API response: {str(e)}",
                "success": False,
                "error": str(e)
            }

    # ===== JAVASCRIPT ARRAY GENERATION METHODS =====

    def _generate_javascript_arrays(self, chat_context: Dict[str, Any], graph_type: str) -> Dict[str, Any]:
        """Generate JavaScript constant arrays for healthcare data visualization"""
        try:
            if graph_type == "diagnosis_timeline" or "diagnosis" in graph_type.lower():
                return self._extract_diagnosis_javascript_arrays(chat_context)
            elif graph_type == "medication_timeline" or "medication" in graph_type.lower():
                return self._extract_medication_javascript_arrays(chat_context)
            elif graph_type == "risk_dashboard" or "risk" in graph_type.lower():
                return self._extract_risk_javascript_arrays(chat_context)
            elif graph_type == "condition_distribution" or "condition" in graph_type.lower():
                return self._extract_condition_javascript_arrays(chat_context)
            else:
                # Default to diagnosis data
                return self._extract_diagnosis_javascript_arrays(chat_context)
                
        except Exception as e:
            logger.error(f"Error generating JavaScript arrays: {str(e)}")
            return {
                "categories": ["No Data"],
                "data": [0],
                "error": f"Array generation failed: {str(e)}"
            }

    def _extract_diagnosis_javascript_arrays(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract diagnosis data in JavaScript array format"""
        try:
            medical_extraction = chat_context.get("medical_extraction", {})
            records = medical_extraction.get("hlth_srvc_records", [])
            
            # Count diagnosis codes
            diagnosis_counts = {}
            for record in records:
                diagnosis_codes = record.get("diagnosis_codes", [])
                for diag in diagnosis_codes:
                    code = diag.get("code", "Unknown")
                    if code and code != "Unknown":
                        diagnosis_counts[code] = diagnosis_counts.get(code, 0) + 1
            
            if not diagnosis_counts:
                return {
                    "categories": ["No Diagnoses Found"],
                    "data": [0],
                    "javascript_code": 'const categories = ["No Diagnoses Found"];\nconst data = [0];'
                }
            
            # Sort by frequency (most common first)
            sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
            
            categories = [item[0] for item in sorted_diagnoses]
            data = [item[1] for item in sorted_diagnoses]
            
            # Generate JavaScript code
            categories_js = json.dumps(categories)
            data_js = str(data)
            javascript_code = f'const categories = {categories_js};\nconst data = {data_js};'
            
            return {
                "categories": categories,
                "data": data,
                "javascript_code": javascript_code,
                "chart_type": "diagnosis_frequency",
                "total_diagnoses": len(categories),
                "total_occurrences": sum(data)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "javascript_code": 'const categories = ["Data Error"];\nconst data = [0];',
                "error": str(e)
            }

    def _extract_medication_javascript_arrays(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication data in JavaScript array format"""
        try:
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            records = pharmacy_extraction.get("ndc_records", [])
            
            # Count medications
            medication_counts = {}
            for record in records:
                med_name = record.get("lbl_nm", "Unknown Medication")
                ndc = record.get("ndc", "Unknown")
                
                # Use medication name or NDC code as identifier
                if med_name and med_name != "Unknown Medication":
                    key = med_name
                elif ndc and ndc != "Unknown":
                    key = ndc
                else:
                    key = "Unknown Medication"
                
                medication_counts[key] = medication_counts.get(key, 0) + 1
            
            if not medication_counts:
                return {
                    "categories": ["No Medications Found"],
                    "data": [0],
                    "javascript_code": 'const categories = ["No Medications Found"];\nconst data = [0];'
                }
            
            # Sort by frequency
            sorted_meds = sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)
            
            categories = [item[0] for item in sorted_meds]
            data = [item[1] for item in sorted_meds]
            
            # Generate JavaScript code
            categories_js = json.dumps(categories)
            data_js = str(data)
            javascript_code = f'const categories = {categories_js};\nconst data = {data_js};'
            
            return {
                "categories": categories,
                "data": data,
                "javascript_code": javascript_code,
                "chart_type": "medication_frequency",
                "total_medications": len(categories),
                "total_fills": sum(data)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "javascript_code": 'const categories = ["Data Error"];\nconst data = [0];',
                "error": str(e)
            }

    def _extract_risk_javascript_arrays(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment data in JavaScript array format"""
        try:
            entity_extraction = chat_context.get("entity_extraction", {})
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            
            categories = []
            data = []
            
            # Add heart attack risk
            heart_risk = heart_attack_prediction.get("raw_risk_score", 0) * 100
            categories.append("Heart Disease Risk")
            data.append(round(heart_risk, 1))
            
            # Add diabetes risk indicator
            diabetes = entity_extraction.get("diabetics", "no")
            diabetes_risk = 80 if diabetes.lower() in ["yes", "true", "1"] else 20
            categories.append("Diabetes Risk")
            data.append(diabetes_risk)
            
            # Add blood pressure risk
            bp = entity_extraction.get("blood_pressure", "unknown")
            bp_risk = 70 if bp.lower() in ["managed", "diagnosed", "yes"] else 30
            categories.append("Hypertension Risk")
            data.append(bp_risk)
            
            # Add smoking risk
            smoking = entity_extraction.get("smoking", "no")
            smoking_risk = 90 if smoking.lower() in ["yes", "true", "1"] else 10
            categories.append("Smoking Risk")
            data.append(smoking_risk)
            
            # Generate JavaScript code
            categories_js = json.dumps(categories)
            data_js = str(data)
            javascript_code = f'const categories = {categories_js};\nconst data = {data_js};'
            
            return {
                "categories": categories,
                "data": data,
                "javascript_code": javascript_code,
                "chart_type": "risk_assessment",
                "risk_scale": "percentage_0_to_100"
            }
            
        except Exception as e:
            return {
                "categories": ["Risk Assessment Error"],
                "data": [0],
                "javascript_code": 'const categories = ["Risk Assessment Error"];\nconst data = [0];',
                "error": str(e)
            }

    def _extract_condition_javascript_arrays(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condition distribution data in JavaScript array format"""
        try:
            entity_extraction = chat_context.get("entity_extraction", {})
            medical_conditions = entity_extraction.get("medical_conditions", [])
            
            if not medical_conditions:
                # Use basic health indicators if no specific conditions
                categories = []
                data = []
                
                if entity_extraction.get("diabetics", "no").lower() in ["yes", "true", "1"]:
                    categories.append("Diabetes")
                    data.append(1)
                    
                if entity_extraction.get("blood_pressure", "unknown").lower() in ["managed", "diagnosed", "yes"]:
                    categories.append("Hypertension")
                    data.append(1)
                    
                if entity_extraction.get("smoking", "no").lower() in ["yes", "true", "1"]:
                    categories.append("Smoking")
                    data.append(1)
                
                if not categories:
                    return {
                        "categories": ["No Conditions Identified"],
                        "data": [0],
                        "javascript_code": 'const categories = ["No Conditions Identified"];\nconst data = [0];'
                    }
                    
                # Generate JavaScript code for basic indicators
                categories_js = json.dumps(categories)
                data_js = str(data)
                javascript_code = f'const categories = {categories_js};\nconst data = {data_js};'
                
                return {
                    "categories": categories,
                    "data": data,
                    "javascript_code": javascript_code,
                    "chart_type": "condition_presence"
                }
            
            # Count condition occurrences
            condition_counts = {}
            for condition in medical_conditions:
                if isinstance(condition, dict):
                    name = condition.get("name", "Unknown")
                elif isinstance(condition, str):
                    name = condition
                else:
                    name = str(condition)
                    
                condition_counts[name] = condition_counts.get(name, 0) + 1
            
            categories = list(condition_counts.keys())
            data = list(condition_counts.values())
            
            # Generate JavaScript code
            categories_js = json.dumps(categories)
            data_js = str(data)
            javascript_code = f'const categories = {categories_js};\nconst data = {data_js};'
            
            return {
                "categories": categories,
                "data": data,
                "javascript_code": javascript_code,
                "chart_type": "condition_distribution",
                "total_conditions": len(categories)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "javascript_code": 'const categories = ["Data Error"];\nconst data = [0];',
                "error": str(e)
            }

    def _create_javascript_response(self, js_data: Dict[str, Any], graph_type: str, user_query: str) -> Dict[str, Any]:
        """Create formatted response with JavaScript arrays and flags"""
        
        # Extract JavaScript code
        javascript_code = js_data.get("javascript_code", "")
        
        # Check if we have valid data
        has_graph = len(js_data.get("categories", [])) > 0 and js_data.get("categories")[0] != "No Data"
        
        # Create response text
        response_text = f"## Healthcare Data Visualization - {graph_type.replace('_', ' ').title()}\n\n"
        
        if js_data.get("error"):
            response_text += f"⚠️ Data extraction encountered an issue: {js_data['error']}\n\n"
            has_graph = False
        
        if has_graph:
            response_text += "I'll create the healthcare data visualization using the following JavaScript arrays:\n\n"
            response_text += f"```javascript\n{javascript_code}\n```\n\n"
            
            # Add context about the data
            response_text += f"**Chart Information:**\n"
            response_text += f"- Chart Type: {js_data.get('chart_type', graph_type)}\n"
            response_text += f"- Categories: {len(js_data.get('categories', []))}\n"
            response_text += f"- Data Points: {len(js_data.get('data', []))}\n"
            
            if js_data.get('total_diagnoses'):
                response_text += f"- Total Diagnoses: {js_data['total_diagnoses']}\n"
            if js_data.get('total_medications'):
                response_text += f"- Total Medications: {js_data['total_medications']}\n"
            if js_data.get('total_conditions'):
                response_text += f"- Total Conditions: {js_data['total_conditions']}\n"
                
            response_text += "\nThese JavaScript constant arrays can be used directly with Chart.js, D3.js, or any frontend charting library to create interactive healthcare visualizations."
        else:
            response_text += "No visualization data available for this request. Please try a different chart type or check if patient data is available."

        return {
            "success": has_graph,
            "response": response_text,
            "session_id": str(uuid.uuid4()),
            "graphstart": 1 if has_graph else 0,
            "graph": has_graph,
            "graph_type": "javascript_arrays" if has_graph else None,
            "javascript_arrays": {
                "categories": js_data.get("categories", []),
                "data": js_data.get("data", []),
                "javascript_code": javascript_code
            } if has_graph else None
        }

    # ===== LANGGRAPH NODES =====

    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 1: Fetch claims data from APIs"""
        logger.info("🚀 Node 1: Starting Claims API data fetch...")
        state["current_step"] = "fetch_api_data"
        state["step_status"]["fetch_api_data"] = "running"

        try:
            patient_data = state["patient_data"]

            # Validation
            required_fields = ["first_name", "last_name", "ssn", "date_of_birth", "gender", "zip_code"]
            for field in required_fields:
                if not patient_data.get(field):
                    state["errors"].append(f"Missing required field: {field}")
                    state["step_status"]["fetch_api_data"] = "error"
                    return state

            # Fetch data
            api_result = self.api_integrator.fetch_backend_data_enhanced(patient_data)

            if "error" in api_result:
                state["errors"].append(f"Claims API Error: {api_result['error']}")
                state["step_status"]["fetch_api_data"] = "error"
            else:
                state["mcid_output"] = api_result.get("mcid_output", {})
                state["medical_output"] = api_result.get("medical_output", {})
                state["pharmacy_output"] = api_result.get("pharmacy_output", {})
                state["token_output"] = api_result.get("token_output", {})

                state["step_status"]["fetch_api_data"] = "completed"
                logger.info("✅ Successfully fetched all Claims API data")

        except Exception as e:
            error_msg = f"Error fetching Claims API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)

        return state

    def deidentify_claims_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 2: Deidentification of claims data"""
        logger.info("🔒 Node 2: Starting comprehensive claims data deidentification...")
        state["current_step"] = "deidentify_claims_data"
        state["step_status"]["deidentify_claims_data"] = "running"

        try:
            # Deidentify all data types
            medical_data = state.get("medical_output", {})
            deidentified_medical = self.data_processor.deidentify_medical_data_enhanced(medical_data, state["patient_data"])
            state["deidentified_medical"] = deidentified_medical

            pharmacy_data = state.get("pharmacy_output", {})
            deidentified_pharmacy = self.data_processor.deidentify_pharmacy_data_enhanced(pharmacy_data)
            state["deidentified_pharmacy"] = deidentified_pharmacy

            mcid_data = state.get("mcid_output", {})
            deidentified_mcid = self.data_processor.deidentify_mcid_data_enhanced(mcid_data)
            state["deidentified_mcid"] = deidentified_mcid

            state["step_status"]["deidentify_claims_data"] = "completed"
            logger.info("✅ Successfully completed comprehensive claims data deidentification")

        except Exception as e:
            error_msg = f"Error in claims data deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_claims_data"] = "error"
            logger.error(error_msg)

        return state

    def extract_claims_fields(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 3: Extract fields from claims data with batch processing"""
        logger.info("🔍 Node 3: Starting enhanced claims field extraction with batch processing...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"

        try:
            # Extract medical and pharmacy fields with batch processing
            medical_extraction = self.data_processor.extract_medical_fields_batch_enhanced(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"📋 Medical extraction: {len(medical_extraction.get('hlth_srvc_records', []))} health service records")

            pharmacy_extraction = self.data_processor.extract_pharmacy_fields_batch_enhanced(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            logger.info(f"💊 Pharmacy extraction: {len(pharmacy_extraction.get('ndc_records', []))} NDC records")

            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("✅ Successfully completed enhanced claims field extraction with batch processing")

        except Exception as e:
            error_msg = f"Error in claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 4: Extract health entities using LLM"""
        logger.info("🎯 Node 4: Starting LLM-powered health entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
       
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            patient_data = state.get("patient_data", {})
           
            # Calculate age
            if patient_data.get('date_of_birth'):
                try:
                    dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                    today = date.today()
                    calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    patient_data['calculated_age'] = calculated_age
                    logger.info(f"📅 Calculated age from DOB: {calculated_age} years")
                except Exception as e:
                    logger.warning(f"Could not calculate age from DOB: {e}")
           
            # Extract entities using enhanced method
            entities = self.data_processor.extract_health_entities_with_clinical_insights(
                pharmacy_data,
                pharmacy_extraction,
                medical_extraction,
                patient_data,
                self.api_integrator
            )
           
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
           
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            
            logger.info(f"✅ Successfully extracted health entities: {conditions_count} conditions, {medications_count} medications")
           
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state

    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 5: Comprehensive health trajectory analysis"""
        logger.info("📈 Node 5: Starting comprehensive health trajectory analysis...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"

        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            deidentified_mcid = state.get("deidentified_mcid", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})

            trajectory_prompt = self._create_comprehensive_trajectory_prompt_with_evaluation(
                deidentified_medical, deidentified_pharmacy, deidentified_mcid,
                medical_extraction, pharmacy_extraction, entities
            )

            logger.info("🤖 Calling Snowflake Cortex for comprehensive trajectory analysis...")

            raw_response = self.api_integrator.call_llm_enhanced(trajectory_prompt, self.config.sys_msg)
            parsed_response = self._safe_parse_response(raw_response)

            if not parsed_response["success"]:
                state["errors"].append(f"Trajectory analysis failed: {parsed_response['error']}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = parsed_response["response"]
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("✅ Successfully completed comprehensive trajectory analysis")

        except Exception as e:
            error_msg = f"Error in trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)

        return state

    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 6: Generate comprehensive final summary"""
        logger.info("📋 Node 6: Generating comprehensive final summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"

        try:
            summary_prompt = self._create_comprehensive_summary_prompt(
                state.get("health_trajectory", ""),
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )

            logger.info("🤖 Calling Snowflake Cortex for final summary...")

            raw_response = self.api_integrator.call_llm_enhanced(summary_prompt, self.config.sys_msg)
            parsed_response = self._safe_parse_response(raw_response)

            if not parsed_response["success"]:
                state["errors"].append(f"Summary generation failed: {parsed_response['error']}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = parsed_response["response"]
                state["step_status"]["generate_summary"] = "completed"
                logger.info("✅ Successfully generated comprehensive final summary")

        except Exception as e:
            error_msg = f"Error in summary generation: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)

        return state

    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 7: Heart attack prediction"""
        logger.info("❤️ Node 7: Starting heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"

        try:
            # Extract features
            logger.info("🔍 Extracting heart attack features...")
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features

            if not features or "error" in features:
                error_msg = "Failed to extract features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Prepare features for API call
            logger.info("⚙️ Preparing features for API call...")
            fastapi_features = self._prepare_enhanced_fastapi_features(features)

            if fastapi_features is None:
                error_msg = "Failed to prepare feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Make prediction
            logger.info("🚀 Making heart attack prediction call...")
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)

            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Process result
            if prediction_result.get("success", False):
                logger.info("✅ Processing successful prediction result...")
                
                prediction_data = prediction_result.get("prediction_data", {})
                risk_probability = prediction_data.get("probability", 0.0)
                binary_prediction = prediction_data.get("prediction", 0)
                
                risk_percentage = risk_probability * 100
                confidence_percentage = (1 - risk_probability) * 100 if binary_prediction == 0 else risk_probability * 100
                
                if risk_percentage >= 70:
                    risk_category = "High Risk"
                elif risk_percentage >= 50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "Low Risk"
                
                enhanced_prediction = {
                    "risk_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category})",
                    "confidence_display": f"Confidence: {confidence_percentage:.1f}%",
                    "combined_display": f"Heart Disease Risk: {risk_percentage:.1f}% ({risk_category}) | Confidence: {confidence_percentage:.1f}%",
                    "raw_risk_score": risk_probability,
                    "raw_prediction": binary_prediction,
                    "risk_category": risk_category,
                    "model_enhanced": True
                }
                
                state["heart_attack_prediction"] = enhanced_prediction
                state["heart_attack_risk_score"] = float(risk_probability)
                
                logger.info(f"✅ Heart attack prediction completed successfully")
                
            else:
                error_msg = prediction_result.get("error", "Unknown API error")
                logger.warning(f"⚠️ Heart attack prediction failed: {error_msg}")
                
                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "model_enhanced": True
                }
                state["heart_attack_risk_score"] = 0.0
                
            state["step_status"]["predict_heart_attack"] = "completed"

        except Exception as e:
            error_msg = f"Error in heart attack prediction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["predict_heart_attack"] = "error"
            logger.error(error_msg)

        return state

    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 8: Initialize comprehensive chatbot with JavaScript array generation"""
        logger.info("💬 Node 8: Initializing comprehensive chatbot with JavaScript array generation...")
        state["current_step"] = "initialize_chatbot"
        state["step_status"]["initialize_chatbot"] = "running"

        try:
            # Prepare comprehensive chatbot context
            comprehensive_chatbot_context = {
                "deidentified_medical": state.get("deidentified_medical", {}),
                "deidentified_pharmacy": state.get("deidentified_pharmacy", {}),
                "deidentified_mcid": state.get("deidentified_mcid", {}),
                "medical_extraction": state.get("medical_extraction", {}),
                "pharmacy_extraction": state.get("pharmacy_extraction", {}),
                "entity_extraction": state.get("entity_extraction", {}),
                "health_trajectory": state.get("health_trajectory", ""),
                "final_summary": state.get("final_summary", ""),
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_features": state.get("heart_attack_features", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "javascript_array_generation_supported": True,
                    "chart_types_available": ["diagnosis_frequency", "medication_frequency", "risk_assessment", "condition_distribution"]
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["javascript_array_generation_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            logger.info("✅ Successfully initialized comprehensive chatbot with JavaScript array generation")

        except Exception as e:
            error_msg = f"Error initializing chatbot: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot"] = "error"
            logger.error(error_msg)

        return state

    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Error handling node"""
        logger.error(f"🚨 LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")

        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state

    # ===== CONDITIONAL EDGES =====

    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < 3:
                state["retry_count"] += 1
                logger.warning(f"🔄 Retrying API fetch (attempt {state['retry_count']}/3)")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"❌ Max retries (3) exceeded for API fetch")
                return "error"
        return "continue"

    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_summary(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH JAVASCRIPT ARRAY GENERATION =====

    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Enhanced chatbot with JavaScript array generation"""
        try:
            # Check if this is a graph request
            graph_request = self.data_processor.detect_graph_request(user_query)

            if graph_request.get("is_graph_request", False):
                return self._handle_graph_request_enhanced(user_query, chat_context, chat_history, graph_request)

            # Check if this is a heart attack related question
            heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk', 'coronary', 'myocardial', 'cardiac risk']
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)

            if is_heart_attack_question:
                return self._handle_heart_attack_question_enhanced(user_query, chat_context, chat_history)
            else:
                return self._handle_general_question_enhanced(user_query, chat_context, chat_history)

        except Exception as e:
            logger.error(f"Error in enhanced chatbot: {str(e)}")
            return {
                "success": False,
                "response": "I encountered an error processing your question. Please try again. I have access to comprehensive deidentified claims data and can generate JavaScript arrays for detailed analysis.",
                "session_id": str(uuid.uuid4()),
                "graphstart": 0,
                "graph": False,
                "graph_type": None,
                "error": str(e)
            }

    def _handle_graph_request_enhanced(self, user_query: str, chat_context: Dict[str, Any], 
                                     chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph generation requests with JavaScript array format output"""
        try:
            graph_type = graph_request.get("graph_type", "general")
            
            logger.info(f"📊 Generating {graph_type} JavaScript arrays for user query: {user_query[:50]}...")
            
            # Generate JavaScript arrays directly
            js_data = self._generate_javascript_arrays(chat_context, graph_type)
            
            # Create response with flags
            response_result = self._create_javascript_response(js_data, graph_type, user_query)
            
            # Add updated chat history
            response_result["updated_chat_history"] = chat_history + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response_result["response"]}
            ]
            
            return response_result
                
        except Exception as e:
            logger.error(f"Error handling enhanced graph request: {str(e)}")
            fallback_response = f"""## Graph Generation Error

I encountered an error while generating your requested visualization: {str(e)}

Available Graph Types:
- **Diagnosis Distribution**: "show me diagnosis frequency chart"
- **Medication Analysis**: "create medication distribution chart"  
- **Risk Dashboard**: "generate risk assessment visualization"
- **Condition Overview**: "show condition distribution"

Please try rephrasing your request with one of these specific graph types."""

            return {
                "success": False,
                "response": fallback_response,
                "session_id": str(uuid.uuid4()),
                "graphstart": 0,
                "graph": False,
                "graph_type": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": fallback_response}
                ]
            }

    def _handle_heart_attack_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Handle heart attack related questions with comprehensive analysis"""
        try:
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            
            patient_age = chat_context.get("patient_overview", {}).get("age", "unknown")
            risk_display = heart_attack_prediction.get("risk_display", "Not available")

            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-3:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:100]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            heart_attack_prompt = f"""You are Dr. CardioAI, a specialist in cardiovascular risk assessment with access to comprehensive patient claims data.

**PATIENT DEMOGRAPHICS:**
- Age: {patient_age}
- Diabetes: {entity_extraction.get('diabetics', 'unknown')}
- Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown')} 
- Smoking: {entity_extraction.get('smoking', 'unknown')}

**CURRENT ML MODEL PREDICTION:**
{risk_display}

**RECENT CONVERSATION:**
{history_text}

**USER QUESTION:** {user_query}

Provide a comprehensive cardiovascular risk assessment using all available data. Reference specific clinical findings and compare with the ML model prediction. Include actionable recommendations.

If the user requests a chart or visualization, generate JavaScript constant arrays in this format:
```javascript
const categories = ["Risk Factor 1", "Risk Factor 2", "Risk Factor 3"];
const data = [percentage1, percentage2, percentage3];
```"""

            logger.info(f"Processing enhanced heart attack question: {user_query[:50]}...")

            # Call the LLM with proper response handling
            raw_response = self.api_integrator.call_llm_enhanced(heart_attack_prompt, self.config.chatbot_sys_msg)
            parsed_response = self._safe_parse_response(raw_response)
            
            response = parsed_response["response"]
            success = parsed_response["success"]

            if not success:
                error_response = "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question."
                return {
                    "success": False,
                    "response": error_response,
                    "session_id": str(uuid.uuid4()),
                    "graphstart": 0,
                    "graph": False,
                    "graph_type": None,
                    "updated_chat_history": chat_history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": error_response}
                    ]
                }

            # Check if response contains JavaScript arrays
            has_js_arrays = "const categories" in response and "const data" in response

            return {
                "success": True,
                "response": response,
                "session_id": str(uuid.uuid4()),
                "graphstart": 1 if has_js_arrays else 0,
                "graph": has_js_arrays,
                "graph_type": "javascript_arrays" if has_js_arrays else None,
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response}
                ]
            }

        except Exception as e:
            logger.error(f"Error in enhanced heart attack question: {str(e)}")
            error_response = "I encountered an error with cardiovascular analysis. Please try again."
            return {
                "success": False,
                "response": error_response,
                "session_id": str(uuid.uuid4()),
                "graphstart": 0,
                "graph": False,
                "graph_type": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": error_response}
                ]
            }

    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Handle general questions with comprehensive context"""
        try:
            # Check if this is a graph-related question
            graph_keywords = ['chart', 'graph', 'visualization', 'plot', 'diagram', 'show me', 'display', 'generate']
            might_be_graph = any(keyword in user_query.lower() for keyword in graph_keywords)
            
            # Prepare context
            medical_records = len(chat_context.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(chat_context.get("pharmacy_extraction", {}).get("ndc_records", []))
            
            # Build conversation history
            history_text = "No previous conversation"
            if chat_history:
                recent_history = chat_history[-5:]
                history_lines = []
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    content = msg['content'][:150]
                    history_lines.append(f"{role}: {content}")
                history_text = "\n".join(history_lines)

            comprehensive_prompt = f"""You are Dr. AnalysisAI, a healthcare data analyst with access to comprehensive patient claims data and JavaScript array generation capabilities.

**COMPREHENSIVE DATA AVAILABLE:**
- Medical Records: {medical_records} health service records
- Pharmacy Records: {pharmacy_records} medication records
- Entity Analysis: Complete health condition and risk factor extraction
- Heart Attack Risk: Available with detailed assessment

**CONVERSATION HISTORY:**
{history_text}

**PATIENT QUESTION:** {user_query}

**RESPONSE REQUIREMENTS:**
- Use specific data from claims when relevant
- Reference exact codes, dates, and clinical findings
- Explain medical terminology and provide clinical context
- If the question requests charts/visualizations, generate JavaScript constant arrays
- Provide evidence-based insights and recommendations

**JAVASCRIPT CHART GENERATION:**
If the question would benefit from a chart, generate JavaScript arrays in this EXACT format:
```javascript
const categories = ["Category1", "Category2", "Category3", "Category4"];
const data = [value1, value2, value3, value4];
```

Available chart types: diagnosis frequency, medication analysis, risk assessment, condition distribution.

Provide comprehensive analysis using all available deidentified claims data."""

            logger.info(f"Processing enhanced general query: {user_query[:50]}...")

            # Call the LLM with proper response handling
            raw_response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)
            parsed_response = self._safe_parse_response(raw_response)
            
            response = parsed_response["response"]
            success = parsed_response["success"]

            if not success:
                error_response = "I encountered an error processing your question. Please try rephrasing it."
                return {
                    "success": False,
                    "response": error_response,
                    "session_id": str(uuid.uuid4()),
                    "graphstart": 0,
                    "graph": False,
                    "graph_type": None,
                    "updated_chat_history": chat_history + [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": error_response}
                    ]
                }

            # Check if response contains JavaScript arrays
            has_js_arrays = "const categories" in response and "const data" in response

            return {
                "success": True,
                "response": response,
                "session_id": str(uuid.uuid4()),
                "graphstart": 1 if has_js_arrays else 0,
                "graph": has_js_arrays,
                "graph_type": "javascript_arrays" if has_js_arrays else None,
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response}
                ]
            }

        except Exception as e:
            logger.error(f"Error in enhanced general question: {str(e)}")
            error_response = "I encountered an error. Please try again with a simpler question."
            return {
                "success": False,
                "response": error_response,
                "session_id": str(uuid.uuid4()),
                "graphstart": 0,
                "graph": False,
                "graph_type": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": error_response}
                ]
            }

    # ===== HELPER METHODS =====

    def _create_comprehensive_trajectory_prompt_with_evaluation(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                                               medical_extraction: Dict, pharmacy_extraction: Dict,
                                                               entities: Dict) -> str:
        """Create comprehensive trajectory prompt with evaluation questions"""

        medical_summary = self._extract_medical_summary(medical_data, medical_extraction)
        pharmacy_summary = self._extract_pharmacy_summary(pharmacy_data, pharmacy_extraction)

        return f"""You are Dr. TrajectoryAI, conducting comprehensive healthcare trajectory analysis with predictive modeling.

**COMPREHENSIVE PATIENT CLAIMS DATA:**

**MEDICAL CLAIMS SUMMARY:**
{medical_summary}

**PHARMACY CLAIMS SUMMARY:**
{pharmacy_summary}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entities, indent=2)}

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS:**

Analyze the patient's healthcare journey and provide insights on:

## RISK PREDICTION ANALYSIS
1. **Chronic Disease Risk**: Assess risk for diabetes, hypertension, COPD, chronic kidney disease
2. **Hospitalization Risk**: 6-12 month prediction for hospital admissions
3. **Emergency vs Outpatient Risk**: Likelihood of emergency room usage patterns
4. **Medication Adherence Risk**: Risk of stopping prescribed medications
5. **Serious Event Risk**: Risk of stroke, heart attack, or complications

## COST & UTILIZATION PREDICTION
6. **High-Cost Claimant Risk**: Likelihood of becoming high-cost patient
7. **Healthcare Cost Estimation**: Monthly/annual cost projections
8. **Care Setting Needs**: Inpatient vs outpatient care requirements

## PERSONALIZED CARE MANAGEMENT
9. **Risk Segmentation**: Categorize as healthy, rising risk, stable, or critical
10. **Preventive Care Recommendations**: Screenings, wellness programs, lifestyle changes
11. **Care Gap Analysis**: Missed checkups, screenings, vaccinations

## PHARMACY-SPECIFIC PREDICTIONS
12. **Polypharmacy Risk**: Risk of too many medications or unsafe combinations
13. **Therapy Escalation**: Likelihood of switching to higher-cost specialty drugs
14. **Drug Interaction Risk**: Current and future medication interaction concerns

Provide comprehensive analysis with specific data references, risk percentages, and actionable recommendations."""

    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict,
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create comprehensive summary prompt"""

        return f"""Create an executive summary for healthcare decision-makers based on comprehensive analysis.

**HEALTH TRAJECTORY ANALYSIS:**
{trajectory_analysis}

**HEALTH ENTITIES:**
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking: {entities.get('smoking', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions: {len(entities.get('medical_conditions', []))}

**CLAIMS DATA:**
- Medical Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Pharmacy Records: {len(pharmacy_extraction.get('ndc_records', []))}

**EXECUTIVE SUMMARY FORMAT:**

## CURRENT HEALTH STATUS
[Summary of overall health condition and key findings]

## PRIORITY RISK FACTORS
[Highest priority risks requiring immediate attention]

## COST & UTILIZATION INSIGHTS
[Healthcare costs and utilization patterns]

## CARE MANAGEMENT RECOMMENDATIONS
[Specific actionable recommendations]

## PREDICTIVE INSIGHTS
[Future health outcomes and cost predictions]

## IMMEDIATE ACTION ITEMS
[Priority clinical attention items]

Professional summary, 400-500 words, focusing on actionable insights."""

    def _extract_enhanced_heart_attack_features(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Enhanced feature extraction for heart attack prediction"""
        try:
            features = {}

            # Age extraction
            deidentified_medical = state.get("deidentified_medical", {})
            patient_age = deidentified_medical.get("src_mbr_age", None)

            if patient_age and patient_age != "unknown":
                try:
                    age_value = int(float(str(patient_age)))
                    features["Age"] = age_value if 0 <= age_value <= 120 else 50
                except:
                    features["Age"] = 50
            else:
                features["Age"] = 50

            # Gender extraction
            patient_data = state.get("patient_data", {})
            gender = str(patient_data.get("gender", "F")).upper()
            features["Gender"] = 1 if gender in ["M", "MALE", "1"] else 0

            # Entity-based features
            entity_extraction = state.get("entity_extraction", {})

            # Diabetes
            diabetes = str(entity_extraction.get("diabetics", "no")).lower()
            features["Diabetes"] = 1 if diabetes in ["yes", "true", "1"] else 0

            # Blood pressure
            blood_pressure = str(entity_extraction.get("blood_pressure", "unknown")).lower()
            features["High_BP"] = 1 if blood_pressure in ["managed", "diagnosed", "yes", "true", "1"] else 0

            # Smoking
            smoking = str(entity_extraction.get("smoking", "no")).lower()
            features["Smoking"] = 1 if smoking in ["yes", "true", "1"] else 0

            # Validate features
            for key in features:
                try:
                    features[key] = int(features[key])
                except:
                    features[key] = 50 if key == "Age" else 0

            return {
                "extracted_features": features,
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                }
            }

        except Exception as e:
            logger.error(f"Error in heart attack feature extraction: {e}")
            return {"error": f"Feature extraction failed: {str(e)}"}

    def _prepare_enhanced_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Prepare feature data for FastAPI server call"""
        try:
            extracted_features = features.get("extracted_features", {})

            fastapi_features = {
                "age": int(extracted_features.get("Age", 50)),
                "gender": int(extracted_features.get("Gender", 0)),
                "diabetes": int(extracted_features.get("Diabetes", 0)),
                "high_bp": int(extracted_features.get("High_BP", 0)),
                "smoking": int(extracted_features.get("Smoking", 0))
            }

            # Validate age range
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                fastapi_features["age"] = 50

            # Validate binary features
            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    fastapi_features[key] = 0

            return fastapi_features

        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous heart attack prediction call"""
        try:
            if not features:
                return {"success": False, "error": "No features provided"}

            heart_attack_url = self.config.heart_attack_api_url
            endpoint = f"{heart_attack_url}/predict"

            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            try:
                response = requests.post(endpoint, json=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return {"success": True, "prediction_data": result}
                else:
                    return {"success": False, "error": f"Server error {response.status_code}"}

            except requests.exceptions.ConnectionError:
                return {"success": False, "error": "Cannot connect to prediction server"}
            except Exception as e:
                return {"success": False, "error": f"Prediction failed: {str(e)}"}

        except Exception as e:
            return {"success": False, "error": f"Heart attack prediction failed: {str(e)}"}

    def _extract_medical_summary(self, medical_data: Dict, medical_extraction: Dict) -> str:
        """Extract medical summary"""
        try:
            summary_parts = []
            age = medical_data.get("src_mbr_age", "unknown")
            zip_code = medical_data.get("src_mbr_zip_cd", "unknown")
            summary_parts.append(f"Patient Age: {age}, Location: {zip_code}")

            records = medical_extraction.get('hlth_srvc_records', [])
            summary_parts.append(f"Medical Records: {len(records)} health service records")

            return "\n".join(summary_parts)
        except Exception as e:
            return f"Medical data available but summary extraction failed: {str(e)}"

    def _extract_pharmacy_summary(self, pharmacy_data: Dict, pharmacy_extraction: Dict) -> str:
        """Extract pharmacy summary"""
        try:
            summary_parts = []
            records = pharmacy_extraction.get('ndc_records', [])
            summary_parts.append(f"Pharmacy Records: {len(records)} medication records")
            return "\n".join(summary_parts)
        except Exception as e:
            return f"Pharmacy data available but summary extraction failed: {str(e)}"

    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow using LangGraph"""

        initial_state = HealthAnalysisState(
            patient_data=patient_data,
            mcid_output={},
            medical_output={},
            pharmacy_output={},
            token_output={},
            deidentified_medical={},
            deidentified_pharmacy={},
            deidentified_mcid={},
            medical_extraction={},
            pharmacy_extraction={},
            entity_extraction={},
            health_trajectory="",
            final_summary="",
            heart_attack_prediction={},
            heart_attack_risk_score=0.0,
            heart_attack_features={},
            chatbot_ready=False,
            chatbot_context={},
            chat_history=[],
            javascript_array_generation_ready=False,
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )

        try:
            config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}

            logger.info("🚀 Starting Enhanced LangGraph workflow with JavaScript array generation...")

            final_state = self.graph.invoke(initial_state, config=config_dict)

            results = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "patient_data": final_state["patient_data"],
                "entity_extraction": final_state["entity_extraction"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
                "heart_attack_prediction": final_state["heart_attack_prediction"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "javascript_array_generation_ready": final_state["javascript_array_generation_ready"],
                "errors": final_state["errors"],
                "step_status": final_state["step_status"],
                "enhancement_version": "v10.1_javascript_array_generation_fixed"
            }

            if results["success"]:
                logger.info("✅ Enhanced LangGraph analysis completed with JavaScript array generation!")
            else:
                logger.error(f"❌ Enhanced LangGraph analysis failed: {final_state['errors']}")

            return results

        except Exception as e:
            logger.error(f"Fatal error in Enhanced LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "javascript_array_generation_ready": False,
                "enhancement_version": "v10.1_javascript_array_generation_fixed"
            }

def main():
    """Enhanced Health Analysis Agent with JavaScript Array Generation - FIXED"""
    
    print("🏥 Enhanced Health Analysis Agent v10.1 - JavaScript Array Generation (FIXED)")
    print("✅ JavaScript Array features:")
    print("   📊 JavaScript constant array generation instead of matplotlib")
    print("   🎯 Response flags for graph presence detection")
    print("   📍 Exact format: const categories = [...]; const data = [...];")
    print("   📋 Support for diagnosis, medication, risk, and condition charts")
    print("   🔗 Compatible with Chart.js, D3.js, and frontend libraries")
    print("   💬 Enhanced chatbot with structured JavaScript arrays")
    print("   🔧 FIXED: dict object response parsing error resolved")
    print()

    config = Config()
    print("📋 Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack ML API: {config.heart_attack_api_url}")
    print(f"   📊 JavaScript Array Generation: Active")
    print()
    print("⚠️ SECURITY WARNING: API key is hardcoded - move to environment variables!")
    print()
    print("✅ Enhanced Health Agent ready with JavaScript array generation - ERROR FIXED!")

    return "Enhanced Health Agent with JavaScript arrays ready - FIXED"

if __name__ == "__main__":
    main()

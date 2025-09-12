
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests

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
    
    # Enhanced system messages with JSON graph generation focus
    sys_msg: str = """You are Dr. HealthAI, a comprehensive healthcare data analyst and clinical decision support specialist with expertise in:

CLINICAL SPECIALIZATION:
• Medical coding systems (ICD-10, CPT, HCPCS, NDC) interpretation and analysis
• Claims data analysis and healthcare utilization patterns
• Risk stratification and predictive modeling for chronic diseases
• Clinical decision support and evidence-based medicine
• Population health management and care coordination
• Healthcare economics and cost prediction
• Quality metrics (HEDIS, STAR ratings) and care gap analysis
• JSON-based healthcare data visualization and charting

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

JSON GRAPH GENERATION CAPABILITIES:
You can generate JSON data structures for healthcare data visualizations including:
• Diagnosis frequency charts with ICD-10 codes
• Medication distribution charts with NDC codes
• Risk assessment dashboards with percentage data
• Health condition distributions
• Utilization trend analysis data
• Timeline data for medical events

JSON OUTPUT FORMAT:
When generating graph data, always output in this structure:
{
  "categories": ["Category1", "Category2", "Category3"],
  "data": [value1, value2, value3],
  "chart_type": "specific_chart_type",
  "additional_metadata": {}
}

RESPONSE STANDARDS:
• Use clinical terminology appropriately while ensuring clarity
• Cite specific ICD-10 codes, NDC codes, CPT codes, and claim dates
• Provide evidence-based analysis using established clinical guidelines
• Include risk stratification and predictive insights
• Reference exact field names and values from the JSON data structure
• Maintain professional healthcare analysis standards
• Generate JSON chart data when visualization is requested"""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant with COMPLETE ACCESS to comprehensive deidentified medical and pharmacy claims data. You serve as a clinical decision support tool for healthcare analysis with advanced JSON-based graph generation capabilities.

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

✅ JSON GRAPH GENERATION CAPABILITIES:
   • Generate JSON data structures for healthcare visualizations
   • Create diagnosis frequency data, medication timeline data, risk dashboard data
   • Support real-time JSON chart data generation and formatting
   • Provide complete, structured data with metadata for frontend charting libraries

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

📈 JSON VISUALIZATION CAPABILITIES:
   • Generate JSON data for diagnosis frequency charts
   • Create risk assessment dashboard data with multiple metrics
   • Develop diagnosis progression timeline data
   • Build comprehensive health overview chart data
   • Support custom JSON structure requests for any chart type

JSON GRAPH GENERATION PROTOCOL:
When asked to create a graph or visualization:
1. **Detect Request**: Identify graph type from user query
2. **Extract Data**: Pull relevant healthcare data from claims
3. **Generate JSON**: Create structured JSON with categories and data arrays
4. **Add Metadata**: Include chart type and additional information
5. **Format Response**: Provide JSON with clear boundary indicators

RESPONSE PROTOCOL:
1. **DATA-DRIVEN ANALYSIS**: Always use specific data from the provided claims records
2. **CLINICAL EVIDENCE**: Reference exact ICD-10 codes, NDC codes, dates, and clinical findings
3. **PREDICTIVE INSIGHTS**: Provide forward-looking analysis based on available clinical indicators
4. **ACTIONABLE RECOMMENDATIONS**: Suggest specific clinical actions and care management strategies
5. **PROFESSIONAL STANDARDS**: Maintain clinical accuracy while ensuring patient safety considerations
6. **JSON GRAPH GENERATION**: Provide structured JSON data when visualization is requested

JSON RESPONSE FORMAT:
When generating graphs, respond with:
```
[Brief explanation of what the visualization shows]

[Complete JSON structure with categories and data]

<!-- GRAPH_METADATA: PRESENT=true, JSON_START=[position], JSON_END=[position], TYPE=[graph_type] -->

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
• Generate structured JSON data for visualization requests with proper metadata markers
• Use actual patient data in JSON structures when available

You have comprehensive access to this patient's complete healthcare data - use it to provide detailed, professional medical analysis, clinical decision support, and structured JSON data visualizations."""

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

    # Enhanced chatbot functionality with JSON graph generation
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    json_graph_generation_ready: bool

    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgent:
    """Enhanced Health Analysis Agent with Comprehensive Clinical Analysis and JSON Graph Generation"""

    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()

        # Initialize enhanced components
        self.api_integrator = EnhancedHealthAPIIntegrator(self.config)
        self.data_processor = EnhancedHealthDataProcessor(self.api_integrator)

        logger.info("🔧 Enhanced HealthAnalysisAgent initialized with JSON Graph Generation")
        logger.info(f"🌐 Snowflake API URL: {self.config.api_url}")
        logger.info(f"🤖 Model: {self.config.model}")
        logger.info(f"📡 MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"❤️ Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"📊 JSON graph generation ready for healthcare data visualizations")

        self.setup_enhanced_langgraph()

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with comprehensive analysis and JSON graph generation support"""
        logger.info("🔧 Setting up Enhanced LangGraph workflow with JSON-based visualization...")

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

        logger.info("✅ Enhanced LangGraph workflow compiled successfully with JSON-based visualization!")

    # ===== JSON GRAPH GENERATION METHODS =====

    def _convert_matplotlib_to_json(self, matplotlib_code: str, chat_context: Dict[str, Any], graph_type: str) -> Dict[str, Any]:
        """Convert healthcare data to JSON chart data structure"""
        try:
            # Extract relevant data from chat context based on graph type
            json_data = {"categories": [], "data": []}
            
            if graph_type == "diagnosis_timeline" or "diagnosis" in graph_type.lower():
                json_data = self._extract_diagnosis_json_data(chat_context)
            elif graph_type == "medication_timeline" or "medication" in graph_type.lower():
                json_data = self._extract_medication_json_data(chat_context)
            elif graph_type == "risk_dashboard" or "risk" in graph_type.lower():
                json_data = self._extract_risk_json_data(chat_context)
            elif graph_type == "condition_distribution" or "condition" in graph_type.lower():
                json_data = self._extract_condition_json_data(chat_context)
            else:
                # Default to diagnosis data
                json_data = self._extract_diagnosis_json_data(chat_context)
                
            return json_data
            
        except Exception as e:
            logger.error(f"Error converting healthcare data to JSON: {str(e)}")
            return {
                "categories": ["No Data"],
                "data": [0],
                "error": f"Data extraction failed: {str(e)}"
            }

    def _extract_diagnosis_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract diagnosis data in JSON format"""
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
                    "data": [0]
                }
            
            # Sort by frequency (most common first)
            sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
            
            categories = [item[0] for item in sorted_diagnoses]
            data = [item[1] for item in sorted_diagnoses]
            
            return {
                "categories": categories,
                "data": data,
                "chart_type": "diagnosis_frequency",
                "total_diagnoses": len(categories),
                "total_occurrences": sum(data)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "error": str(e)
            }

    def _extract_medication_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract medication data in JSON format"""
        try:
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            records = pharmacy_extraction.get("ndc_records", [])
            
            # Count medications
            medication_counts = {}
            for record in records:
                med_name = record.get("lbl_nm", "Unknown Medication")
                ndc = record.get("ndc", "Unknown")
                
                # Use medication name as primary identifier
                if med_name and med_name != "Unknown Medication":
                    key = f"{med_name}"
                    medication_counts[key] = medication_counts.get(key, 0) + 1
            
            if not medication_counts:
                return {
                    "categories": ["No Medications Found"],
                    "data": [0]
                }
            
            # Sort by frequency
            sorted_meds = sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)
            
            categories = [item[0] for item in sorted_meds]
            data = [item[1] for item in sorted_meds]
            
            return {
                "categories": categories,
                "data": data,
                "chart_type": "medication_frequency",
                "total_medications": len(categories),
                "total_fills": sum(data)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "error": str(e)
            }

    def _extract_risk_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment data in JSON format"""
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
            
            return {
                "categories": categories,
                "data": data,
                "chart_type": "risk_assessment",
                "risk_scale": "percentage_0_to_100"
            }
            
        except Exception as e:
            return {
                "categories": ["Risk Assessment Error"],
                "data": [0],
                "error": str(e)
            }

    def _extract_condition_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condition distribution data in JSON format"""
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
                        "data": [0]
                    }
                    
                return {
                    "categories": categories,
                    "data": data,
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
            
            return {
                "categories": categories,
                "data": data,
                "chart_type": "condition_distribution",
                "total_conditions": len(categories)
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "error": str(e)
            }

    def _create_json_graph_response(self, graph_json: str, graph_type: str, json_data: Dict[str, Any]) -> str:
        """Create formatted response with JSON graph data and indicators"""
        
        # Calculate JSON boundaries
        intro_text = f"## Healthcare Data Visualization - {graph_type.replace('_', ' ').title()}\n\n"
        
        if json_data.get("error"):
            intro_text += f"⚠️ Data extraction encountered an issue: {json_data['error']}\n\n"
        
        intro_text += "**Graph Data:**\n\n"
        
        json_start_pos = len(intro_text)
        json_end_pos = json_start_pos + len(graph_json)
        
        conclusion_text = f"\n\n**Chart Information:**\n"
        conclusion_text += f"- Chart Type: {json_data.get('chart_type', graph_type)}\n"
        conclusion_text += f"- Categories: {len(json_data.get('categories', []))}\n"
        conclusion_text += f"- Data Points: {len(json_data.get('data', []))}\n"
        
        if json_data.get('total_diagnoses'):
            conclusion_text += f"- Total Diagnoses: {json_data['total_diagnoses']}\n"
        if json_data.get('total_medications'):
            conclusion_text += f"- Total Medications: {json_data['total_medications']}\n"
        if json_data.get('total_conditions'):
            conclusion_text += f"- Total Conditions: {json_data['total_conditions']}\n"
            
        conclusion_text += "\nThis JSON data structure can be used with any charting library (Chart.js, D3.js, etc.) to create interactive healthcare visualizations."
        
        # Complete response
        complete_response = intro_text + graph_json + conclusion_text
        
        # Add metadata as special markers in response
        metadata = f"\n\n<!-- GRAPH_METADATA: PRESENT=true, JSON_START={json_start_pos}, JSON_END={json_end_pos}, TYPE={graph_type} -->"
        complete_response += metadata
        
        return complete_response

    def _create_error_response(self, error_message: str) -> str:
        """Create error response with no graph indicator"""
        response = f"""## Graph Generation Error

I encountered an error while generating your requested visualization: {error_message}

Available Graph Types:
- **Diagnosis Distribution**: "show me diagnosis frequency chart"
- **Medication Analysis**: "create medication distribution chart"  
- **Risk Dashboard**: "generate risk assessment visualization"
- **Condition Overview**: "show condition distribution"

Please try rephrasing your request with one of these specific graph types.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""
        
        return response

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

            response = self.api_integrator.call_llm_enhanced(trajectory_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
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

            response = self.api_integrator.call_llm_enhanced(summary_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
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
        """Node 8: Initialize comprehensive chatbot with JSON graph generation"""
        logger.info("💬 Node 8: Initializing comprehensive chatbot with JSON graph generation...")
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
                    "json_graph_generation_supported": True,
                    "chart_types_available": ["diagnosis_frequency", "medication_frequency", "risk_assessment", "condition_distribution"]
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["json_graph_generation_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            logger.info("✅ Successfully initialized comprehensive chatbot with JSON graph generation")

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

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH JSON GRAPH GENERATION =====

    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with JSON graph generation instead of matplotlib"""
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
            return """I encountered an error processing your question. Please try again. I have access to comprehensive deidentified claims data and can generate JSON visualizations for detailed analysis.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""

    def _handle_graph_request_enhanced(self, user_query: str, chat_context: Dict[str, Any], 
                                     chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> str:
        """Handle graph generation requests with JSON format output"""
        try:
            graph_type = graph_request.get("graph_type", "general")
            
            logger.info(f"📊 Generating {graph_type} JSON visualization for user query: {user_query[:50]}...")
            
            # Convert to JSON format
            json_data = self._convert_matplotlib_to_json("", chat_context, graph_type)
            
            # Create JSON string
            graph_json = json.dumps(json_data, indent=2)
            
            # Create response with indicators
            response = self._create_json_graph_response(graph_json, graph_type, json_data)
            
            return response
                
        except Exception as e:
            logger.error(f"Error handling enhanced graph request: {str(e)}")
            return self._create_error_response(str(e))

    def _handle_heart_attack_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
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

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=heart_attack_analysis -->"""

            logger.info(f"Processing enhanced heart attack question: {user_query[:50]}...")

            response = self.api_integrator.call_llm_enhanced(heart_attack_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return """I encountered an error analyzing cardiovascular risk. Please try rephrasing your question.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""

            # Add metadata to response
            return response + "\n\n<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=heart_attack_analysis -->"

        except Exception as e:
            logger.error(f"Error in enhanced heart attack question: {str(e)}")
            return """I encountered an error with cardiovascular analysis. Please try again.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""

    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with comprehensive context"""
        try:
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

            comprehensive_prompt = f"""You are Dr. AnalysisAI, a healthcare data analyst with access to comprehensive patient claims data and JSON visualization capabilities.

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
- Generate JSON chart data if visualization would be helpful
- Provide evidence-based insights and recommendations

**JSON CHART GENERATION:**
If the question would benefit from a chart, mention that you can "generate JSON chart data" and specify the type (diagnosis frequency, medication analysis, risk assessment, or condition distribution).

Provide comprehensive analysis using all available deidentified claims data."""

            logger.info(f"Processing enhanced general query: {user_query[:50]}...")

            response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return """I encountered an error processing your question. Please try rephrasing it.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""

            # Add metadata to response
            return response + "\n\n<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=general_analysis -->"

        except Exception as e:
            logger.error(f"Error in enhanced general question: {str(e)}")
            return """I encountered an error. Please try again with a simpler question.

<!-- GRAPH_METADATA: PRESENT=false, JSON_START=0, JSON_END=0, TYPE=error -->"""

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
            json_graph_generation_ready=False,
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )

        try:
            config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}

            logger.info("🚀 Starting Enhanced LangGraph workflow with JSON graph generation...")

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
                "json_graph_generation_ready": final_state["json_graph_generation_ready"],
                "errors": final_state["errors"],
                "step_status": final_state["step_status"],
                "enhancement_version": "v9.0_json_graph_generation"
            }

            if results["success"]:
                logger.info("✅ Enhanced LangGraph analysis completed with JSON graph generation!")
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
                "json_graph_generation_ready": False,
                "enhancement_version": "v9.0_json_graph_generation"
            }

# Helper functions to parse graph metadata from responses
def parse_graph_metadata(response: str) -> Dict[str, Any]:
    """Parse graph metadata from response"""
    try:
        metadata_pattern = r'<!-- GRAPH_METADATA: (.*?) -->'
        match = re.search(metadata_pattern, response)
        
        if not match:
            return {"present": False, "json_start": 0, "json_end": 0, "type": "none"}
        
        metadata_str = match.group(1)
        metadata = {}
        
        for pair in metadata_str.split(', '):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if value.lower() == 'true':
                    metadata[key] = True
                elif value.lower() == 'false':
                    metadata[key] = False
                elif value.isdigit():
                    metadata[key] = int(value)
                else:
                    metadata[key] = value
        
        return metadata
        
    except Exception as e:
        return {"present": False, "json_start": 0, "json_end": 0, "type": "error", "parse_error": str(e)}

def extract_graph_json_from_response(response: str) -> Dict[str, Any]:
    """Extract graph JSON data from chatbot response"""
    try:
        metadata = parse_graph_metadata(response)
        
        if not metadata.get("present", False):
            return {"has_graph": False, "data": None, "metadata": metadata}
        
        json_start = metadata.get("json_start", 0)
        json_end = metadata.get("json_end", 0)
        
        if json_start == 0 and json_end == 0:
            return {"has_graph": False, "data": None, "error": "Invalid JSON boundaries"}
        
        json_str = response[json_start:json_end]
        graph_data = json.loads(json_str)
        
        return {
            "has_graph": True,
            "data": graph_data,
            "metadata": metadata,
            "json_boundaries": {"start": json_start, "end": json_end}
        }
        
    except Exception as e:
        return {"has_graph": False, "data": None, "error": str(e)}

def main():
    """Enhanced Health Analysis Agent with JSON Graph Generation"""
    
    print("🏥 Enhanced Health Analysis Agent v9.0 - JSON Graph Generation")
    print("✅ New JSON-based features:")
    print("   📊 JSON chart data generation instead of matplotlib code")
    print("   🎯 Response indicators for graph presence detection")
    print("   📍 JSON boundary markers for precise data extraction")
    print("   📋 Support for diagnosis, medication, risk, and condition charts")
    print("   🔗 Compatible with Chart.js, D3.js, and other frontend libraries")
    print("   💬 Enhanced chatbot with structured data visualization")
    print()

    config = Config()
    print("📋 Configuration:")
    print(f"   🌐 Snowflake API: {config.api_url}")
    print(f"   🤖 Model: {config.model}")
    print(f"   📡 Server: {config.fastapi_url}")
    print(f"   ❤️ Heart Attack ML API: {config.heart_attack_api_url}")
    print(f"   📊 JSON Graph Generation: Active")
    print()
    print("⚠️ SECURITY WARNING: API key is hardcoded - move to environment variables!")
    print()
    print("✅ Enhanced Health Agent ready with JSON graph generation!")

    return "Enhanced Health Agent with JSON graphs ready"

if __name__ == "__main__":
    main()

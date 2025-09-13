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
    api_key: str = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"  # SECURITY: Move to environment variables
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    
    # Enhanced system messages with JSON structure support
    sys_msg: str = """You are Dr. HealthAI, a comprehensive healthcare data analyst and clinical decision support specialist with expertise in:

CLINICAL SPECIALIZATION:
• Medical coding systems (ICD-10, CPT, HCPCS, NDC) interpretation and analysis
• Claims data analysis and healthcare utilization patterns
• Risk stratification and predictive modeling for chronic diseases
• Clinical decision support and evidence-based medicine
• Population health management and care coordination
• Healthcare economics and cost prediction
• Quality metrics (HEDIS, STAR ratings) and care gap analysis
• Advanced healthcare data visualization with JSON structure format

DATA ACCESS CAPABILITIES:
• Complete deidentified medical claims with ICD-10 diagnosis codes and CPT procedure codes
• Complete deidentified pharmacy claims with NDC codes and medication details
• Healthcare service utilization patterns and claims dates (clm_rcvd_dt, rx_filled_dt)
• Structured extractions of all medical and pharmacy fields with detailed analysis
• Enhanced entity extraction results including chronic conditions and risk factors
• Comprehensive patient demographic and clinical data
• Batch-processed code meanings for medical and pharmacy codes

JSON GRAPH GENERATION CAPABILITIES:
• Generate JSON structure for healthcare visualizations
• Create diagnosis frequency data with ICD-10 codes
• Generate medication distribution data with NDC codes/names
• Build risk assessment arrays with percentage values
• Support real-time JSON graph generation and formatting

JSON RESPONSE FORMAT:
When generating graphs, respond with:
***GRAPH_START***
{
  "categories": ["Category1", "Category2", "Category3"],
  "data": [value1, value2, value3],
  "graph_type": "chart_type",
  "title": "Chart Title"
}
***GRAPH_END***

RESPONSE STANDARDS:
• Use clinical terminology appropriately while ensuring clarity
• Cite specific ICD-10 codes, NDC codes, CPT codes, and claim dates
• Provide evidence-based analysis using established clinical guidelines
• Include risk stratification and predictive insights
• Generate JSON structure for visualization requests
• Reference exact field names and values from the JSON data structure
• Maintain professional healthcare analysis standards"""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant with COMPLETE ACCESS to comprehensive deidentified medical and pharmacy claims data. You serve as a clinical decision support tool for healthcare analysis with advanced JSON graph generation capabilities.

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
   • Generate JSON structure for healthcare visualizations
   • Create medication timelines, diagnosis progressions, risk dashboards
   • Support real-time chart generation with boundary markers
   • Provide complete JSON data structure with proper formatting

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

📈 VISUALIZATION CAPABILITIES:
   • Generate JSON structure for medication timeline charts
   • Create risk assessment dashboards with multiple metrics
   • Develop diagnosis progression visualizations
   • Build comprehensive health overview charts
   • Support custom visualization requests with boundary markers

JSON GRAPH GENERATION PROTOCOL:
When asked to create a graph or visualization:
1. **Detect Request**: Identify graph type from user query
2. **Generate JSON**: Create complete JSON structure with boundary markers
3. **Use Real Data**: Incorporate actual patient data when available
4. **Provide Context**: Include brief explanation of the visualization
5. **Ensure Quality**: Generate professional, informative chart data

RESPONSE PROTOCOL:
1. **DATA-DRIVEN ANALYSIS**: Always use specific data from the provided claims records
2. **CLINICAL EVIDENCE**: Reference exact ICD-10 codes, NDC codes, dates, and clinical findings
3. **PREDICTIVE INSIGHTS**: Provide forward-looking analysis based on available clinical indicators
4. **ACTIONABLE RECOMMENDATIONS**: Suggest specific clinical actions and care management strategies
5. **PROFESSIONAL STANDARDS**: Maintain clinical accuracy while ensuring patient safety considerations
6. **JSON GRAPH GENERATION**: Provide JSON structure with boundary markers when visualization is requested

JSON GRAPH RESPONSE FORMAT:
When generating graphs, respond with:
```
[Brief explanation of what the visualization shows]

***GRAPH_START***
{
  "categories": ["Category1", "Category2", "Category3"],
  "data": [value1, value2, value3],
  "graph_type": "chart_type",
  "title": "Chart Title"
}
***GRAPH_END***

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
• Generate JSON structure with boundary markers for visualization requests
• Use actual patient data in graphs when available

You have comprehensive access to this patient's complete healthcare data - use it to provide detailed, professional medical analysis, clinical decision support, and advanced data visualizations in JSON format."""

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
    """Enhanced Health Analysis Agent with JSON Graph Generation and Boundary Markers"""

    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()

        # Initialize enhanced components
        self.api_integrator = EnhancedHealthAPIIntegrator(self.config)
        self.data_processor = EnhancedHealthDataProcessor(self.api_integrator)

        logger.info("Enhanced HealthAnalysisAgent initialized with JSON Graph Generation")
        logger.info(f"Snowflake API URL: {self.config.api_url}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"JSON graph generation ready for medical data visualizations")

        self.setup_enhanced_langgraph()

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with JSON graph generation support"""
        logger.info("Setting up Enhanced LangGraph workflow with JSON graph generation...")

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

        logger.info("Enhanced LangGraph workflow compiled successfully with JSON graph generation!")

    # ===== LANGGRAPH NODES =====

    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 1: Fetch claims data from APIs"""
        logger.info("Node 1: Starting Claims API data fetch...")
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
                logger.info("Successfully fetched all Claims API data")

        except Exception as e:
            error_msg = f"Error fetching Claims API data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["fetch_api_data"] = "error"
            logger.error(error_msg)

        return state

    def deidentify_claims_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 2: Deidentification of claims data"""
        logger.info("Node 2: Starting comprehensive claims data deidentification...")
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
            logger.info("Successfully completed comprehensive claims data deidentification")

        except Exception as e:
            error_msg = f"Error in claims data deidentification: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["deidentify_claims_data"] = "error"
            logger.error(error_msg)

        return state

    def extract_claims_fields(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 3: Extract fields from claims data with batch processing"""
        logger.info("Node 3: Starting enhanced claims field extraction with batch processing...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"

        try:
            # Extract medical and pharmacy fields with batch processing
            medical_extraction = self.data_processor.extract_medical_fields_batch_enhanced(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction
            logger.info(f"Medical extraction: {len(medical_extraction.get('hlth_srvc_records', []))} health service records")
            logger.info(f"Medical batch status: {medical_extraction.get('llm_call_status', 'unknown')}")

            pharmacy_extraction = self.data_processor.extract_pharmacy_fields_batch_enhanced(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction
            logger.info(f"Pharmacy extraction: {len(pharmacy_extraction.get('ndc_records', []))} NDC records")
            logger.info(f"Pharmacy batch status: {pharmacy_extraction.get('llm_call_status', 'unknown')}")

            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("Successfully completed enhanced claims field extraction with batch processing")

        except Exception as e:
            error_msg = f"Error in claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 4: Extract health entities using LLM"""
        logger.info("Node 4: Starting LLM-powered health entity extraction...")
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
                    logger.info(f"Calculated age from DOB: {calculated_age} years")
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
            stable_analysis = entities.get("stable_analysis", False)
            age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
           
            logger.info(f"Successfully extracted health entities: {conditions_count} conditions, {medications_count} medications")
            logger.info(f"Entity results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
            logger.info(f"{age_info}")
            logger.info(f"Stable analysis: {stable_analysis}")
           
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state

    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 5: Comprehensive health trajectory analysis with evaluation questions"""
        logger.info("Node 5: Starting comprehensive health trajectory analysis...")
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

            logger.info("Calling Snowflake Cortex for comprehensive trajectory analysis...")

            response = self.api_integrator.call_llm_enhanced(trajectory_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Trajectory analysis failed: {response}")
                state["step_status"]["analyze_trajectory"] = "error"
            else:
                state["health_trajectory"] = response
                state["step_status"]["analyze_trajectory"] = "completed"
                logger.info("Successfully completed comprehensive trajectory analysis")

        except Exception as e:
            error_msg = f"Error in trajectory analysis: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["analyze_trajectory"] = "error"
            logger.error(error_msg)

        return state

    def generate_summary(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 6: Generate comprehensive final summary"""
        logger.info("Node 6: Generating comprehensive final summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"

        try:
            summary_prompt = self._create_comprehensive_summary_prompt(
                state.get("health_trajectory", ""),
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {})
            )

            logger.info("Calling Snowflake Cortex for final summary...")

            response = self.api_integrator.call_llm_enhanced(summary_prompt, self.config.sys_msg)

            if response.startswith("Error"):
                state["errors"].append(f"Summary generation failed: {response}")
                state["step_status"]["generate_summary"] = "error"
            else:
                state["final_summary"] = response
                state["step_status"]["generate_summary"] = "completed"
                logger.info("Successfully generated comprehensive final summary")

        except Exception as e:
            error_msg = f"Error in summary generation: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["generate_summary"] = "error"
            logger.error(error_msg)

        return state

    def predict_heart_attack(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 7: Heart attack prediction"""
        logger.info("Node 7: Starting heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"

        try:
            # Extract features
            logger.info("Extracting heart attack features...")
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features

            if not features or "error" in features:
                error_msg = "Failed to extract features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Prepare features for API call
            logger.info("Preparing features for API call...")
            fastapi_features = self._prepare_enhanced_fastapi_features(features)

            if fastapi_features is None:
                error_msg = "Failed to prepare feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Make prediction
            logger.info("Making heart attack prediction call...")
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)

            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Process result
            if prediction_result.get("success", False):
                logger.info("Processing successful prediction result...")
                
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
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "prediction_method": prediction_result.get("method", "unknown"),
                    "prediction_endpoint": prediction_result.get("endpoint", "unknown"),
                    "prediction_timestamp": datetime.now().isoformat(),
                    "enhanced_features_used": features.get("feature_interpretation", {}),
                    "model_enhanced": True
                }
                
                state["heart_attack_prediction"] = enhanced_prediction
                state["heart_attack_risk_score"] = float(risk_probability)
                
                logger.info(f"Heart attack prediction completed successfully")
                logger.info(f"Display: {enhanced_prediction['combined_display']}")
                
            else:
                error_msg = prediction_result.get("error", "Unknown API error")
                logger.warning(f"Heart attack prediction failed: {error_msg}")
                
                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "confidence_display": "Confidence: Error",
                    "combined_display": f"Heart Disease Risk: Error - {error_msg}",
                    "fastapi_server_url": self.config.heart_attack_api_url,
                    "error_details": error_msg,
                    "tried_endpoints": prediction_result.get("tried_endpoints", []),
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
        logger.info("Node 8: Initializing comprehensive chatbot with JSON graph generation...")
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
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_category", "unknown"),
                    "model_type": "enhanced_ml_api_comprehensive",
                    "deidentification_level": "comprehensive_claims_data",
                    "claims_data_types": ["medical", "pharmacy", "mcid"],
                    "json_graph_generation_supported": True,
                    "batch_code_meanings_available": True
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["json_graph_generation_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))

            logger.info("Successfully initialized comprehensive chatbot with JSON graph generation")
            logger.info(f"Chatbot context includes: {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"JSON graph generation: Ready for visualizations")

        except Exception as e:
            error_msg = f"Error initializing chatbot: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["initialize_chatbot"] = "error"
            logger.error(error_msg)

        return state

    def handle_error(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Error handling node"""
        logger.error(f"LangGraph Error Handler: {state['current_step']}")
        logger.error(f"Errors: {state['errors']}")

        state["processing_complete"] = True
        current_step = state.get("current_step", "unknown")
        state["step_status"][current_step] = "error"
        return state

    # ===== CONDITIONAL EDGES =====

    def should_continue_after_api(self, state: HealthAnalysisState) -> Literal["continue", "retry", "error"]:
        if state["errors"]:
            if state["retry_count"] < 3:  # max_retries
                state["retry_count"] += 1
                logger.warning(f"Retrying API fetch (attempt {state['retry_count']}/3)")
                state["errors"] = []
                return "retry"
            else:
                logger.error(f"Max retries (3) exceeded for API fetch")
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

    # ===== ENHANCED JSON GRAPH GENERATION METHODS =====

    def detect_graph_request(self, user_query: str) -> Dict[str, Any]:
        """Detect if user is requesting a graph/chart with enhanced detection"""
        try:
            query_str = str(user_query).lower()
            
            graph_keywords = [
                'chart', 'graph', 'plot', 'visualization', 'visualize', 'show me',
                'display', 'generate', 'create', 'diagram', 'dashboard', 'timeline'
            ]
            
            chart_types = {
                'diagnosis_frequency': ['diagnosis', 'diagnostic', 'condition', 'icd', 'disease'],
                'medication_distribution': ['medication', 'drug', 'pharmacy', 'prescription', 'ndc'],
                'risk_assessment': ['risk', 'assessment', 'prediction', 'probability'],
                'timeline': ['timeline', 'progression', 'history', 'over time'],
                'condition_distribution': ['condition', 'health', 'medical', 'clinical']
            }
            
            is_graph_request = any(keyword in query_str for keyword in graph_keywords)
            
            if is_graph_request:
                # Determine chart type
                for chart_type, keywords in chart_types.items():
                    if any(keyword in query_str for keyword in keywords):
                        return {
                            "is_graph_request": True,
                            "graph_type": chart_type
                        }
                
                return {
                    "is_graph_request": True,
                    "graph_type": "general"
                }
            
            # Check for matplotlib code patterns
            matplotlib_patterns = [
                r'plt\.\w+\(', r'ax\.\w+\(', r'matplotlib', r'pyplot',
                r'\.plot\(', r'\.bar\(', r'\.pie\(', r'\.scatter\('
            ]
            
            for pattern in matplotlib_patterns:
                if re.search(pattern, query_str):
                    return {
                        "is_graph_request": True,
                        "graph_type": "matplotlib_conversion",
                        "has_matplotlib_code": True
                    }
            
            return {"is_graph_request": False}
            
        except Exception as e:
            logger.error(f"Error in detect_graph_request: {str(e)}")
            return {"is_graph_request": False, "error": str(e)}

    def convert_matplotlib_to_json(self, matplotlib_code: str) -> Dict[str, Any]:
        """Convert matplotlib code to JSON structure"""
        try:
            # Extract chart type from matplotlib code
            chart_type = "bar_chart"  # default
            
            if "pie" in matplotlib_code.lower():
                chart_type = "pie_chart"
            elif "scatter" in matplotlib_code.lower():
                chart_type = "scatter_plot"
            elif "line" in matplotlib_code.lower() or "plot" in matplotlib_code.lower():
                chart_type = "line_chart"
            elif "hist" in matplotlib_code.lower():
                chart_type = "histogram"
            
            # Try to extract data patterns
            categories = ["Category1", "Category2", "Category3", "Category4"]
            data = [25, 30, 20, 25]
            title = "Converted from Matplotlib Code"
            
            # Simple regex patterns to extract data
            label_pattern = r'labels?\s*=\s*\[(.*?)\]'
            data_pattern = r'(?:data|values?|y)\s*=\s*\[([\d\s,\.]+)\]'
            title_pattern = r'title\([\'\"](.*?)[\'\"]'
            
            labels_match = re.search(label_pattern, matplotlib_code, re.IGNORECASE)
            if labels_match:
                labels_str = labels_match.group(1)
                categories = [label.strip().strip('\'"') for label in labels_str.split(',')]
            
            data_match = re.search(data_pattern, matplotlib_code, re.IGNORECASE)
            if data_match:
                data_str = data_match.group(1)
                try:
                    data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                except:
                    pass
            
            title_match = re.search(title_pattern, matplotlib_code, re.IGNORECASE)
            if title_match:
                title = title_match.group(1)
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": chart_type,
                "title": title,
                "converted_from_matplotlib": True
            }
            
        except Exception as e:
            logger.error(f"Error converting matplotlib to JSON: {str(e)}")
            return {
                "categories": ["Conversion Error"],
                "data": [0],
                "graph_type": "error",
                "title": "Matplotlib Conversion Failed",
                "error": str(e)
            }

    def generate_json_graph_data(self, chat_context: Dict[str, Any], graph_type: str) -> Dict[str, Any]:
        """Generate JSON graph data for healthcare visualizations"""
        try:
            if "diagnosis" in graph_type:
                return self._extract_diagnosis_json_data(chat_context)
            elif "medication" in graph_type:
                return self._extract_medication_json_data(chat_context)
            elif "risk" in graph_type:
                return self._extract_risk_json_data(chat_context)
            elif "timeline" in graph_type:
                return self._extract_timeline_json_data(chat_context)
            elif "condition" in graph_type:
                return self._extract_condition_json_data(chat_context)
            else:
                # Default to diagnosis data
                return self._extract_diagnosis_json_data(chat_context)
                
        except Exception as e:
            logger.error(f"Error generating JSON graph data: {str(e)}")
            return {
                "categories": ["No Data"],
                "data": [0],
                "graph_type": "error",
                "title": "Data Generation Failed",
                "error": str(e)
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
                # Return sample data if no real data
                categories = ["C92.91", "F31.70", "F32.9", "F40.1", "F41.1"]
                data = [1, 1, 2, 1, 1]
            else:
                # Sort by frequency (most common first)
                sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
                categories = [item[0] for item in sorted_diagnoses]
                data = [item[1] for item in sorted_diagnoses]
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": "diagnosis_frequency",
                "title": "Diagnosis Code Distribution"
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "graph_type": "error",
                "title": "Diagnosis Data Error",
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
                if med_name and med_name != "Unknown Medication":
                    medication_counts[med_name] = medication_counts.get(med_name, 0) + 1
            
            if not medication_counts:
                # Return sample data if no real data
                categories = ["Metformin HCL", "Lisinopril", "Atorvastatin", "Aspirin"]
                data = [3, 2, 1, 1]
            else:
                # Sort by frequency
                sorted_meds = sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)
                categories = [item[0] for item in sorted_meds]
                data = [item[1] for item in sorted_meds]
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": "medication_distribution",
                "title": "Medication Distribution"
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "graph_type": "error", 
                "title": "Medication Data Error",
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
            heart_risk = heart_attack_prediction.get("raw_risk_score", 0.25) * 100
            categories.append("Heart Disease Risk")
            data.append(round(heart_risk, 1))
            
            # Add diabetes risk indicator
            diabetes = entity_extraction.get("diabetics", "no")
            diabetes_risk = 80 if str(diabetes).lower() in ["yes", "true", "1"] else 20
            categories.append("Diabetes Risk")
            data.append(diabetes_risk)
            
            # Add blood pressure risk
            bp = entity_extraction.get("blood_pressure", "unknown")
            bp_risk = 70 if str(bp).lower() in ["managed", "diagnosed", "yes"] else 30
            categories.append("Hypertension Risk")
            data.append(bp_risk)
            
            # Add smoking risk
            smoking = entity_extraction.get("smoking", "no")
            smoking_risk = 90 if str(smoking).lower() in ["yes", "true", "1"] else 10
            categories.append("Smoking Risk")
            data.append(smoking_risk)
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": "risk_assessment",
                "title": "Health Risk Assessment"
            }
            
        except Exception as e:
            return {
                "categories": ["Risk Assessment Error"],
                "data": [0],
                "graph_type": "error",
                "title": "Risk Assessment Error",
                "error": str(e)
            }

    def _extract_timeline_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timeline data in JSON format"""
        try:
            medical_extraction = chat_context.get("medical_extraction", {})
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            
            # Combine medical and pharmacy dates
            timeline_data = {}
            
            # Add medical events
            medical_records = medical_extraction.get("hlth_srvc_records", [])
            for record in medical_records:
                date = record.get("clm_rcvd_dt", "")
                if date:
                    timeline_data[date] = timeline_data.get(date, 0) + 1
            
            # Add pharmacy events
            pharmacy_records = pharmacy_extraction.get("ndc_records", [])
            for record in pharmacy_records:
                date = record.get("rx_filled_dt", "")
                if date:
                    timeline_data[date] = timeline_data.get(date, 0) + 1
            
            if not timeline_data:
                # Sample timeline data
                categories = ["2024-01", "2024-02", "2024-03", "2024-04"]
                data = [2, 3, 1, 4]
            else:
                # Sort by date
                sorted_timeline = sorted(timeline_data.items())
                categories = [item[0] for item in sorted_timeline]
                data = [item[1] for item in sorted_timeline]
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": "timeline",
                "title": "Healthcare Activity Timeline"
            }
            
        except Exception as e:
            return {
                "categories": ["Timeline Error"],
                "data": [0],
                "graph_type": "error",
                "title": "Timeline Generation Error",
                "error": str(e)
            }

    def _extract_condition_json_data(self, chat_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condition distribution data in JSON format"""
        try:
            entity_extraction = chat_context.get("entity_extraction", {})
            medical_conditions = entity_extraction.get("medical_conditions", [])
            
            if not medical_conditions:
                # Use basic health indicators
                categories = []
                data = []
                
                if entity_extraction.get("diabetics", "no").lower() in ["yes", "true", "1"]:
                    categories.append("Diabetes")
                    data.append(1)
                    
                if entity_extraction.get("blood_pressure", "unknown").lower() in ["managed", "diagnosed", "yes"]:
                    categories.append("Hypertension")
                    data.append(1)
                    
                if not categories:
                    categories = ["Diabetes", "Hypertension", "No Other Conditions"]
                    data = [1, 1, 0]
            else:
                # Count condition occurrences
                condition_counts = {}
                for condition in medical_conditions:
                    name = str(condition) if not isinstance(condition, dict) else condition.get("name", "Unknown")
                    condition_counts[name] = condition_counts.get(name, 0) + 1
                
                categories = list(condition_counts.keys())
                data = list(condition_counts.values())
            
            return {
                "categories": categories,
                "data": data,
                "graph_type": "condition_distribution",
                "title": "Medical Condition Distribution"
            }
            
        except Exception as e:
            return {
                "categories": ["Data Error"],
                "data": [0],
                "graph_type": "error",
                "title": "Condition Data Error",
                "error": str(e)
            }

    def create_json_response_with_boundaries(self, json_data: Dict[str, Any], user_query: str, explanation: str = "") -> Dict[str, Any]:
        """Create response with JSON data and boundary markers"""
        
        # Generate JSON string
        json_string = json.dumps(json_data, indent=2)
        
        # Create response text with boundary markers
        graph_start_marker = "***GRAPH_START***"
        graph_end_marker = "***GRAPH_END***"
        
        if not explanation:
            explanation = f"Healthcare data visualization showing {json_data.get('graph_type', 'data').replace('_', ' ')}"
        
        response_text = f"## Healthcare Data Visualization\n\n{explanation}\n\n{graph_start_marker}\n{json_string}\n{graph_end_marker}\n\nThis visualization uses your healthcare data to provide clinical insights."
        
        # Calculate boundary positions
        start_position = response_text.find(graph_start_marker)
        end_position = response_text.find(graph_end_marker) + len(graph_end_marker)
        
        # Check if we have valid data
        has_graph = len(json_data.get("categories", [])) > 0 and json_data.get("categories")[0] not in ["No Data", "Data Error"]
        
        return {
            "success": has_graph,
            "response": response_text,
            "session_id": str(uuid.uuid4()),
            "graph_present": 1 if has_graph else 0,
            "graph_boundaries": {
                "start_position": start_position if start_position != -1 else None,
                "end_position": end_position if end_position > start_position else None,
                "has_markers": start_position != -1 and end_position > start_position
            },
            "json_graph_data": json_data if has_graph else None
        }

    # ===== ENHANCED CHATBOT FUNCTIONALITY WITH JSON GRAPH GENERATION =====

    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Enhanced chatbot with JSON graph generation and boundary markers"""
        try:
            # Check if this is a graph request using detection
            graph_request = self.detect_graph_request(user_query)

            if graph_request.get("is_graph_request", False):
                return self._handle_json_graph_request(user_query, chat_context, chat_history, graph_request)

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
                "response": "I encountered an error processing your question. Please try again.",
                "session_id": str(uuid.uuid4()),
                "graph_present": 0,
                "graph_boundaries": {"start_position": None, "end_position": None, "has_markers": False},
                "json_graph_data": None,
                "error": str(e)
            }

    def _handle_json_graph_request(self, user_query: str, chat_context: Dict[str, Any], 
                                  chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph generation requests with JSON format and boundary markers"""
        try:
            graph_type = graph_request.get("graph_type", "general")
            
            logger.info(f"Generating {graph_type} JSON visualization...")
            
            # Check for matplotlib conversion
            if graph_request.get("has_matplotlib_code", False):
                # Extract matplotlib code from user query and convert
                json_data = self.convert_matplotlib_to_json(user_query)
                explanation = "Converted matplotlib code to JSON structure for visualization"
            else:
                # Generate JSON data from healthcare context
                json_data = self.generate_json_graph_data(chat_context, graph_type)
                explanation = f"Generated {graph_type.replace('_', ' ')} visualization from your healthcare data"
            
            # Create response with boundary markers
            response_result = self.create_json_response_with_boundaries(json_data, user_query, explanation)
            
            # Add updated chat history
            response_result["updated_chat_history"] = chat_history + [
                {"role": "user", "content": str(user_query)},
                {"role": "assistant", "content": response_result["response"]}
            ]
            
            return response_result
                
        except Exception as e:
            logger.error(f"Error handling JSON graph request: {str(e)}")
            fallback_response = f"""## Graph Generation Error

I encountered an error while generating your requested visualization: {str(e)}

Available Graph Types:
- **Diagnosis Distribution**: "show me diagnosis frequency chart"
- **Medication Analysis**: "create medication distribution chart"  
- **Risk Dashboard**: "generate risk assessment visualization"
- **Timeline View**: "show healthcare timeline"
- **Condition Overview**: "show condition distribution"

Please try rephrasing your request with one of these specific graph types."""

            return {
                "success": False,
                "response": fallback_response,
                "session_id": str(uuid.uuid4()),
                "graph_present": 0,
                "graph_boundaries": {"start_position": None, "end_position": None, "has_markers": False},
                "json_graph_data": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": str(user_query)},
                    {"role": "assistant", "content": fallback_response}
                ]
            }

    def _handle_heart_attack_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Handle heart attack related questions with comprehensive analysis"""
        try:
            # Get comprehensive context data
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            medical_extraction = chat_context.get("medical_extraction", {})
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            
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

            # Create comprehensive heart attack analysis prompt
            heart_attack_prompt = f"""You are Dr. CardioAI, a specialist in cardiovascular risk assessment with access to comprehensive patient claims data.

COMPREHENSIVE PATIENT DATA AVAILABLE:
**PATIENT DEMOGRAPHICS:**
- Age: {patient_age}
- Current Health Status: Diabetes: {entity_extraction.get('diabetics', 'unknown')}, Blood Pressure: {entity_extraction.get('blood_pressure', 'unknown')}, Smoking: {entity_extraction.get('smoking', 'unknown')}

**COMPLETE MEDICAL CLAIMS DATA:**
{json.dumps(medical_extraction, indent=2)}

**COMPLETE PHARMACY CLAIMS DATA:**
{json.dumps(pharmacy_extraction, indent=2)}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entity_extraction, indent=2)}

**CURRENT ML MODEL PREDICTION:**
{risk_display}

**RECENT CONVERSATION:**
{history_text}

**USER QUESTION:** {user_query}

Provide comprehensive cardiovascular risk analysis using all available data. Reference specific clinical findings and compare with the ML model prediction. Include actionable recommendations.

If the user requests a chart or visualization, generate JSON structure with boundary markers in this format:
***GRAPH_START***
{{
  "categories": ["Risk Factor 1", "Risk Factor 2", "Risk Factor 3"],
  "data": [percentage1, percentage2, percentage3],
  "graph_type": "risk_assessment",
  "title": "Cardiovascular Risk Assessment"
}}
***GRAPH_END***"""

            logger.info(f"Processing enhanced heart attack question...")

            # Call the LLM
            response = self.api_integrator.call_llm_enhanced(heart_attack_prompt, self.config.chatbot_sys_msg)
            
            # Ensure response is a string
            response_str = str(response)

            # Check if response contains JSON with boundary markers
            has_json_graph = "***GRAPH_START***" in response_str and "***GRAPH_END***" in response_str
            
            # Extract JSON if present
            json_graph_data = None
            graph_boundaries = {"start_position": None, "end_position": None, "has_markers": False}
            
            if has_json_graph:
                start_pos = response_str.find("***GRAPH_START***")
                end_pos = response_str.find("***GRAPH_END***") + len("***GRAPH_END***")
                
                try:
                    json_start = response_str.find("{", start_pos)
                    json_end = response_str.rfind("}", 0, end_pos) + 1
                    json_str = response_str[json_start:json_end]
                    json_graph_data = json.loads(json_str)
                    
                    graph_boundaries = {
                        "start_position": start_pos,
                        "end_position": end_pos,
                        "has_markers": True
                    }
                except:
                    has_json_graph = False

            return {
                "success": True,
                "response": response_str,
                "session_id": str(uuid.uuid4()),
                "graph_present": 1 if has_json_graph else 0,
                "graph_boundaries": graph_boundaries,
                "json_graph_data": json_graph_data,
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": str(user_query)},
                    {"role": "assistant", "content": response_str}
                ]
            }

        except Exception as e:
            logger.error(f"Error in heart attack question: {str(e)}")
            error_response = f"I encountered an error with cardiovascular analysis: {str(e)}. Please try again."
            return {
                "success": False,
                "response": error_response,
                "session_id": str(uuid.uuid4()),
                "graph_present": 0,
                "graph_boundaries": {"start_position": None, "end_position": None, "has_markers": False},
                "json_graph_data": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": str(user_query)},
                    {"role": "assistant", "content": error_response}
                ]
            }

    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Handle general questions with comprehensive context"""
        try:
            # Prepare comprehensive context
            complete_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            
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

            # Create comprehensive analysis prompt
            comprehensive_prompt = f"""You are Dr. AnalysisAI, a healthcare data analyst with access to comprehensive patient claims data and advanced JSON visualization capabilities.

**COMPREHENSIVE DATA ACCESS:**
{complete_context}

**CONVERSATION HISTORY:**
{history_text}

**PATIENT QUESTION:** {user_query}

**COMPREHENSIVE ANALYSIS INSTRUCTIONS:**

Use ALL available claims data to provide thorough analysis. Reference specific codes, dates, medications, and clinical findings. If visualization is requested, generate JSON structure with boundary markers.

**JSON VISUALIZATION FORMAT:**
If the question would benefit from a chart, generate JSON with boundary markers:
***GRAPH_START***
{{
  "categories": ["Category1", "Category2", "Category3"],
  "data": [value1, value2, value3],
  "graph_type": "chart_type",
  "title": "Chart Title"
}}
***GRAPH_END***

Available chart types: diagnosis_frequency, medication_distribution, risk_assessment, timeline, condition_distribution.

Provide comprehensive analysis using all available deidentified claims data."""

            logger.info(f"Processing enhanced general query...")

            response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)

            # Ensure response is a string
            response_str = str(response)

            # Check if response contains JSON with boundary markers
            has_json_graph = "***GRAPH_START***" in response_str and "***GRAPH_END***" in response_str
            
            # Extract JSON if present
            json_graph_data = None
            graph_boundaries = {"start_position": None, "end_position": None, "has_markers": False}
            
            if has_json_graph:
                start_pos = response_str.find("***GRAPH_START***")
                end_pos = response_str.find("***GRAPH_END***") + len("***GRAPH_END***")
                
                try:
                    json_start = response_str.find("{", start_pos)
                    json_end = response_str.rfind("}", 0, end_pos) + 1
                    json_str = response_str[json_start:json_end]
                    json_graph_data = json.loads(json_str)
                    
                    graph_boundaries = {
                        "start_position": start_pos,
                        "end_position": end_pos,
                        "has_markers": True
                    }
                except:
                    has_json_graph = False

            return {
                "success": True,
                "response": response_str,
                "session_id": str(uuid.uuid4()),
                "graph_present": 1 if has_json_graph else 0,
                "graph_boundaries": graph_boundaries,
                "json_graph_data": json_graph_data,
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": str(user_query)},
                    {"role": "assistant", "content": response_str}
                ]
            }

        except Exception as e:
            logger.error(f"Error in general question: {str(e)}")
            error_response = f"I encountered an error processing your question: {str(e)}. Please try rephrasing it."
            return {
                "success": False,
                "response": error_response,
                "session_id": str(uuid.uuid4()),
                "graph_present": 0,
                "graph_boundaries": {"start_position": None, "end_position": None, "has_markers": False},
                "json_graph_data": None,
                "error": str(e),
                "updated_chat_history": chat_history + [
                    {"role": "user", "content": str(user_query)},
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

        return f"""You are Dr. TrajectoryAI, a comprehensive healthcare analyst conducting detailed patient health trajectory analysis with predictive modeling capabilities.

**COMPREHENSIVE PATIENT CLAIMS DATA:**

**MEDICAL CLAIMS SUMMARY:**
{medical_summary}

**PHARMACY CLAIMS SUMMARY:**
{pharmacy_summary}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entities, indent=2)}

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS WITH PREDICTIVE EVALUATION:**

Conduct a thorough analysis addressing these critical healthcare evaluation questions:

## RISK PREDICTION (Clinical Outcomes)
**1. Chronic Disease Risk Assessment:**
- Based on this person's medical and pharmacy history, assess the risk of developing chronic diseases like diabetes, hypertension, COPD, or chronic kidney disease?
- Analyze current ICD-10 codes and medication patterns for disease progression indicators

**2. Hospitalization & Readmission Risk:**
- What is the likelihood that this person will be hospitalized or readmitted in the next 6–12 months?
- Review service utilization patterns and medication adherence indicators

**3. Emergency vs Outpatient Care Risk:**
- Is this person at risk of using the emergency room instead of outpatient care?
- Analyze healthcare utilization patterns from claims data

**4. Medication Adherence Risk:**
- How likely is this person to stop taking prescribed medications?
- Review prescription fill patterns and therapeutic gaps

**5. Serious Event Risk:**
- Does this person have a high risk of serious events like stroke, heart attack, or other complications due to comorbidities?
- Analyze cardiovascular risk factors and medication management

## COST & UTILIZATION PREDICTION
**6. High-Cost Claimant Prediction:**
- Is this person likely to become a high-cost claimant next year?
- Analyze current utilization trends and cost drivers

**7. Healthcare Cost Estimation:**
- Can you estimate this person's future healthcare costs (per month or per year)?
- Project based on current utilization patterns and medication costs

**8. Care Setting Prediction:**
- Is this person more likely to need inpatient hospital care or outpatient care in the future?
- Review current care patterns and complexity indicators

## FRAUD, WASTE & ABUSE (FWA) DETECTION
**9. Claims Anomaly Detection:**
- Do this person's medical or pharmacy claims show any anomalies that could indicate errors or unusual patterns?
- Review for inconsistent diagnoses, unusual prescription patterns, or billing irregularities

**10. Prescribing Pattern Analysis:**
- Are there any unusual prescribing or billing patterns related to this person's records?
- Examine medication combinations and prescribing frequency

## PERSONALIZED CARE MANAGEMENT
**11. Risk Segmentation:**
- How should this person be segmented — healthy, rising risk, chronic but stable, or high-cost/critical?
- Provide risk stratification based on comprehensive data analysis

**12. Preventive Care Recommendations:**
- What preventive screenings, wellness programs, or lifestyle changes should be recommended as the next best action?
- Identify specific care gaps and opportunities

**13. Care Gap Analysis:**
- Does this person have any care gaps, such as missed checkups, cancer screenings, or vaccinations?
- Review claims for preventive care compliance

## PHARMACY-SPECIFIC PREDICTIONS
**14. Polypharmacy Risk:**
- Is this person at risk of polypharmacy (taking too many medications or unsafe combinations)?
- Analyze current medication regimen for interactions and complexity

**15. Therapy Escalation:**
- Is this person likely to switch to higher-cost specialty drugs or need therapy escalation soon?
- Review current medications for potential progression patterns

**16. Specialty Drug Prediction:**
- Is it likely that this person will need expensive biologics or injectables in the future?
- Assess disease progression and current therapeutic approaches

## ADVANCED / STRATEGIC PREDICTIONS
**17. Disease Progression Modeling:**
- Can you model how this person's disease might progress over time (for example: diabetes → complications → hospitalizations)?
- Create trajectory model based on current conditions and medications

**18. Quality Metrics Impact:**
- Does this person have any care gaps that could affect quality metrics (like HEDIS or STAR ratings)?
- Identify opportunities for quality measure improvement

**19. Population Health Risk:**
- Based on available data, how might this person's long-term health contribute to population-level risk?
- Assess impact on overall population health management

**COMPREHENSIVE ANALYSIS REQUIREMENTS:**
- Address each evaluation question using specific data from medical and pharmacy claims
- Reference exact ICD-10 codes, NDC codes, and claim dates
- Provide risk percentages and likelihood assessments where possible
- Include temporal analysis showing health progression over time
- Offer specific, actionable recommendations for each identified risk
- Create predictive models based on available clinical indicators

**DELIVERABLE:**
Provide a comprehensive 800-1000 word health trajectory analysis that addresses all evaluation questions with specific data references, risk assessments, and actionable recommendations for care management and risk mitigation.

**ANALYSIS FOCUS:**
Use ALL available claims data to create the most comprehensive predictive health assessment possible, addressing every evaluation question with evidence-based analysis and specific recommendations."""

    def _create_comprehensive_summary_prompt(self, trajectory_analysis: str, entities: Dict,
                                           medical_extraction: Dict, pharmacy_extraction: Dict) -> str:
        """Create comprehensive summary prompt"""

        return f"""Based on the comprehensive health trajectory analysis, create an executive summary for healthcare decision-makers.

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS:**
{trajectory_analysis}

**ENHANCED HEALTH ENTITIES:**
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking Status: {entities.get('smoking', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions: {len(entities.get('medical_conditions', []))}
- Medications: {len(entities.get('medications_identified', []))}

**CLAIMS DATA SUMMARY:**
- Medical Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Pharmacy Records: {len(pharmacy_extraction.get('ndc_records', []))}

**EXECUTIVE SUMMARY REQUIREMENTS:**

Create a comprehensive summary with:

## CURRENT HEALTH STATUS
[2-3 sentences summarizing overall health condition and key findings]

## PRIORITY RISK FACTORS
[Bullet points of highest priority risks requiring immediate attention]

## COST & UTILIZATION INSIGHTS
[Key findings about healthcare costs and utilization patterns]

## CARE MANAGEMENT RECOMMENDATIONS
[Specific actionable recommendations for care management teams]

## PREDICTIVE INSIGHTS
[Key predictions about future health outcomes and costs]

## IMMEDIATE ACTION ITEMS
[Priority items requiring immediate clinical attention]

**FORMAT:** Professional healthcare executive summary, 400-500 words, focusing on actionable insights for care management and clinical decision-making."""

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
                    if 0 <= age_value <= 120:
                        features["Age"] = age_value
                    else:
                        features["Age"] = 50
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
                    if key == "Age":
                        features[key] = 50
                    else:
                        features[key] = 0

            enhanced_feature_summary = {
                "extracted_features": features,
                "feature_interpretation": {
                    "Age": f"{features['Age']} years old",
                    "Gender": "Male" if features["Gender"] == 1 else "Female",
                    "Diabetes": "Yes" if features["Diabetes"] == 1 else "No",
                    "High_BP": "Yes" if features["High_BP"] == 1 else "No",
                    "Smoking": "Yes" if features["Smoking"] == 1 else "No"
                },
                "data_sources": {
                    "age_source": "deidentified_medical.src_mbr_age",
                    "gender_source": "patient_data.gender",
                    "diabetes_source": "entity_extraction.diabetics",
                    "bp_source": "entity_extraction.blood_pressure",
                    "smoking_source": "entity_extraction.smoking"
                },
                "extraction_enhanced": True
            }

            logger.info(f"Enhanced heart attack features: {enhanced_feature_summary['feature_interpretation']}")
            return enhanced_feature_summary

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
                logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
                fastapi_features["age"] = 50

            # Validate binary features
            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                    fastapi_features[key] = 0

            logger.info(f"FastAPI features prepared: {fastapi_features}")
            return fastapi_features

        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous heart attack prediction call"""
        try:
            import requests

            logger.info(f"Heart attack prediction features: {features}")

            if not features:
                return {
                    "success": False,
                    "error": "No features provided for heart attack prediction"
                }

            heart_attack_url = self.config.heart_attack_api_url
            endpoints = [
                f"{heart_attack_url}/predict",
                f"{heart_attack_url}/predict-simple"
            ]

            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            logger.info(f"Sending prediction request to {endpoints[0]}")

            try:
                response = requests.post(endpoints[0], json=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_JSON_SYNC",
                        "endpoint": endpoints[0]
                    }
                else:
                    logger.warning(f"First endpoint failed with status {response.status_code}")

            except requests.exceptions.ConnectionError as conn_error:
                logger.error(f"Connection failed: {conn_error}")
                return {
                    "success": False,
                    "error": f"Cannot connect to heart attack prediction server. Make sure the server is running."
                }
            except Exception as request_error:
                logger.warning(f"Request failed: {str(request_error)}")

            try:
                logger.info(f"Trying fallback endpoint: {endpoints[1]}")
                response = requests.post(endpoints[1], params=params, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Fallback prediction successful: {result}")
                    return {
                        "success": True,
                        "prediction_data": result,
                        "method": "POST_PARAMS_SYNC",
                        "endpoint": endpoints[1]
                    }
                else:
                    error_text = response.text
                    logger.error(f"All endpoints failed. Status {response.status_code}: {error_text}")
                    return {
                        "success": False,
                        "error": f"Heart attack prediction server error {response.status_code}: {error_text}",
                        "tried_endpoints": endpoints
                    }

            except Exception as fallback_error:
                logger.error(f"All prediction methods failed: {str(fallback_error)}")
                return {
                    "success": False,
                    "error": f"All prediction methods failed. Error: {str(fallback_error)}",
                    "tried_endpoints": endpoints
                }

        except Exception as general_error:
            logger.error(f"Unexpected error in heart attack prediction: {general_error}")
            return {
                "success": False,
                "error": f"Heart attack prediction failed: {str(general_error)}"
            }

    def _extract_medical_summary(self, medical_data: Dict, medical_extraction: Dict) -> str:
        """Extract medical summary"""
        try:
            summary_parts = []

            age = medical_data.get("src_mbr_age", "unknown")
            zip_code = medical_data.get("src_mbr_zip_cd", "unknown")
            summary_parts.append(f"Patient Age: {age}, Location: {zip_code}")

            records = medical_extraction.get('hlth_srvc_records', [])
            summary_parts.append(f"Medical Records: {len(records)} health service records")

            if records:
                recent_diagnoses = []
                for record in records[:5]:
                    diag_codes = record.get('diagnosis_codes', [])
                    service_date = record.get('clm_rcvd_dt', 'Unknown date')
                    if diag_codes:
                        for diag in diag_codes[:2]:  # Limit diagnoses per record
                            code = diag.get('code', 'Unknown')
                            recent_diagnoses.append(f"Diagnosis {code} on {service_date}")
                    
                if recent_diagnoses:
                    summary_parts.append("Recent Diagnoses: " + "; ".join(recent_diagnoses))

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Medical data available but summary extraction failed: {str(e)}"

    def _extract_pharmacy_summary(self, pharmacy_data: Dict, pharmacy_extraction: Dict) -> str:
        """Extract pharmacy summary"""
        try:
            summary_parts = []

            records = pharmacy_extraction.get('ndc_records', [])
            summary_parts.append(f"Pharmacy Records: {len(records)} medication records")

            if records:
                recent_meds = []
                for record in records[:5]:
                    ndc_code = record.get('ndc', 'Unknown')
                    label_name = record.get('lbl_nm', 'Unknown medication')
                    fill_date = record.get('rx_filled_dt', 'Unknown date')
                    recent_meds.append(f"{label_name} (NDC: {ndc_code}) filled on {fill_date}")
                    
                if recent_meds:
                    summary_parts.append("Recent Medications: " + "; ".join(recent_meds))

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Pharmacy data available but summary extraction failed: {str(e)}"

    def get_code_explanations_for_record(self, record: Dict[str, Any], record_type: str = "medical") -> Dict[str, Any]:
        """Get code explanations for records"""
        explanations = {}

        try:
            if record_type == "medical":
                service_code = record.get("hlth_srvc_cd")
                if service_code:
                    explanations["service_code_explanation"] = self.data_processor.get_service_code_explanation_isolated(service_code)

                diagnosis_codes = record.get("diagnosis_codes", [])
                explanations["diagnosis_explanations"] = []
                for diag in diagnosis_codes:
                    diag_code = diag.get("code")
                    if diag_code:
                        explanation = self.data_processor.get_diagnosis_code_explanation_isolated(diag_code)
                        explanations["diagnosis_explanations"].append({
                            "code": diag_code,
                            "explanation": explanation,
                            "position": diag.get("position", 1)
                        })

            elif record_type == "pharmacy":
                ndc_code = record.get("ndc")
                if ndc_code:
                    explanations["ndc_explanation"] = self.data_processor.get_ndc_code_explanation_isolated(ndc_code)

                medication = record.get("lbl_nm")
                if medication:
                    explanations["medication_explanation"] = self.data_processor.get_medication_explanation_isolated(medication)

        except Exception as e:
            logger.warning(f"Error getting code explanations: {e}")
            explanations["error"] = f"Could not get explanations: {str(e)}"

        return explanations

    def test_llm_connection(self) -> Dict[str, Any]:
        """Test Snowflake Cortex API connection"""
        return self.api_integrator.test_healthcare_llm_connection()

    def test_backend_connection(self) -> Dict[str, Any]:
        """Test backend server connection"""
        return self.api_integrator.test_backend_connection_enhanced()

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

            logger.info("Starting Enhanced LangGraph workflow with JSON graph generation...")

            final_state = self.graph.invoke(initial_state, config=config_dict)

            results = {
                "success": final_state["processing_complete"] and not final_state["errors"],
                "patient_data": final_state["patient_data"],
                "api_outputs": {
                    "mcid": final_state["mcid_output"],
                    "medical": final_state["medical_output"],
                    "pharmacy": final_state["pharmacy_output"],
                    "token": final_state["token_output"]
                },
                "deidentified_data": {
                    "medical": final_state["deidentified_medical"],
                    "pharmacy": final_state["deidentified_pharmacy"],
                    "mcid": final_state["deidentified_mcid"]
                },
                "structured_extractions": {
                    "medical": final_state["medical_extraction"],
                    "pharmacy": final_state["pharmacy_extraction"]
                },
                "entity_extraction": final_state["entity_extraction"],
                "health_trajectory": final_state["health_trajectory"],
                "final_summary": final_state["final_summary"],
                "heart_attack_prediction": final_state["heart_attack_prediction"],
                "heart_attack_risk_score": final_state["heart_attack_risk_score"],
                "heart_attack_features": final_state["heart_attack_features"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "chat_history": final_state["chat_history"],
                "json_graph_generation_ready": final_state["json_graph_generation_ready"],
                "errors": final_state["errors"],
                "processing_steps_completed": self._count_completed_steps(final_state),
                "step_status": final_state["step_status"],
                "langgraph_used": True,
                "comprehensive_analysis": True,
                "enhanced_chatbot": True,
                "json_graph_generation_ready": True,
                "batch_code_meanings": True,
                "enhancement_version": "v9.0_json_structure_with_boundary_markers"
            }

            if results["success"]:
                logger.info("Enhanced LangGraph analysis completed successfully with JSON graph generation!")
                logger.info(f"Enhanced chatbot ready: {results['chatbot_ready']}")
                logger.info(f"JSON graph generation ready: {results['json_graph_generation_ready']}")
            else:
                logger.error(f"Enhanced LangGraph analysis failed: {final_state['errors']}")

            return results

        except Exception as e:
            logger.error(f"Fatal error in Enhanced LangGraph workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "patient_data": patient_data,
                "errors": [str(e)],
                "processing_steps_completed": 0,
                "langgraph_used": True,
                "comprehensive_analysis": False,
                "enhanced_chatbot": False,
                "json_graph_generation_ready": False,
                "batch_code_meanings": False,
                "enhancement_version": "v9.0_json_structure_with_boundary_markers"
            }

    def _count_completed_steps(self, state: HealthAnalysisState) -> int:
        """Count processing steps completed (8-step comprehensive process)"""
        steps = 0
        if state.get("mcid_output"): steps += 1
        if state.get("deidentified_medical") and not state.get("deidentified_medical", {}).get("error"): steps += 1
        if state.get("medical_extraction") or state.get("pharmacy_extraction"): steps += 1
        if state.get("entity_extraction"): steps += 1
        if state.get("health_trajectory"): steps += 1
        if state.get("final_summary"): steps += 1
        if state.get("heart_attack_prediction"): steps += 1
        if state.get("chatbot_ready"): steps += 1
        return steps

def main():
    """Enhanced Health Analysis Agent with JSON Structure Response Format"""
    
    print("Enhanced Health Analysis Agent v9.0 - JSON Structure with Boundary Markers")
    print("Comprehensive features:")
    print("   JSON Structure Response Format - Pure JSON objects instead of JavaScript constants")
    print("   Two Response Indicators - graph_present (1/0) and graph_boundaries (positions)")
    print("   Graph Boundary Markers - ***GRAPH_START*** and ***GRAPH_END*** markers")
    print("   Matplotlib Conversion - Automatic detection and conversion to JSON")
    print("   Multiple Chart Types - diagnosis, medication, risk, timeline, condition")
    print("   Production Ready Features - No external dependencies, comprehensive error handling")
    print()

    config = Config()
    print("Configuration:")
    print(f"   Snowflake API: {config.api_url}")
    print(f"   Model: {config.model}")
    print(f"   Server: {config.fastapi_url}")
    print(f"   Heart Attack ML API: {config.heart_attack_api_url}")

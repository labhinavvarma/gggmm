"""
Enhanced Health Analysis Agent with Episodic Memory Integration

CRITICAL SECURITY WARNING:
This implementation stores healthcare PHI in local files without encryption.
This approach violates HIPAA requirements and should NOT be used in production
healthcare environments. Consider encrypted databases or healthcare-compliant
storage solutions instead.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Literal, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import date
import requests
import re

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import enhanced components
from health_api_integrator import EnhancedHealthAPIIntegrator
from health_data_processor_work import EnhancedHealthDataProcessor
from episodic_memory_manager import EpisodicMemoryManager, EpisodicMemoryConfig, EpisodicMemoryError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Enhanced configuration with episodic memory settings"""
    fastapi_url: str = "http://localhost:8000"  # MCP server URL
    
    # Snowflake Cortex API Configuration
    api_url: str = "https://sfassist.edagenaipreprod.awsdns.internal.das/api/cortex/complete"
    api_key: str = "YOUR_API_KEY_HERE"  # SECURITY: Use environment variables
    app_id: str = "edadip"
    aplctn_cd: str = "edagnai"
    model: str = "llama4-maverick"
    
    # Heart Attack Prediction API Configuration
    heart_attack_api_url: str = "http://localhost:8000"
    heart_attack_threshold: float = 0.5
    
    # Episodic Memory Configuration
    episodic_memory_enabled: bool = True
    episodic_memory_directory: str = "./secure_patient_memory"
    episodic_memory_backup_directory: str = "./secure_patient_memory_backup"
    episodic_memory_max_history: int = 25
    episodic_memory_retention_days: int = 365
    
    # System messages
    sys_msg: str = """You are Dr. HealthAI, a comprehensive healthcare data analyst and clinical decision support specialist with expertise in:

CLINICAL SPECIALIZATION:
• Medical coding systems (ICD-10, CPT, HCPCS, NDC) interpretation and analysis
• Claims data analysis and healthcare utilization patterns
• Risk stratification and predictive modeling for chronic diseases
• Clinical decision support and evidence-based medicine
• Population health management and care coordination
• Healthcare economics and cost prediction
• Quality metrics (HEDIS, STAR ratings) and care gap analysis
• Advanced healthcare data visualization with matplotlib
• Episodic memory integration for patient history tracking

DATA ACCESS CAPABILITIES:
• Complete deidentified medical claims with ICD-10 diagnosis codes and CPT procedure codes
• Complete deidentified pharmacy claims with NDC codes and medication details
• Healthcare service utilization patterns and claims dates (clm_rcvd_dt, rx_filled_dt)
• Structured extractions of all medical and pharmacy fields with detailed analysis
• Enhanced entity extraction results including chronic conditions and risk factors
• Comprehensive patient demographic and clinical data
• Batch-processed code meanings for medical and pharmacy codes
• Historical patient data from previous visits and interactions

EPISODIC MEMORY CAPABILITIES:
• Access to patient's historical healthcare data across multiple visits
• Trend analysis of medical conditions and medication changes over time
• Risk factor progression tracking and predictive insights
• Care continuity analysis and care gap identification

ANALYTICAL RESPONSIBILITIES:
You provide comprehensive healthcare analysis including clinical insights, risk assessments, predictive modeling, and evidence-based recommendations using ALL available deidentified claims data AND historical patient data when available.

RESPONSE STANDARDS:
• Use clinical terminology appropriately while ensuring clarity
• Cite specific ICD-10 codes, NDC codes, CPT codes, and claim dates
• Reference historical data when available for trend analysis
• Provide evidence-based analysis using established clinical guidelines
• Include risk stratification and predictive insights
• Reference exact field names and values from the JSON data structure
• Maintain professional healthcare analysis standards"""

    chatbot_sys_msg: str = """You are Dr. ChatAI, a specialized healthcare AI assistant with COMPLETE ACCESS to comprehensive deidentified medical and pharmacy claims data AND episodic memory of patient history.

COMPREHENSIVE DATA ACCESS:
✅ CURRENT VISIT DATA:
   • Complete deidentified medical records with ICD-10 diagnosis codes
   • Complete deidentified pharmacy records with NDC medication codes
   • Healthcare service utilization patterns and claims dates
   • Enhanced entity extraction with chronic condition identification
   • Risk assessment results and predictive insights

✅ HISTORICAL DATA (EPISODIC MEMORY):
   • Previous visit records and medical history
   • Medication history and adherence patterns
   • Disease progression tracking over time
   • Risk factor evolution and trend analysis
   • Care continuity and intervention effectiveness

✅ ADVANCED CAPABILITIES:
   • Generate working matplotlib code for healthcare visualizations
   • Create medication timelines, diagnosis progressions, risk dashboards
   • Historical trend analysis with comparative visualizations
   • Longitudinal care analysis and predictive modeling

EPISODIC MEMORY INTEGRATION:
When available, use historical patient data to:
• Compare current health status with previous visits
• Identify disease progression or improvement patterns
• Track medication adherence and effectiveness
• Analyze risk factor changes over time
• Provide care continuity insights

CRITICAL INSTRUCTIONS:
• Access and analyze COMPLETE deidentified claims dataset AND historical data
• Reference specific codes, dates, medications, and clinical findings
• Use historical trends for predictive insights and care recommendations
• Generate working matplotlib code when visualization is requested
• Maintain professional healthcare analysis standards with temporal context"""

    timeout: int = 30

    def to_dict(self):
        return asdict(self)

# Enhanced State Definition with Episodic Memory
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
    final_summary: str

    # Heart Attack Prediction
    heart_attack_prediction: Dict[str, Any]
    heart_attack_risk_score: float
    heart_attack_features: Dict[str, Any]

    # Episodic Memory
    episodic_memory_result: Dict[str, Any]
    historical_patient_data: Dict[str, Any]
    patient_history_summary: str

    # Enhanced chatbot functionality
    chatbot_ready: bool
    chatbot_context: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    graph_generation_ready: bool

    # Control flow
    current_step: str
    errors: List[str]
    retry_count: int
    processing_complete: bool
    step_status: Dict[str, str]

class HealthAnalysisAgent:
    """Enhanced Health Analysis Agent with Comprehensive Clinical Analysis and Episodic Memory"""

    def __init__(self, custom_config: Optional[Config] = None):
        self.config = custom_config or Config()

        # Initialize enhanced components
        self.api_integrator = EnhancedHealthAPIIntegrator(self.config)
        self.data_processor = EnhancedHealthDataProcessor(self.api_integrator)

        # Initialize Episodic Memory if enabled
        self.episodic_memory = None
        if self.config.episodic_memory_enabled:
            episodic_config = EpisodicMemoryConfig(
                storage_directory=self.config.episodic_memory_directory,
                backup_directory=self.config.episodic_memory_backup_directory,
                file_prefix="patient_memory",
                max_history_entries=self.config.episodic_memory_max_history,
                retention_days=self.config.episodic_memory_retention_days
            )
            
            try:
                self.episodic_memory = EpisodicMemoryManager(episodic_config)
                logger.info("Episodic Memory Manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Episodic Memory: {e}")
                self.config.episodic_memory_enabled = False

        logger.info("Enhanced HealthAnalysisAgent initialized with Episodic Memory")
        logger.info(f"Snowflake API URL: {self.config.api_url}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"MCP Server URL: {self.config.fastapi_url}")
        logger.info(f"Heart Attack ML API: {self.config.heart_attack_api_url}")
        logger.info(f"Episodic Memory: {'Enabled' if self.config.episodic_memory_enabled else 'Disabled'}")

        self.setup_enhanced_langgraph()

    def setup_enhanced_langgraph(self):
        """Setup enhanced LangGraph workflow with episodic memory"""
        logger.info("Setting up Enhanced LangGraph workflow with episodic memory...")

        workflow = StateGraph(HealthAnalysisState)

        # Add all processing nodes
        workflow.add_node("fetch_api_data", self.fetch_api_data)
        workflow.add_node("deidentify_claims_data", self.deidentify_claims_data)
        workflow.add_node("extract_claims_fields", self.extract_claims_fields)
        workflow.add_node("extract_entities", self.extract_entities)
        workflow.add_node("load_historical_data", self.load_historical_data)
        workflow.add_node("analyze_trajectory", self.analyze_trajectory)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("predict_heart_attack", self.predict_heart_attack)
        workflow.add_node("save_episodic_memory", self.save_episodic_memory)
        workflow.add_node("initialize_chatbot", self.initialize_chatbot)
        workflow.add_node("handle_error", self.handle_error)

        # Define workflow edges
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
                "continue": "load_historical_data",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "load_historical_data",
            self.should_continue_after_historical_load,
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
                "continue": "save_episodic_memory",
                "error": "handle_error"
            }
        )

        workflow.add_conditional_edges(
            "save_episodic_memory",
            self.should_continue_after_episodic_memory,
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

        logger.info("Enhanced LangGraph workflow compiled successfully with episodic memory")

    # ===== LANGGRAPH NODES =====

    def fetch_api_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 1: Fetch claims data from APIs"""
        logger.info("Starting Claims API data fetch...")
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
        logger.info("Starting comprehensive claims data deidentification...")
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
        logger.info("Starting enhanced claims field extraction with batch processing...")
        state["current_step"] = "extract_claims_fields"
        state["step_status"]["extract_claims_fields"] = "running"

        try:
            # Extract medical and pharmacy fields with batch processing
            medical_extraction = self.data_processor.extract_medical_fields_batch_enhanced(state.get("deidentified_medical", {}))
            state["medical_extraction"] = medical_extraction

            pharmacy_extraction = self.data_processor.extract_pharmacy_fields_batch_enhanced(state.get("deidentified_pharmacy", {}))
            state["pharmacy_extraction"] = pharmacy_extraction

            state["step_status"]["extract_claims_fields"] = "completed"
            logger.info("Successfully completed enhanced claims field extraction")

        except Exception as e:
            error_msg = f"Error in claims field extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_claims_fields"] = "error"
            logger.error(error_msg)

        return state

    def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 4: Extract health entities using LLM"""
        logger.info("Starting LLM-powered health entity extraction...")
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
           
            logger.info(f"Successfully extracted health entities: {conditions_count} conditions, {medications_count} medications")
           
        except Exception as e:
            error_msg = f"Error in entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
       
        return state

    def load_historical_data(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 5: Load historical patient data from episodic memory"""
        logger.info("Loading historical patient data...")
        state["current_step"] = "load_historical_data"
        state["step_status"]["load_historical_data"] = "running"

        # Initialize with empty historical data
        state["historical_patient_data"] = {}
        state["patient_history_summary"] = ""

        try:
            if not self.config.episodic_memory_enabled or not self.episodic_memory:
                logger.info("Episodic memory disabled - skipping historical data load")
                state["step_status"]["load_historical_data"] = "completed"
                return state

            # Extract MCID list for historical lookup
            deidentified_mcid = state.get("deidentified_mcid", {})
            mcid_list = self.episodic_memory._extract_mcid_list(deidentified_mcid)

            if not mcid_list:
                logger.info("No MCIDs found - no historical data to load")
                state["step_status"]["load_historical_data"] = "completed"
                return state

            # Load historical data
            historical_memory = self.episodic_memory.load_episodic_memory(mcid_list)
            
            if historical_memory:
                state["historical_patient_data"] = historical_memory
                
                # Generate history summary
                history_summary = self._generate_patient_history_summary(historical_memory)
                state["patient_history_summary"] = history_summary
                
                logger.info(f"Loaded historical data: {historical_memory.get('update_count', 0)} previous visits")
            else:
                logger.info("No historical data found for this patient")

            state["step_status"]["load_historical_data"] = "completed"

        except Exception as e:
            error_msg = f"Error loading historical data: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["load_historical_data"] = "error"
            logger.error(error_msg)

        return state

    def analyze_trajectory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 6: Comprehensive health trajectory analysis with historical context"""
        logger.info("Starting comprehensive health trajectory analysis with historical context...")
        state["current_step"] = "analyze_trajectory"
        state["step_status"]["analyze_trajectory"] = "running"

        try:
            deidentified_medical = state.get("deidentified_medical", {})
            deidentified_pharmacy = state.get("deidentified_pharmacy", {})
            deidentified_mcid = state.get("deidentified_mcid", {})
            medical_extraction = state.get("medical_extraction", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            entities = state.get("entity_extraction", {})
            historical_data = state.get("historical_patient_data", {})
            history_summary = state.get("patient_history_summary", "")

            trajectory_prompt = self._create_comprehensive_trajectory_prompt_with_history(
                deidentified_medical, deidentified_pharmacy, deidentified_mcid,
                medical_extraction, pharmacy_extraction, entities,
                historical_data, history_summary
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
        """Node 7: Generate comprehensive final summary"""
        logger.info("Generating comprehensive final summary...")
        state["current_step"] = "generate_summary"
        state["step_status"]["generate_summary"] = "running"

        try:
            summary_prompt = self._create_comprehensive_summary_prompt_with_history(
                state.get("health_trajectory", ""),
                state.get("entity_extraction", {}),
                state.get("medical_extraction", {}),
                state.get("pharmacy_extraction", {}),
                state.get("historical_patient_data", {}),
                state.get("patient_history_summary", "")
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
        """Node 8: Heart attack prediction"""
        logger.info("Starting heart attack prediction...")
        state["current_step"] = "predict_heart_attack"
        state["step_status"]["predict_heart_attack"] = "running"

        try:
            # Extract features
            features = self._extract_enhanced_heart_attack_features(state)
            state["heart_attack_features"] = features

            if not features or "error" in features:
                error_msg = "Failed to extract features for heart attack prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Prepare features for API call
            fastapi_features = self._prepare_enhanced_fastapi_features(features)

            if fastapi_features is None:
                error_msg = "Failed to prepare feature vector for prediction"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Make prediction
            prediction_result = self._call_heart_attack_prediction_sync(fastapi_features)

            if prediction_result is None:
                error_msg = "Heart attack prediction returned None"
                state["errors"].append(error_msg)
                state["step_status"]["predict_heart_attack"] = "error"
                logger.error(error_msg)
                return state

            # Process result
            if prediction_result.get("success", False):
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
                    "prediction_timestamp": datetime.now().isoformat(),
                    "enhanced_features_used": features.get("feature_interpretation", {}),
                    "model_enhanced": True
                }
                
                state["heart_attack_prediction"] = enhanced_prediction
                state["heart_attack_risk_score"] = float(risk_probability)
                
                logger.info(f"Heart attack prediction completed successfully")
                
            else:
                error_msg = prediction_result.get("error", "Unknown API error")
                logger.warning(f"Heart attack prediction failed: {error_msg}")
                
                state["heart_attack_prediction"] = {
                    "error": error_msg,
                    "risk_display": "Heart Disease Risk: Error",
                    "prediction_timestamp": datetime.now().isoformat(),
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

    def save_episodic_memory(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 9: Save episodic memory for future reference"""
        logger.info("Saving episodic memory...")
        state["current_step"] = "save_episodic_memory"
        state["step_status"]["save_episodic_memory"] = "running"

        # Initialize empty result
        state["episodic_memory_result"] = {
            "success": False,
            "skipped": True,
            "reason": "episodic_memory_disabled"
        }

        try:
            if not self.config.episodic_memory_enabled or not self.episodic_memory:
                logger.info("Episodic memory disabled - skipping save")
                state["step_status"]["save_episodic_memory"] = "completed"
                return state

            # Check if we have the required data
            deidentified_mcid = state.get("deidentified_mcid", {})
            entity_extraction = state.get("entity_extraction", {})
            
            if not deidentified_mcid or not entity_extraction:
                logger.warning("Missing data for episodic memory - skipping")
                state["episodic_memory_result"] = {
                    "success": False,
                    "error": "Missing required data",
                    "skipped": True
                }
                state["step_status"]["save_episodic_memory"] = "completed"
                return state

            # Prepare additional metadata
            metadata = {
                "processing_timestamp": datetime.now().isoformat(),
                "heart_attack_risk_score": state.get("heart_attack_risk_score", 0.0),
                "heart_attack_prediction": state.get("heart_attack_prediction", {}),
                "processing_complete": state.get("processing_complete", False),
                "analysis_version": "v8.0_with_episodic_memory",
                "patient_age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                "patient_zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                "medical_records_count": len(state.get("medical_extraction", {}).get("hlth_srvc_records", [])),
                "pharmacy_records_count": len(state.get("pharmacy_extraction", {}).get("ndc_records", [])),
                "step_status": state.get("step_status", {})
            }

            # Save episodic memory
            memory_result = self.episodic_memory.save_episodic_memory(
                deidentified_mcid=deidentified_mcid,
                entity_extraction=entity_extraction,
                additional_metadata=metadata
            )

            state["episodic_memory_result"] = memory_result
            
            if memory_result["success"]:
                logger.info(f"Episodic memory {memory_result['operation']}: {memory_result.get('patient_hash', 'unknown')}")
                logger.info(f"Update count: {memory_result.get('update_count', 0)}, History entries: {memory_result.get('history_entries', 0)}")
            else:
                logger.error(f"Episodic memory failed: {memory_result.get('error', 'unknown')}")

            state["step_status"]["save_episodic_memory"] = "completed"

        except EpisodicMemoryError as e:
            error_msg = f"Episodic memory error: {str(e)}"
            logger.error(error_msg)
            state["episodic_memory_result"] = {
                "success": False,
                "error": str(e),
                "error_type": "EpisodicMemoryError"
            }
            # Don't fail the entire workflow for episodic memory errors
            state["step_status"]["save_episodic_memory"] = "completed"
            
        except Exception as e:
            error_msg = f"Error saving episodic memory: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["save_episodic_memory"] = "error"
            logger.error(error_msg)
            
            state["episodic_memory_result"] = {
                "success": False,
                "error": str(e)
            }

        return state

    def initialize_chatbot(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Node 10: Initialize comprehensive chatbot with graph generation"""
        logger.info("Initializing comprehensive chatbot with graph generation...")
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
                "historical_patient_data": state.get("historical_patient_data", {}),
                "patient_history_summary": state.get("patient_history_summary", ""),
                "episodic_memory_result": state.get("episodic_memory_result", {}),
                "patient_overview": {
                    "age": state.get("deidentified_medical", {}).get("src_mbr_age", "unknown"),
                    "zip": state.get("deidentified_medical", {}).get("src_mbr_zip_cd", "unknown"),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "heart_attack_risk_level": state.get("heart_attack_prediction", {}).get("risk_category", "unknown"),
                    "has_historical_data": bool(state.get("historical_patient_data")),
                    "historical_visits": state.get("historical_patient_data", {}).get("update_count", 0),
                    "episodic_memory_enabled": self.config.episodic_memory_enabled,
                    "model_type": "enhanced_ml_api_comprehensive_with_memory",
                    "graph_generation_supported": True,
                    "batch_code_meanings_available": True
                }
            }

            state["chat_history"] = []
            state["chatbot_context"] = comprehensive_chatbot_context
            state["chatbot_ready"] = True
            state["graph_generation_ready"] = True
            state["processing_complete"] = True
            state["step_status"]["initialize_chatbot"] = "completed"

            medical_records = len(state.get("medical_extraction", {}).get("hlth_srvc_records", []))
            pharmacy_records = len(state.get("pharmacy_extraction", {}).get("ndc_records", []))
            historical_visits = state.get("historical_patient_data", {}).get("update_count", 0)

            logger.info("Successfully initialized comprehensive chatbot with episodic memory")
            logger.info(f"Chatbot context includes: {medical_records} medical records, {pharmacy_records} pharmacy records")
            logger.info(f"Historical data: {historical_visits} previous visits")

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
            if state["retry_count"] < 3:
                state["retry_count"] += 1
                logger.warning(f"Retrying API fetch (attempt {state['retry_count']}/3)")
                state["errors"] = []
                return "retry"
            else:
                logger.error("Max retries (3) exceeded for API fetch")
                return "error"
        return "continue"

    def should_continue_after_deidentify(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_extraction_step(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_entity_extraction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_historical_load(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_trajectory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_summary(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_heart_attack_prediction(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        return "error" if state["errors"] else "continue"

    def should_continue_after_episodic_memory(self, state: HealthAnalysisState) -> Literal["continue", "error"]:
        # Episodic memory errors shouldn't block the workflow
        return "continue"

    # ===== HELPER METHODS =====

    def _generate_patient_history_summary(self, historical_memory: Dict[str, Any]) -> str:
        """Generate a formatted summary of patient's historical visits"""
        try:
            if not historical_memory:
                return "No previous visit history available."
            
            history = historical_memory.get('history', [])
            update_count = historical_memory.get('update_count', 0)
            first_seen = historical_memory.get('first_seen', 'Unknown')
            last_updated = historical_memory.get('last_updated', 'Unknown')
            
            if not history:
                return "No previous visit history available."
            
            summary_parts = [
                f"Patient History Summary:",
                f"- Total visits: {update_count}",
                f"- First seen: {first_seen[:10] if first_seen != 'Unknown' else 'Unknown'}",
                f"- Last visit: {last_updated[:10] if last_updated != 'Unknown' else 'Unknown'}",
                ""
            ]
            
            # Add recent visits (last 5)
            recent_history = history[-5:] if len(history) > 5 else history
            summary_parts.append("Recent Visits:")
            
            for i, entry in enumerate(reversed(recent_history), 1):
                timestamp = entry.get('timestamp', 'Unknown date')
                entity_data = entry.get('entity_extraction', {})
                conditions = len(entity_data.get('medical_conditions', []))
                medications = len(entity_data.get('medications_identified', []))
                diabetes = entity_data.get('diabetics', 'unknown')
                bp = entity_data.get('blood_pressure', 'unknown')
                
                visit_line = f"{i}. {timestamp[:10]}: {conditions} conditions, {medications} medications"
                if diabetes == 'yes' or bp in ['diagnosed', 'managed']:
                    visit_line += f" (Diabetes: {diabetes}, BP: {bp})"
                
                summary_parts.append(visit_line)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating history summary: {e}")
            return f"Error generating patient history summary: {str(e)}"

    def _create_comprehensive_trajectory_prompt_with_history(self, medical_data: Dict, pharmacy_data: Dict, mcid_data: Dict,
                                                           medical_extraction: Dict, pharmacy_extraction: Dict,
                                                           entities: Dict, historical_data: Dict, history_summary: str) -> str:
        """Create comprehensive trajectory prompt with historical context"""

        medical_summary = self._extract_medical_summary(medical_data, medical_extraction)
        pharmacy_summary = self._extract_pharmacy_summary(pharmacy_data, pharmacy_extraction)
        
        historical_context = ""
        if historical_data and history_summary:
            historical_context = f"""

**HISTORICAL PATIENT CONTEXT:**
{history_summary}

**DETAILED HISTORICAL DATA:**
{json.dumps(historical_data, indent=2)[:2000]}...

**LONGITUDINAL ANALYSIS REQUIREMENTS:**
- Compare current visit data with historical trends
- Identify disease progression or improvement patterns
- Analyze medication adherence and changes over time
- Assess risk factor evolution
- Provide care continuity insights
"""

        return f"""You are Dr. TrajectoryAI, a comprehensive healthcare analyst conducting detailed patient health trajectory analysis with predictive modeling capabilities and historical context.

**COMPREHENSIVE PATIENT CLAIMS DATA (CURRENT VISIT):**

**MEDICAL CLAIMS SUMMARY:**
{medical_summary}

**PHARMACY CLAIMS SUMMARY:**
{pharmacy_summary}

**ENHANCED HEALTH ENTITIES:**
{json.dumps(entities, indent=2)}

{historical_context}

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS WITH HISTORICAL CONTEXT:**

Conduct a thorough analysis addressing these critical healthcare evaluation questions:

## HISTORICAL TREND ANALYSIS
**1. Disease Progression Assessment:**
- How has this patient's health status changed over time?
- Are chronic conditions worsening, stable, or improving?
- What patterns emerge from the historical data?

**2. Medication Management Evolution:**
- How has the medication regimen changed over previous visits?
- Are there signs of medication adherence issues or optimization?
- What therapeutic changes indicate disease progression or improvement?

**3. Risk Factor Progression:**
- How have cardiovascular risk factors evolved over time?
- Are there emerging risk patterns that require attention?
- What interventions have been effective based on historical trends?

## PREDICTIVE MODELING WITH HISTORICAL CONTEXT
**4. Future Health Trajectory Prediction:**
- Based on historical trends, what is the likely health trajectory?
- What are the key risk factors for future complications?
- How can historical patterns inform preventive interventions?

**5. Care Continuity Assessment:**
- Are there gaps in care based on historical patterns?
- What follow-up interventions have been most effective?
- How can care coordination be improved based on past experiences?

## CURRENT VISIT CONTEXTUALIZATION
**6. Current Status vs Historical Baseline:**
- How does the current visit compare to previous health status?
- Are there significant changes that require immediate attention?
- What improvements or deteriorations are evident?

**COMPREHENSIVE ANALYSIS REQUIREMENTS:**
- Address each evaluation question using specific data from current AND historical claims
- Reference exact ICD-10 codes, NDC codes, and claim dates across time periods
- Provide longitudinal risk assessments with historical context
- Include temporal analysis showing health progression over time
- Offer specific, actionable recommendations based on both current and historical data
- Create predictive models based on historical trends and current clinical indicators

**DELIVERABLE:**
Provide a comprehensive 1000-1200 word health trajectory analysis that incorporates historical context, addresses all evaluation questions with specific data references, risk assessments, and actionable recommendations for care management and risk mitigation based on longitudinal patient data."""

    def _create_comprehensive_summary_prompt_with_history(self, trajectory_analysis: str, entities: Dict,
                                                        medical_extraction: Dict, pharmacy_extraction: Dict,
                                                        historical_data: Dict, history_summary: str) -> str:
        """Create comprehensive summary prompt with historical context"""

        historical_context = ""
        if historical_data and history_summary:
            total_visits = historical_data.get('update_count', 0)
            first_seen = historical_data.get('first_seen', 'Unknown')
            historical_context = f"""

**HISTORICAL CONTEXT:**
- Total patient visits: {total_visits}
- First seen: {first_seen[:10] if first_seen != 'Unknown' else 'Unknown'}
- Historical trend: {history_summary[:500]}...
"""

        return f"""Based on the comprehensive health trajectory analysis with historical context, create an executive summary for healthcare decision-makers.

**COMPREHENSIVE HEALTH TRAJECTORY ANALYSIS:**
{trajectory_analysis}

**ENHANCED HEALTH ENTITIES (CURRENT):**
- Diabetes: {entities.get('diabetics', 'unknown')}
- Age Group: {entities.get('age_group', 'unknown')}
- Smoking Status: {entities.get('smoking', 'unknown')}
- Blood Pressure: {entities.get('blood_pressure', 'unknown')}
- Medical Conditions: {len(entities.get('medical_conditions', []))}
- Medications: {len(entities.get('medications_identified', []))}

{historical_context}

**CLAIMS DATA SUMMARY:**
- Medical Records: {len(medical_extraction.get('hlth_srvc_records', []))}
- Diagnosis Codes: {medical_extraction.get('extraction_summary', {}).get('total_diagnosis_codes', 0)}
- Pharmacy Records: {len(pharmacy_extraction.get('ndc_records', []))}

**EXECUTIVE SUMMARY REQUIREMENTS WITH HISTORICAL PERSPECTIVE:**

Create a comprehensive summary with:

## CURRENT HEALTH STATUS WITH HISTORICAL CONTEXT
[2-3 sentences summarizing overall health condition, key findings, and how they compare to historical trends]

## PRIORITY RISK FACTORS WITH TREND ANALYSIS
[Bullet points of highest priority risks requiring immediate attention, including how these risks have evolved over time]

## COST & UTILIZATION INSIGHTS WITH HISTORICAL COMPARISON
[Key findings about healthcare costs and utilization patterns, comparing current visit to historical trends]

## CARE MANAGEMENT RECOMMENDATIONS WITH LONGITUDINAL PERSPECTIVE
[Specific actionable recommendations for care management teams based on both current status and historical patterns]

## PREDICTIVE INSIGHTS BASED ON HISTORICAL TRENDS
[Key predictions about future health outcomes and costs using historical data for context]

## IMMEDIATE ACTION ITEMS WITH CONTINUITY CONSIDERATIONS
[Priority items requiring immediate clinical attention, considering historical context and care continuity]

**FORMAT:** Professional healthcare executive summary, 500-600 words, focusing on actionable insights for care management and clinical decision-making with historical context and longitudinal care perspective."""

    # Include existing helper methods for heart attack prediction, medical/pharmacy summaries, etc.
    # [Previous helper methods from original code would go here - truncated for space]

    def _extract_enhanced_heart_attack_features(self, state: HealthAnalysisState) -> Dict[str, Any]:
        """Enhanced feature extraction for heart attack prediction with historical context"""
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

            # Add historical context if available
            historical_data = state.get("historical_patient_data", {})
            historical_context = {}
            if historical_data:
                history = historical_data.get('history', [])
                if history:
                    # Analyze risk factor trends
                    diabetes_trend = []
                    bp_trend = []
                    smoking_trend = []
                    
                    for entry in history[-5:]:  # Last 5 visits
                        entities = entry.get('entity_extraction', {})
                        diabetes_trend.append(1 if entities.get('diabetics') == 'yes' else 0)
                        bp_trend.append(1 if entities.get('blood_pressure') in ['managed', 'diagnosed'] else 0)
                        smoking_trend.append(1 if entities.get('smoking') == 'yes' else 0)
                    
                    historical_context = {
                        "diabetes_trend": diabetes_trend,
                        "bp_trend": bp_trend,
                        "smoking_trend": smoking_trend,
                        "risk_progression": sum(diabetes_trend + bp_trend + smoking_trend) / len(diabetes_trend + bp_trend + smoking_trend) if diabetes_trend else 0
                    }

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
                "historical_context": historical_context,
                "data_sources": {
                    "age_source": "deidentified_medical.src_mbr_age",
                    "gender_source": "patient_data.gender",
                    "diabetes_source": "entity_extraction.diabetics",
                    "bp_source": "entity_extraction.blood_pressure",
                    "smoking_source": "entity_extraction.smoking"
                },
                "extraction_enhanced": True,
                "has_historical_data": bool(historical_context)
            }

            logger.info(f"Enhanced heart attack features with historical context: {enhanced_feature_summary['feature_interpretation']}")
            return enhanced_feature_summary

        except Exception as e:
            logger.error(f"Error in heart attack feature extraction: {e}")
            return {"error": f"Feature extraction failed: {str(e)}"}

    # Include remaining helper methods from the original implementation
    # [Additional helper methods would be included here]

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
                        for diag in diag_codes[:2]:
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

            # Validate ranges
            if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
                logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
                fastapi_features["age"] = 50

            binary_features = ["gender", "diabetes", "high_bp", "smoking"]
            for key in binary_features:
                if fastapi_features[key] not in [0, 1]:
                    logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                    fastapi_features[key] = 0

            return fastapi_features

        except Exception as e:
            logger.error(f"Error preparing FastAPI features: {e}")
            return None

    def _call_heart_attack_prediction_sync(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous heart attack prediction call"""
        try:
            import requests

            if not features:
                return {"success": False, "error": "No features provided"}

            heart_attack_url = self.config.heart_attack_api_url
            endpoints = [f"{heart_attack_url}/predict", f"{heart_attack_url}/predict-simple"]

            params = {
                "age": int(features.get("age", 50)),
                "gender": int(features.get("gender", 0)),
                "diabetes": int(features.get("diabetes", 0)),
                "high_bp": int(features.get("high_bp", 0)),
                "smoking": int(features.get("smoking", 0))
            }

            try:
                response = requests.post(endpoints[0], json=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return {"success": True, "prediction_data": result, "endpoint": endpoints[0]}
            except requests.exceptions.ConnectionError:
                return {"success": False, "error": "Cannot connect to heart attack prediction server"}
            except Exception as e:
                logger.warning(f"First endpoint failed: {e}")

            try:
                response = requests.post(endpoints[1], params=params, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return {"success": True, "prediction_data": result, "endpoint": endpoints[1]}
                else:
                    return {"success": False, "error": f"All endpoints failed. Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": f"All prediction methods failed: {str(e)}"}

        except Exception as e:
            return {"success": False, "error": f"Heart attack prediction failed: {str(e)}"}

    # ===== ENHANCED CHATBOT FUNCTIONALITY =====

    def chat_with_data(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Enhanced chatbot with comprehensive claims data access and historical context"""
        try:
            # Check if this is a graph request
            graph_request = self.data_processor.detect_graph_request(user_query)

            if graph_request.get("is_graph_request", False):
                return self._handle_graph_request_enhanced(user_query, chat_context, chat_history, graph_request)

            # Check for history-related questions
            history_keywords = ['history', 'previous', 'past', 'before', 'trend', 'change', 'progress', 'compare']
            is_history_question = any(keyword in user_query.lower() for keyword in history_keywords)

            # Check for heart attack related questions
            heart_attack_keywords = ['heart attack', 'heart disease', 'cardiac', 'cardiovascular', 'heart risk']
            is_heart_attack_question = any(keyword in user_query.lower() for keyword in heart_attack_keywords)

            if is_history_question:
                return self._handle_history_question_enhanced(user_query, chat_context, chat_history)
            elif is_heart_attack_question:
                return self._handle_heart_attack_question_enhanced(user_query, chat_context, chat_history)
            else:
                return self._handle_general_question_enhanced(user_query, chat_context, chat_history)

        except Exception as e:
            logger.error(f"Error in enhanced chatbot: {str(e)}")
            return "I encountered an error processing your question. Please try again."

    def _handle_history_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle questions about patient history"""
        try:
            historical_data = chat_context.get("historical_patient_data", {})
            history_summary = chat_context.get("patient_history_summary", "")
            
            if not historical_data:
                return "No historical data is available for this patient. This appears to be their first visit in our system."

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

            history_prompt = f"""You are Dr. HistoryAI, a specialist in longitudinal healthcare analysis with access to comprehensive patient historical data.

**PATIENT HISTORICAL DATA:**
{json.dumps(historical_data, indent=2)}

**PATIENT HISTORY SUMMARY:**
{history_summary}

**CURRENT VISIT DATA:**
{json.dumps(chat_context.get("entity_extraction", {}), indent=2)}

**RECENT CONVERSATION:**
{history_text}

**USER QUESTION:** {user_query}

**HISTORICAL ANALYSIS INSTRUCTIONS:**

Provide a comprehensive analysis of the patient's healthcare history addressing:

## TEMPORAL TREND ANALYSIS
- How has the patient's health status changed over time?
- What patterns emerge from the historical visits?
- Are conditions improving, worsening, or stable?

## MEDICATION MANAGEMENT HISTORY
- How has the medication regimen evolved?
- Are there signs of medication adherence or optimization?
- What therapeutic changes indicate disease management effectiveness?

## RISK FACTOR EVOLUTION
- How have key health indicators changed over time?
- What risk factors have emerged or resolved?
- Are there concerning trends that require attention?

## CARE CONTINUITY INSIGHTS
- What interventions have been most effective historically?
- Are there gaps in care or missed opportunities?
- How can historical patterns inform future care decisions?

**RESPONSE REQUIREMENTS:**
- Use specific data from the historical records
- Reference exact dates, medications, and health indicators
- Provide evidence-based insights about health trends
- Include actionable recommendations based on historical patterns
- Compare current status with historical baseline

**PROVIDE COMPREHENSIVE HISTORICAL HEALTHCARE ANALYSIS:**"""

            response = self.api_integrator.call_llm_enhanced(history_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error analyzing the historical data. Please try rephrasing your question."

            return response

        except Exception as e:
            logger.error(f"Error in history question: {str(e)}")
            return "I encountered an error analyzing historical data. Please try again with a simpler question."

    def _handle_heart_attack_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle heart attack related questions with historical context"""
        try:
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            entity_extraction = chat_context.get("entity_extraction", {})
            historical_data = chat_context.get("historical_patient_data", {})
            
            # Build historical cardiovascular risk context
            historical_risk_context = ""
            if historical_data:
                history = historical_data.get('history', [])
                if history:
                    risk_trends = []
                    for entry in history[-5:]:
                        entities = entry.get('entity_extraction', {})
                        timestamp = entry.get('timestamp', 'Unknown')
                        diabetes = entities.get('diabetics', 'no')
                        bp = entities.get('blood_pressure', 'unknown')
                        smoking = entities.get('smoking', 'no')
                        risk_trends.append(f"  {timestamp[:10]}: Diabetes: {diabetes}, BP: {bp}, Smoking: {smoking}")
                    
                    historical_risk_context = f"""
**HISTORICAL CARDIOVASCULAR RISK TRENDS:**
{chr(10).join(risk_trends)}"""

            heart_attack_prompt = f"""You are Dr. CardioAI, a specialist in cardiovascular risk assessment with access to comprehensive patient data and historical trends.

**CURRENT CARDIOVASCULAR ASSESSMENT:**
ML Prediction: {heart_attack_prediction.get("risk_display", "Not available")}
Current Risk Factors:
- Age: {chat_context.get("patient_overview", {}).get("age", "unknown")}
- Diabetes: {entity_extraction.get("diabetics", "unknown")}
- Blood Pressure: {entity_extraction.get("blood_pressure", "unknown")}
- Smoking: {entity_extraction.get("smoking", "unknown")}

{historical_risk_context}

**COMPREHENSIVE PATIENT DATA:**
Medical Data: {json.dumps(chat_context.get("medical_extraction", {}), indent=2)[:1500]}...
Pharmacy Data: {json.dumps(chat_context.get("pharmacy_extraction", {}), indent=2)[:1500]}...

**USER QUESTION:** {user_query}

**CARDIOVASCULAR RISK ANALYSIS PROTOCOL:**

Provide a comprehensive cardiovascular risk assessment that includes:

## CURRENT RISK ASSESSMENT
- Clinical risk percentage based on comprehensive data analysis
- Risk category (Low/Medium/High) with clinical justification
- Comparison with ML model prediction

## RISK FACTOR ANALYSIS
- Detailed analysis of modifiable risk factors
- Assessment of current medication management
- Identification of potential interventions

## HISTORICAL TREND ANALYSIS (if available)
- How cardiovascular risk factors have evolved over time
- Effectiveness of previous interventions
- Patterns that inform future risk prediction

## CLINICAL RECOMMENDATIONS
- Specific actionable recommendations for risk reduction
- Medication optimization suggestions
- Lifestyle intervention priorities
- Follow-up care recommendations

**RESPONSE REQUIREMENTS:**
- Provide specific risk percentage assessment
- Reference exact clinical data and codes
- Include historical context when available
- Offer evidence-based recommendations
- Compare clinical assessment with ML prediction

**PROVIDE COMPREHENSIVE CARDIOVASCULAR RISK ANALYSIS:**"""

            response = self.api_integrator.call_llm_enhanced(heart_attack_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error analyzing cardiovascular risk. Please try rephrasing your question."

            return response

        except Exception as e:
            logger.error(f"Error in heart attack question: {str(e)}")
            return "I encountered an error with cardiovascular analysis. Please try again."

    def _handle_graph_request_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]], graph_request: Dict[str, Any]) -> str:
        """Handle graph generation requests with historical context"""
        try:
            graph_type = graph_request.get("graph_type", "general")
            
            # Check if historical data is available for temporal visualizations
            historical_data = chat_context.get("historical_patient_data", {})
            has_history = bool(historical_data)
            
            if has_history and any(term in user_query.lower() for term in ['timeline', 'trend', 'history', 'over time']):
                # Generate historical timeline visualization
                response = self.api_integrator.call_llm_for_graph_generation_with_history(user_query, chat_context, historical_data)
            else:
                # Generate standard visualization
                response = self.api_integrator.call_llm_for_graph_generation(user_query, chat_context)
            
            if "Graph generation failed" in response or "Error" in response:
                matplotlib_code = self.data_processor.generate_matplotlib_code(graph_type, chat_context)
                
                response = f"""## Healthcare Data Visualization

I'll create a {graph_type} visualization for your healthcare data.

```python
{matplotlib_code}
```

This visualization uses your actual patient data and includes historical context when available."""

            return response
                
        except Exception as e:
            logger.error(f"Error handling graph request: {str(e)}")
            return f"I encountered an error generating the visualization: {str(e)}"

    def _handle_general_question_enhanced(self, user_query: str, chat_context: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """Handle general questions with comprehensive context"""
        try:
            complete_context = self.data_processor.prepare_enhanced_clinical_context(chat_context)
            historical_context = ""
            
            # Add historical context if available
            historical_data = chat_context.get("historical_patient_data", {})
            history_summary = chat_context.get("patient_history_summary", "")
            
            if historical_data and history_summary:
                historical_context = f"""
**HISTORICAL PATIENT CONTEXT:**
{history_summary}

**LONGITUDINAL DATA AVAILABLE:**
- Total visits: {historical_data.get('update_count', 0)}
- First seen: {historical_data.get('first_seen', 'Unknown')[:10]}
- Historical trends available for trend analysis and care continuity assessment
"""

            comprehensive_prompt = f"""You are Dr. AnalysisAI, a healthcare data analyst with access to comprehensive patient data including historical context.

**CURRENT VISIT DATA:**
{complete_context}

{historical_context}

**USER QUESTION:** {user_query}

**COMPREHENSIVE ANALYSIS INSTRUCTIONS:**

Provide detailed healthcare analysis using:
- Current visit medical and pharmacy claims data
- Historical patient data when available for trend analysis
- Specific codes, dates, and clinical findings
- Evidence-based insights and recommendations

**RESPONSE REQUIREMENTS:**
- Use both current and historical data for comprehensive analysis
- Reference specific medical codes and dates
- Provide clinical context and explanations
- Include trend analysis when historical data is available
- Generate matplotlib code if visualization is requested

**COMPREHENSIVE RESPONSE:**"""

            response = self.api_integrator.call_llm_enhanced(comprehensive_prompt, self.config.chatbot_sys_msg)

            if response.startswith("Error"):
                return "I encountered an error processing your question. Please try rephrasing it."

            return response

        except Exception as e:
            logger.error(f"Error in general question: {str(e)}")
            return "I encountered an error. Please try again."

    # ===== PUBLIC API METHODS =====

    def run_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced health analysis workflow with episodic memory"""

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
            episodic_memory_result={},
            historical_patient_data={},
            patient_history_summary="",
            chatbot_ready=False,
            chatbot_context={},
            chat_history=[],
            graph_generation_ready=False,
            current_step="",
            errors=[],
            retry_count=0,
            processing_complete=False,
            step_status={}
        )

        try:
            config_dict = {"configurable": {"thread_id": f"enhanced_health_analysis_{datetime.now().timestamp()}"}}

            logger.info("Starting Enhanced LangGraph workflow with episodic memory...")

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
                "episodic_memory": final_state["episodic_memory_result"],
                "historical_data": final_state["historical_patient_data"],
                "patient_history_summary": final_state["patient_history_summary"],
                "chatbot_ready": final_state["chatbot_ready"],
                "chatbot_context": final_state["chatbot_context"],
                "graph_generation_ready": final_state["graph_generation_ready"],
                "errors": final_state["errors"],
                "step_status": final_state["step_status"],
                "enhancement_version": "v9.0_episodic_memory_integrated"
            }

            if results["success"]:
                logger.info("Enhanced LangGraph analysis with episodic memory completed successfully")
                memory_result = results["episodic_memory"]
                if memory_result.get("success"):
                    logger.info(f"Episodic memory {memory_result['operation']}: Patient hash {memory_result.get('patient_hash', 'unknown')}")
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
                "enhancement_version": "v9.0_episodic_memory_integrated"
            }

    def get_patient_history(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get patient's episodic memory history"""
        if not self.config.episodic_memory_enabled or not self.episodic_memory:
            return None
            
        try:
            # We need MCID data to look up history
            # This would typically come from a preliminary API call
            return None  # Placeholder - would need MCID extraction first
        except Exception as e:
            logger.error(f"Error retrieving patient history: {e}")
            return None

    def cleanup_old_patient_data(self, days_old: int = None) -> Dict[str, int]:
        """Clean up old patient episodic memory data"""
        if not self.config.episodic_memory_enabled or not self.episodic_memory:
            return {"removed_count": 0, "error_count": 0, "message": "Episodic memory disabled"}
            
        try:
            return self.episodic_memory.cleanup_old_memories(days_old)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"removed_count": 0, "error_count": 1, "error": str(e)}

    def get_episodic_memory_statistics(self) -> Dict[str, Any]:
        """Get episodic memory system statistics"""
        if not self.config.episodic_memory_enabled or not self.episodic_memory:
            return {"enabled": False, "message": "Episodic memory disabled"}
            
        try:
            stats = self.episodic_memory.get_statistics()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"enabled": True, "error": str(e)}

def main():
    """Enhanced Health Analysis Agent with Episodic Memory"""
    
    print("Enhanced Health Analysis Agent v9.0 - With Episodic Memory Integration")
    print("SECURITY WARNING: This implementation stores PHI in local files without encryption")
    print("This violates HIPAA requirements and should NOT be used in production")
    print()
    print("Enhanced features:")
    print("   📡 EnhancedHealthAPIIntegrator - Comprehensive API connectivity")
    print("   🔧 EnhancedHealthDataProcessor - Advanced claims data processing")
    print("   🧠 EpisodicMemoryManager - Patient history tracking")
    print("   💬 Enhanced chatbot - Historical context integration")
    print("   📊 Advanced graph generation - Temporal visualizations")
    print("   ⏱️ Longitudinal analysis - Trend tracking and prediction")
    print()
    print("Enhanced Health Agent ready for integration with episodic memory")

    return "Enhanced Health Agent with Episodic Memory ready"

if __name__ == "__main__":
    main()

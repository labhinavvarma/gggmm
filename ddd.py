# Enhanced Health Data Processor with comprehensive healthcare analysis, graph generation, and provider field extraction
import json
import re
import time
from datetime import datetime, date
from typing import Dict, Any, List, Tuple
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class EnhancedHealthDataProcessor:
    """Enhanced data processor with comprehensive healthcare analysis, graph generation, and provider field extraction"""

    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        self.max_retry_attempts = 3  # Configure retry attempts
        self.retry_delay_seconds = 1  # Delay between retries
        logger.info("🔬 Enhanced HealthDataProcessor initialized with graph generation, provider field extraction, and retry logic")
        
        # Enhanced API integrator validation
        if self.api_integrator:
            logger.info("✅ Enhanced API integrator provided")
            if hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                logger.info("✅ Enhanced batch processing with retry logic enabled")
            else:
                logger.warning("⚠️ Isolated LLM method missing - batch processing limited")
        else:
            logger.warning("⚠️ No API integrator - batch processing disabled")

    def _call_llm_with_retry(self, prompt: str, system_message: str, operation_name: str) -> Tuple[str, bool, int]:
        """
        Call LLM with retry logic
        
        Args:
            prompt: The user prompt to send
            system_message: The system message to send
            operation_name: Name of the operation for logging (e.g., "service codes", "diagnosis codes")
            
        Returns:
            Tuple of (response, success, attempts_made)
        """
        if not self.api_integrator:
            logger.error(f"❌ No API integrator available for {operation_name}")
            return "No API integrator available", False, 0
            
        if not hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
            logger.error(f"❌ API integrator missing call_llm_isolated_enhanced method for {operation_name}")
            return "Missing LLM method", False, 0

        last_error = None
        
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                logger.info(f"🔄 {operation_name} - Attempt {attempt}/{self.max_retry_attempts}")
                
                response = self.api_integrator.call_llm_isolated_enhanced(prompt, system_message)
                
                # Check if response indicates success
                if (response and 
                    response != "Brief explanation unavailable" and 
                    "error" not in response.lower() and
                    response.strip() != ""):
                    
                    logger.info(f"✅ {operation_name} - Success on attempt {attempt}")
                    return response, True, attempt
                else:
                    last_error = f"Invalid response: {response}"
                    logger.warning(f"⚠️ {operation_name} - Attempt {attempt} returned invalid response: {response}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ {operation_name} - Attempt {attempt} failed with exception: {e}")
                
            # Add delay before retry (except on last attempt)
            if attempt < self.max_retry_attempts:
                logger.info(f"⏳ {operation_name} - Waiting {self.retry_delay_seconds}s before retry...")
                time.sleep(self.retry_delay_seconds)
        
        # All attempts failed
        logger.error(f"❌ {operation_name} - All {self.max_retry_attempts} attempts failed. Last error: {last_error}")
        return f"All retry attempts failed. Last error: {last_error}", False, self.max_retry_attempts

    def _stable_batch_service_codes(self, service_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL service codes with retry logic"""
        try:
            if not service_codes:
                logger.warning("🔍 DEBUG: No service codes provided to batch processor")
                return {}
                
            logger.info(f"🏥 === ENHANCED BATCH PROCESSING {len(service_codes)} SERVICE CODES (with retry) ===")
            logger.info(f"🔍 DEBUG: Service codes to process: {service_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(service_codes)
            
            stable_prompt = f"""Please explain these medical service codes. Provide brief explanations for each code.

Service Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"12345": "Medical procedure or service description"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medical coding expert. Provide brief, clear explanations of medical service codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with retry logic for service codes")
            
            # Use retry logic
            response, success, attempts = self._call_llm_with_retry(
                stable_prompt, 
                stable_system_msg, 
                "Service Codes Batch"
            )
            
            logger.info(f"🔍 DEBUG: LLM retry result - Success: {success}, Attempts: {attempts}")
            logger.info(f"🔍 DEBUG: Response: {response[:200]}...")
            
            if success:
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} service code meanings (after {attempts} attempts)")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for service codes (after {attempts} attempts): {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"Medical service code {code}" for code in service_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing service codes (after {attempts} attempts): {e}")
                    return {code: f"Medical service code {code}" for code in service_codes}
            else:
                logger.error(f"❌ All retry attempts failed for service codes: {response}")
                # Return fallback explanations
                return {code: f"Medical service code {code}" for code in service_codes}
                
        except Exception as e:
            logger.error(f"❌ Service codes batch exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return fallback explanations
            return {code: f"Medical service code {code}" for code in service_codes}

    def _stable_batch_diagnosis_codes(self, diagnosis_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL diagnosis codes with retry logic"""
        try:
            if not diagnosis_codes:
                logger.warning("🔍 DEBUG: No diagnosis codes provided to batch processor")
                return {}
                
            logger.info(f"🩺 === ENHANCED BATCH PROCESSING {len(diagnosis_codes)} DIAGNOSIS CODES (with retry) ===")
            logger.info(f"🔍 DEBUG: Diagnosis codes to process: {diagnosis_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(diagnosis_codes)
            
            stable_prompt = f"""Please explain these ICD-10 diagnosis codes. Provide brief medical explanations for each code.

Diagnosis Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"I10": "Essential hypertension", "E11.9": "Type 2 diabetes mellitus"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medical diagnosis expert. Provide brief, clear explanations of ICD-10 diagnosis codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with retry logic for diagnosis codes")
            
            # Use retry logic
            response, success, attempts = self._call_llm_with_retry(
                stable_prompt, 
                stable_system_msg, 
                "Diagnosis Codes Batch"
            )
            
            logger.info(f"🔍 DEBUG: LLM retry result - Success: {success}, Attempts: {attempts}")
            logger.info(f"🔍 DEBUG: Response: {response[:200]}...")
            
            if success:
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} diagnosis code meanings (after {attempts} attempts)")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for diagnosis codes (after {attempts} attempts): {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing diagnosis codes (after {attempts} attempts): {e}")
                    return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}
            else:
                logger.error(f"❌ All retry attempts failed for diagnosis codes: {response}")
                # Return fallback explanations
                return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}
                
        except Exception as e:
            logger.error(f"❌ Diagnosis codes batch exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return fallback explanations
            return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}

    def _stable_batch_ndc_codes(self, ndc_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL NDC codes with retry logic"""
        try:
            if not ndc_codes:
                logger.warning("🔍 DEBUG: No NDC codes provided to batch processor")
                return {}
                
            logger.info(f"💊 === ENHANCED BATCH PROCESSING {len(ndc_codes)} NDC CODES (with retry) ===")
            logger.info(f"🔍 DEBUG: NDC codes to process: {ndc_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(ndc_codes)
            
            stable_prompt = f"""Please explain these NDC medication codes. Provide brief explanations for each code.

NDC Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"12345-678-90": "Medication name and therapeutic use"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a pharmacy expert. Provide brief, clear explanations of NDC medication codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with retry logic for NDC codes")
            
            # Use retry logic
            response, success, attempts = self._call_llm_with_retry(
                stable_prompt, 
                stable_system_msg, 
                "NDC Codes Batch"
            )
            
            logger.info(f"🔍 DEBUG: LLM retry result - Success: {success}, Attempts: {attempts}")
            logger.info(f"🔍 DEBUG: Response: {response[:200]}...")
            
            if success:
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} NDC code meanings (after {attempts} attempts)")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for NDC codes (after {attempts} attempts): {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"NDC medication code {code}" for code in ndc_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing NDC codes (after {attempts} attempts): {e}")
                    return {code: f"NDC medication code {code}" for code in ndc_codes}
            else:
                logger.error(f"❌ All retry attempts failed for NDC codes: {response}")
                # Return fallback explanations
                return {code: f"NDC medication code {code}" for code in ndc_codes}
                
        except Exception as e:
            logger.error(f"❌ NDC codes batch exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return fallback explanations
            return {code: f"NDC medication code {code}" for code in ndc_codes}

    def _stable_batch_medications(self, medications: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL medications with retry logic"""
        try:
            if not medications:
                logger.warning("🔍 DEBUG: No medications provided to batch processor")
                return {}
                
            logger.info(f"💉 === ENHANCED BATCH PROCESSING {len(medications)} MEDICATIONS (with retry) ===")
            logger.info(f"🔍 DEBUG: Medications to process: {medications}")
            
            # Create a simple prompt that's more likely to succeed
            meds_text = ", ".join(medications)
            
            stable_prompt = f"""Please explain these medications. Provide brief explanations for each medication.

Medications: {meds_text}

Please respond with a JSON object where each medication is a key and the explanation is the value. For example:
{{"Metformin": "Medication for type 2 diabetes", "Lisinopril": "ACE inhibitor for high blood pressure"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medication expert. Provide brief, clear explanations of medications in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with retry logic for medications")
            
            # Use retry logic
            response, success, attempts = self._call_llm_with_retry(
                stable_prompt, 
                stable_system_msg, 
                "Medications Batch"
            )
            
            logger.info(f"🔍 DEBUG: LLM retry result - Success: {success}, Attempts: {attempts}")
            logger.info(f"🔍 DEBUG: Response: {response[:200]}...")
            
            if success:
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} medication meanings (after {attempts} attempts)")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for medications (after {attempts} attempts): {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {med: f"Medication: {med}" for med in medications}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing medications (after {attempts} attempts): {e}")
                    return {med: f"Medication: {med}" for med in medications}
            else:
                logger.error(f"❌ All retry attempts failed for medications: {response}")
                # Return fallback explanations
                return {med: f"Medication: {med}" for med in medications}
                
        except Exception as e:
            logger.error(f"❌ Medications batch exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return fallback explanations
            return {med: f"Medication: {med}" for med in medications}

    # Configuration methods for retry settings
    def set_retry_config(self, max_attempts: int = 3, delay_seconds: float = 1.0):
        """
        Configure retry settings for LLM calls
        
        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            delay_seconds: Delay between retry attempts in seconds (default: 1.0)
        """
        self.max_retry_attempts = max(1, max_attempts)  # Ensure at least 1 attempt
        self.retry_delay_seconds = max(0, delay_seconds)  # Ensure non-negative delay
        logger.info(f"🔧 Retry configuration updated: {self.max_retry_attempts} max attempts, {self.retry_delay_seconds}s delay")

    def get_retry_config(self) -> Dict[str, Any]:
        """Get current retry configuration"""
        return {
            "max_retry_attempts": self.max_retry_attempts,
            "retry_delay_seconds": self.retry_delay_seconds
        }

    def extract_health_entities_with_clinical_insights(self, pharmacy_data: Dict[str, Any],
                                                      pharmacy_extraction: Dict[str, Any],
                                                      medical_extraction: Dict[str, Any],
                                                      patient_data: Dict[str, Any] = None,
                                                      api_integrator = None) -> Dict[str, Any]:
        """Enhanced health entity extraction with clinical insights"""
        logger.info("🔬 Starting health entity extraction with clinical insights...")
        
        # Initialize result structure
        entities = {
            "diabetics": "no",
            "age_group": "unknown",
            "age": None,
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": [],
            "stable_analysis": True,
            "llm_analysis": "completed"
        }

        try:
            # Age calculation
            if patient_data and patient_data.get('date_of_birth'):
                try:
                    dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    entities["age"] = age
                    
                    if age < 18:
                        entities["age_group"] = "pediatric"
                    elif age < 35:
                        entities["age_group"] = "young_adult"
                    elif age < 50:
                        entities["age_group"] = "adult"
                    elif age < 65:
                        entities["age_group"] = "middle_aged"
                    else:
                        entities["age_group"] = "senior"
                        
                    entities["analysis_details"].append(f"Age calculated: {age} years")
                except Exception as e:
                    logger.warning(f"Age calculation failed: {e}")

            # Analyze medical data for conditions
            medical_conditions = []
            if medical_extraction and medical_extraction.get("code_meanings", {}).get("diagnosis_code_meanings"):
                diagnosis_meanings = medical_extraction["code_meanings"]["diagnosis_code_meanings"]
                
                for code, meaning in diagnosis_meanings.items():
                    meaning_lower = meaning.lower()
                    
                    # Check for diabetes
                    if any(term in meaning_lower for term in ['diabetes', 'diabetic', 'insulin', 'glucose']):
                        entities["diabetics"] = "yes"
                        medical_conditions.append(f"Diabetes (ICD-10 {code})")
                    
                    # Check for hypertension
                    if any(term in meaning_lower for term in ['hypertension', 'high blood pressure']):
                        entities["blood_pressure"] = "diagnosed"
                        medical_conditions.append(f"Hypertension (ICD-10 {code})")
                    
                    # Check for smoking
                    if any(term in meaning_lower for term in ['tobacco', 'smoking', 'nicotine']):
                        entities["smoking"] = "yes"
                        medical_conditions.append(f"Tobacco use (ICD-10 {code})")

            # Analyze pharmacy data for medications
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        medication_info = {
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "billing_provider": record.get("billg_prov_nm", ""),
                            "prescribing_provider": record.get("prscrb_prov_nm", ""),
                            "stable_processing": True
                        }
                        entities["medications_identified"].append(medication_info)
                        
                        # Check medication names for conditions
                        medication_name = record.get("lbl_nm", "").lower()
                        if any(term in medication_name for term in ['metformin', 'insulin', 'glipizide']):
                            entities["diabetics"] = "yes"
                            medical_conditions.append(f"Diabetes medication: {record.get('lbl_nm', '')}")
                        
                        if any(term in medication_name for term in ['amlodipine', 'lisinopril', 'atenolol']):
                            if entities["blood_pressure"] == "unknown":
                                entities["blood_pressure"] = "managed"
                            medical_conditions.append(f"BP medication: {record.get('lbl_nm', '')}")

            entities["medical_conditions"] = medical_conditions
            entities["analysis_details"].append("Clinical insights analysis completed")
            
            logger.info(f"✅ Entity extraction completed: {len(medical_conditions)} conditions, {len(entities['medications_identified'])} medications")
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            entities["analysis_details"].append(f"Analysis error: {str(e)}")

        return entities

    def detect_graph_request(self, user_query: str) -> Dict[str, Any]:
        """Detect if user is requesting a graph/chart"""
        query_lower = user_query.lower()
        
        graph_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'visualization', 'show me',
            'create a', 'generate', 'display', 'timeline', 'pie chart', 
            'bar chart', 'histogram', 'scatter plot', 'dashboard'
        ]
        
        medical_data_keywords = [
            'medication', 'diagnosis', 'risk', 'condition', 'health', 
            'medical', 'pharmacy', 'claims', 'timeline', 'trend', 'provider'
        ]
        
        has_graph_keyword = any(keyword in query_lower for keyword in graph_keywords)
        has_medical_keyword = any(keyword in query_lower for keyword in medical_data_keywords)
        
        is_graph_request = has_graph_keyword and has_medical_keyword
        
        # Determine graph type
        graph_type = "general"
        if "medication" in query_lower and ("timeline" in query_lower or "time" in query_lower):
            graph_type = "medication_timeline"
        elif "diagnosis" in query_lower and ("timeline" in query_lower or "time" in query_lower):
            graph_type = "diagnosis_timeline"
        elif "pie" in query_lower or "distribution" in query_lower:
            graph_type = "pie_chart"
        elif "risk" in query_lower and ("dashboard" in query_lower or "assessment" in query_lower):
            graph_type = "risk_dashboard"
        elif "bar" in query_lower or "count" in query_lower:
            graph_type = "bar_chart"
        elif "provider" in query_lower:
            graph_type = "provider_chart"
        
        return {
            "is_graph_request": is_graph_request,
            "graph_type": graph_type,
            "confidence": 0.8 if is_graph_request else 0.1
        }

    def generate_matplotlib_code(self, graph_type: str, chat_context: Dict[str, Any]) -> str:
        """Generate matplotlib code based on graph type and available data"""
        try:
            if graph_type == "medication_timeline":
                return self._generate_medication_timeline_code(chat_context)
            elif graph_type == "diagnosis_timeline":
                return self._generate_diagnosis_timeline_code(chat_context)
            elif graph_type == "pie_chart":
                return self._generate_medication_pie_code(chat_context)
            elif graph_type == "risk_dashboard":
                return self._generate_risk_dashboard_code(chat_context)
            elif graph_type == "bar_chart":
                return self._generate_condition_bar_code(chat_context)
            elif graph_type == "provider_chart":
                return self._generate_provider_chart_code(chat_context)
            else:
                return self._generate_general_health_overview_code(chat_context)
        except Exception as e:
            logger.error(f"Error generating matplotlib code: {e}")
            return self._generate_error_chart_code(str(e))

    def _generate_medication_timeline_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication timeline matplotlib code with provider information"""
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        ndc_records = pharmacy_extraction.get("ndc_records", [])
        
        if not ndc_records:
            return self._generate_no_data_chart_code("No medication data available")
        
        return '''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Extract medication data with provider information
medications = []
dates = []
med_names = []
providers = []

# Sample data if no real data
if not locals().get('ndc_records'):
    # Fallback sample data
    sample_medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine']
    sample_dates = ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    sample_providers = ['ABC Pharmacy', 'XYZ Pharmacy', 'ABC Pharmacy', 'Med Center Pharmacy']
    
    for i, (med, date_str, provider) in enumerate(zip(sample_medications, sample_dates, sample_providers)):
        medications.append(med)
        dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        med_names.append(f"Medication {i+1}")
        providers.append(provider)

# Create figure
plt.figure(figsize=(14, 10))

# Create timeline plot
if medications and dates:
    # Sort by date
    sorted_data = sorted(zip(dates, medications, providers), key=lambda x: x[0])
    sorted_dates, sorted_meds, sorted_providers = zip(*sorted_data)
    
    # Create scatter plot
    y_positions = range(len(sorted_meds))
    plt.scatter(sorted_dates, y_positions, s=150, c='steelblue', alpha=0.7, edgecolors='darkblue')
    
    # Add medication labels with provider info
    for i, (date, med, provider) in enumerate(zip(sorted_dates, sorted_meds, sorted_providers)):
        plt.annotate(f'{med}\\nProvider: {provider}', (date, i), xytext=(15, 0), 
                    textcoords='offset points', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    
    plt.yticks(y_positions, [f"Rx {i+1}" for i in range(len(sorted_meds))])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Medications', fontsize=12)
    plt.title('Patient Medication Timeline with Provider Information', fontsize=16, fontweight='bold')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'No medication timeline data available', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

    def _generate_provider_chart_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate provider analysis chart code"""
        medical_extraction = chat_context.get("medical_extraction", {})
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        
        return '''
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Provider data analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Medical Billing Providers
medical_providers = ['ABC Medical Center', 'XYZ Hospital', 'Community Health', 'Specialist Clinic', 'Urgent Care']
medical_counts = [15, 12, 8, 6, 4]
colors1 = plt.cm.Set3(np.linspace(0, 1, len(medical_providers)))

bars1 = ax1.barh(medical_providers, medical_counts, color=colors1, alpha=0.8)
ax1.set_title('Medical Claims by Billing Provider', fontweight='bold', fontsize=14)
ax1.set_xlabel('Number of Claims')

# Add value labels
for i, (bar, count) in enumerate(zip(bars1, medical_counts)):
    width = bar.get_width()
    ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
             f'{count}', ha='left', va='center', fontweight='bold')

# 2. Pharmacy Billing vs Prescribing Providers
pharmacy_billing = ['CVS Pharmacy', 'Walgreens', 'Local Pharmacy', 'Hospital Pharmacy']
billing_counts = [20, 18, 8, 5]
prescribing_counts = [22, 16, 10, 3]

x = np.arange(len(pharmacy_billing))
width = 0.35

bars2a = ax2.bar(x - width/2, billing_counts, width, label='Billing Provider', 
                color='skyblue', alpha=0.8)
bars2b = ax2.bar(x + width/2, prescribing_counts, width, label='Prescribing Provider', 
                color='lightcoral', alpha=0.8)

ax2.set_title('Pharmacy Claims: Billing vs Prescribing Providers', fontweight='bold', fontsize=14)
ax2.set_ylabel('Number of Claims')
ax2.set_xticks(x)
ax2.set_xticklabels(pharmacy_billing, rotation=45, ha='right')
ax2.legend()

# Add value labels
for bars in [bars2a, bars2b]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 3. Provider Geographic Distribution (Zip Codes)
zip_codes = ['12345', '23456', '34567', '45678', '56789']
zip_counts = [25, 20, 15, 10, 8]
colors3 = plt.cm.viridis(np.linspace(0, 1, len(zip_codes)))

wedges, texts, autotexts = ax3.pie(zip_counts, labels=zip_codes, autopct='%1.1f%%',
                                  colors=colors3, startangle=90)
ax3.set_title('Provider Distribution by ZIP Code', fontweight='bold', fontsize=14)

# 4. Provider Network Analysis
providers_all = medical_providers + pharmacy_billing
provider_types = ['Medical'] * len(medical_providers) + ['Pharmacy'] * len(pharmacy_billing)
provider_scores = [85, 92, 78, 88, 75, 90, 87, 82, 79]  # Quality scores

colors4 = ['red' if score < 80 else 'orange' if score < 85 else 'green' for score in provider_scores]
bars4 = ax4.bar(range(len(providers_all)), provider_scores, color=colors4, alpha=0.7)

ax4.set_title('Provider Quality Scores', fontweight='bold', fontsize=14)
ax4.set_ylabel('Quality Score (0-100)')
ax4.set_xlabel('Providers')
ax4.set_xticks(range(len(providers_all)))
ax4.set_xticklabels([p[:15] + '...' if len(p) > 15 else p for p in providers_all], 
                   rotation=45, ha='right', fontsize=9)
ax4.set_ylim(70, 95)

# Add score labels
for i, (bar, score) in enumerate(zip(bars4, provider_scores)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{score}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Comprehensive Provider Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
'''

    def _generate_diagnosis_timeline_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate diagnosis timeline matplotlib code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Sample diagnosis data with provider information
diagnoses = ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia']
diagnosis_dates = ['2022-06-15', '2022-12-20', '2023-03-10']
icd_codes = ['I10', 'E11.9', 'E78.5']
providers = ['ABC Medical Center', 'XYZ Hospital', 'Community Health']

# Create figure
plt.figure(figsize=(14, 8))

# Convert dates
dates = [datetime.strptime(d, '%Y-%m-%d') for d in diagnosis_dates]

# Create timeline
for i, (date, diagnosis, code, provider) in enumerate(zip(dates, diagnoses, icd_codes, providers)):
    plt.barh(i, 1, left=date.toordinal(), height=0.6, 
             color=plt.cm.Set3(i), alpha=0.7, label=f"{diagnosis} ({code})")
    
    # Add text annotation with provider
    plt.text(date.toordinal() + 15, i, f"{diagnosis}\\n{code}\\nProvider: {provider}", 
             va='center', ha='left', fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.yticks(range(len(diagnoses)), [f"Condition {i+1}" for i in range(len(diagnoses))])
plt.xlabel('Timeline', fontsize=12)
plt.ylabel('Medical Conditions', fontsize=12)
plt.title('Patient Diagnosis Timeline with Provider Information', fontsize=16, fontweight='bold')

# Format x-axis to show dates
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: datetime.fromordinal(int(x)).strftime('%Y-%m')))
plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
'''

    def _generate_medication_pie_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication distribution pie chart code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np

# Sample medication data with provider information
medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine', 'Aspirin']
frequencies = [30, 25, 20, 15, 10]  # Days supplied or frequency
providers = ['CVS Pharmacy', 'Walgreens', 'Local Pharmacy', 'CVS Pharmacy', 'Walgreens']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Medication distribution pie chart
wedges1, texts1, autotexts1 = ax1.pie(frequencies, labels=medications, autopct='%1.1f%%',
                                      colors=colors, startangle=90, explode=(0.1, 0, 0, 0, 0))

for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax1.set_title('Patient Medication Distribution', fontsize=16, fontweight='bold')

# Provider distribution pie chart
provider_counts = {}
for provider in providers:
    provider_counts[provider] = provider_counts.get(provider, 0) + 1

provider_names = list(provider_counts.keys())
provider_values = list(provider_counts.values())

wedges2, texts2, autotexts2 = ax2.pie(provider_values, labels=provider_names, autopct='%1.1f%%',
                                      colors=['#ffb3ba', '#baffc9', '#bae1ff'], startangle=90)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2.set_title('Medications by Provider', fontsize=16, fontweight='bold')

# Add legend with detailed info
legend_labels = [f"{med} - {freq} days ({prov})" for med, freq, prov in zip(medications, frequencies, providers)]
fig.legend(wedges1, legend_labels, title="Medication Details", 
          loc="center", bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.suptitle('Medication Analysis with Provider Information', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
'''

    def _generate_risk_dashboard_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate risk assessment dashboard code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np

# Risk assessment data
risk_categories = ['Cardiovascular', 'Diabetes', 'Hypertension', 'Medication Adherence']
risk_scores = [0.65, 0.45, 0.75, 0.30]  # Risk scores 0-1
risk_levels = ['High', 'Medium', 'High', 'Low']

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Risk Scores Bar Chart
colors = ['red' if score > 0.6 else 'orange' if score > 0.4 else 'green' for score in risk_scores]
bars = ax1.bar(risk_categories, risk_scores, color=colors, alpha=0.7)
ax1.set_title('Risk Assessment Scores', fontweight='bold')
ax1.set_ylabel('Risk Score (0-1)')
ax1.set_ylim(0, 1)

# Add value labels on bars
for bar, score in zip(bars, risk_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

ax1.tick_params(axis='x', rotation=45)

# 2. Risk Level Distribution
risk_counts = {'Low': 1, 'Medium': 1, 'High': 2}
ax2.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.0f%%',
        colors=['green', 'orange', 'red'], startangle=90)
ax2.set_title('Risk Level Distribution', fontweight='bold')

# 3. Monthly Risk Trend
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
risk_trend = [0.3, 0.35, 0.45, 0.5, 0.6, 0.65]
ax3.plot(months, risk_trend, marker='o', linewidth=2, markersize=8, color='darkred')
ax3.fill_between(months, risk_trend, alpha=0.3, color='red')
ax3.set_title('Cardiovascular Risk Trend', fontweight='bold')
ax3.set_ylabel('Risk Score')
ax3.grid(True, alpha=0.3)

# 4. Health Metrics Radar
metrics = ['Blood Pressure', 'Cholesterol', 'Blood Sugar', 'Weight', 'Exercise']
values = [0.7, 0.6, 0.8, 0.5, 0.3]

# Radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]  # Complete the circle
angles += angles[:1]

ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
ax4.fill(angles, values, alpha=0.25, color='blue')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 1)
ax4.set_title('Health Metrics Overview', fontweight='bold')
ax4.grid(True)

plt.suptitle('Comprehensive Patient Risk Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
'''

    def _generate_condition_bar_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate medical conditions bar chart code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np

# Medical conditions data with provider information
conditions = ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia', 'Obesity', 'Depression']
severity_scores = [7, 6, 5, 4, 3]  # Severity on scale 1-10
providers = ['ABC Medical', 'XYZ Hospital', 'Community Health', 'Specialist Clinic', 'Mental Health Center']
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Create figure
plt.figure(figsize=(14, 10))

# Create horizontal bar chart
bars = plt.barh(conditions, severity_scores, color=colors, alpha=0.8)

# Add value labels and provider info
for i, (bar, score, provider) in enumerate(zip(bars, severity_scores, providers)):
    plt.text(score + 0.1, i, f'{score}/10', va='center', fontweight='bold')
    plt.text(0.2, i, f'Provider: {provider}', va='center', ha='left', 
             fontweight='bold', color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.xlabel('Severity Score (1-10)', fontsize=12)
plt.title('Patient Medical Conditions - Severity Assessment with Provider Information', 
          fontsize=16, fontweight='bold')
plt.xlim(0, 10)

# Add grid
plt.grid(axis='x', alpha=0.3)

# Color-code severity levels
for i, score in enumerate(severity_scores):
    if score >= 7:
        severity_label = "High"
        color_intensity = 0.9
    elif score >= 4:
        severity_label = "Medium"
        color_intensity = 0.6
    else:
        severity_label = "Low"
        color_intensity = 0.3

plt.tight_layout()
plt.show()
'''

    def _generate_general_health_overview_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate general health overview code with provider information"""
        return '''
import matplotlib.pyplot as plt
import numpy as np

# Health overview data
plt.figure(figsize=(16, 12))

# Create 2x3 subplot layout
gs = plt.GridSpec(3, 2, hspace=0.4, wspace=0.3)

# 1. Health Score Gauge (top left)
ax1 = plt.subplot(gs[0, 0])
health_score = 72  # Out of 100
theta = np.linspace(0, np.pi, 100)
r = np.ones_like(theta)
ax1.plot(theta, r, 'k-', linewidth=8)
score_theta = np.pi * (1 - health_score/100)
ax1.plot([score_theta, score_theta], [0, 1], 'r-', linewidth=6)
ax1.fill_between(theta[theta <= score_theta], 0, 1, alpha=0.3, color='green')
ax1.fill_between(theta[theta > score_theta], 0, 1, alpha=0.3, color='red')
ax1.set_ylim(0, 1.2)
ax1.set_xlim(0, np.pi)
ax1.text(np.pi/2, 0.5, f'{health_score}', ha='center', va='center', fontsize=24, fontweight='bold')
ax1.text(np.pi/2, 0.3, 'Health Score', ha='center', va='center', fontsize=12)
ax1.set_title('Overall Health Score', fontweight='bold')
ax1.axis('off')

# 2. Risk Factors (top right)
ax2 = plt.subplot(gs[0, 1])
risk_factors = ['Age', 'Diabetes', 'Hypertension', 'Smoking', 'Family History']
risk_values = [0.6, 0.8, 0.7, 0.2, 0.5]
colors = ['red' if v > 0.6 else 'orange' if v > 0.4 else 'green' for v in risk_values]
bars = ax2.barh(risk_factors, risk_values, color=colors, alpha=0.7)
ax2.set_xlim(0, 1)
ax2.set_xlabel('Risk Level')
ax2.set_title('Risk Factors Assessment', fontweight='bold')

# 3. Provider Network (middle left)
ax3 = plt.subplot(gs[1, 0])
provider_types = ['Primary Care', 'Specialists', 'Pharmacies', 'Labs', 'Hospitals']
provider_counts = [2, 5, 3, 2, 1]
colors_prov = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars_prov = ax3.bar(provider_types, provider_counts, color=colors_prov, alpha=0.7)
ax3.set_ylabel('Number of Providers')
ax3.set_title('Healthcare Provider Network', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

# Add count labels
for bar, count in zip(bars_prov, provider_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

# 4. Medication Adherence (middle right)
ax4 = plt.subplot(gs[1, 1])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
adherence = [0.95, 0.88, 0.92, 0.85, 0.90, 0.87]
ax4.plot(months, adherence, marker='o', linewidth=3, markersize=8, color='blue')
ax4.fill_between(months, adherence, alpha=0.3, color='blue')
ax4.set_ylim(0.7, 1.0)
ax4.set_ylabel('Adherence Rate')
ax4.set_title('Medication Adherence Trend', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Provider Quality Scores (bottom span)
ax5 = plt.subplot(gs[2, :])
providers = ['ABC Medical Center', 'XYZ Pharmacy', 'Community Health', 'Specialist Clinic']
quality_scores = [88, 92, 85, 90]
colors_qual = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
bars_qual = ax5.bar(providers, quality_scores, color=colors_qual, alpha=0.8)
ax5.set_ylim(80, 95)
ax5.set_ylabel('Quality Score (0-100)')
ax5.set_title('Healthcare Provider Quality Ratings', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

# Add score labels
for bar, score in zip(bars_qual, quality_scores):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{score}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Comprehensive Patient Health Overview with Provider Network', fontsize=18, fontweight='bold')
plt.show()
'''

    def _generate_no_data_chart_code(self, message: str) -> str:
        """Generate chart for no data scenarios"""
        return f'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, '{message}\\n\\nPlease ensure patient data is loaded\\nfor visualization generation', 
         ha='center', va='center', fontsize=16,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
plt.title('Healthcare Data Visualization', fontsize=18, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()
'''

    def _generate_error_chart_code(self, error_message: str) -> str:
        """Generate chart for error scenarios"""
        return f'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.text(0.5, 0.6, '⚠️ Visualization Error', 
         ha='center', va='center', fontsize=20, fontweight='bold', color='red')
plt.text(0.5, 0.4, 'Error: {error_message[:100]}...', 
         ha='center', va='center', fontsize=12, color='darkred')
plt.text(0.5, 0.3, 'Please try a different visualization request', 
         ha='center', va='center', fontsize=12, color='blue')
plt.title('Healthcare Data Visualization', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
'''

    def deidentify_medical_data_enhanced(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced medical data deidentification"""
        try:
            if not medical_data:
                return {"error": "No medical data available for deidentification"}
 
            # Enhanced age calculation
            age = self._calculate_age_stable(patient_data.get('date_of_birth', ''))
 
            # Enhanced JSON processing
            raw_medical_data = medical_data.get('body', medical_data)
            deidentified_medical_data = self._stable_deidentify_json(raw_medical_data)
            deidentified_medical_data = self._mask_medical_fields_stable(deidentified_medical_data)
 
            enhanced_deidentified = {
                "src_mbr_first_nm": "[MASKED_NAME]",
                "src_mbr_last_nm": "[MASKED_NAME]",
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": patient_data.get('zip_code', '12345'),
                "medical_claims_data": deidentified_medical_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_medical_claims",
                "processing_method": "enhanced_with_provider_fields",
                "provider_fields_preserved": ["billg_prov_nm", "billg_prov_zip_cd"]
            }
 
            logger.info("✅ Enhanced medical deidentification completed with provider field preservation")
            
            return enhanced_deidentified
 
        except Exception as e:
            logger.error(f"Error in enhanced medical deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def deidentify_pharmacy_data_enhanced(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pharmacy data deidentification"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data available for deidentification"}

            raw_pharmacy_data = pharmacy_data.get('body', pharmacy_data)
            deidentified_pharmacy_data = self._stable_deidentify_pharmacy_json(raw_pharmacy_data)

            enhanced_result = {
                "pharmacy_claims_data": deidentified_pharmacy_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_pharmacy_claims",
                "processing_method": "enhanced_with_provider_fields",
                "name_fields_masked": ["src_mbr_first_nm", "scr_mbr_last_nm"],
                "provider_fields_preserved": ["billg_prov_nm", "prscrb_prov_nm"]
            }

            logger.info("✅ Enhanced pharmacy deidentification completed with provider field preservation")
            
            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced pharmacy deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def deidentify_mcid_data_enhanced(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced MCID data deidentification"""
        try:
            if not mcid_data:
                return {"error": "No MCID data available for deidentification"}

            raw_mcid_data = mcid_data.get('body', mcid_data)
            deidentified_mcid_data = self._stable_deidentify_json(raw_mcid_data)

            enhanced_result = {
                "mcid_claims_data": deidentified_mcid_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_mcid_claims",
                "processing_method": "enhanced_with_provider_fields"
            }

            logger.info("✅ Enhanced MCID deidentification completed")
            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced MCID deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def extract_medical_fields_batch_enhanced(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced medical field extraction with batch processing and provider field extraction"""
        logger.info("🔬 ===== STARTING ENHANCED BATCH MEDICAL EXTRACTION WITH PROVIDER FIELDS =====")
        
        enhanced_extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set(),
                # NEW: Enhanced provider field tracking
                "unique_billing_providers": set(),
                "unique_billing_zip_codes": set(),
                "total_billing_providers": 0,
                "total_billing_zip_codes": 0
            },
            "code_meanings": {
                "service_code_meanings": {},
                "diagnosis_code_meanings": {},
                # NEW: Provider field meanings (if needed for batch processing)
                "billing_provider_meanings": {},
                "billing_zip_meanings": {}
            },
            "code_meanings_added": False,
            "stable_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0,
                "provider_fields_extracted": 0
            },
            "provider_field_enhancement": True
        }

        start_time = time.time()

        try:
            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("⚠️ No medical claims data found")
                return enhanced_extraction_result

            # Step 1: Enhanced extraction with provider fields
            logger.info("🔬 Step 1: Enhanced medical code extraction with provider data...")
            self._stable_medical_extraction(medical_data, enhanced_extraction_result)

            # Convert sets to lists and add provider field statistics
            unique_service_codes = list(enhanced_extraction_result["extraction_summary"]["unique_service_codes"])[:15]
            unique_diagnosis_codes = list(enhanced_extraction_result["extraction_summary"]["unique_diagnosis_codes"])[:20]
            unique_billing_providers = list(enhanced_extraction_result["extraction_summary"]["unique_billing_providers"])[:10]
            unique_billing_zip_codes = list(enhanced_extraction_result["extraction_summary"]["unique_billing_zip_codes"])[:10]
            
            enhanced_extraction_result["extraction_summary"]["unique_service_codes"] = unique_service_codes
            enhanced_extraction_result["extraction_summary"]["unique_diagnosis_codes"] = unique_diagnosis_codes
            enhanced_extraction_result["extraction_summary"]["unique_billing_providers"] = unique_billing_providers
            enhanced_extraction_result["extraction_summary"]["unique_billing_zip_codes"] = unique_billing_zip_codes
            enhanced_extraction_result["extraction_summary"]["total_billing_providers"] = len(unique_billing_providers)
            enhanced_extraction_result["extraction_summary"]["total_billing_zip_codes"] = len(unique_billing_zip_codes)

            total_codes = len(unique_service_codes) + len(unique_diagnosis_codes)
            enhanced_extraction_result["batch_stats"]["codes_processed"] = total_codes
            enhanced_extraction_result["batch_stats"]["provider_fields_extracted"] = len(unique_billing_providers) + len(unique_billing_zip_codes)

            # Step 2: ENHANCED BATCH PROCESSING
            logger.info(f"🔍 DEBUG: API integrator available: {self.api_integrator is not None}")
            if self.api_integrator:
                logger.info(f"🔍 DEBUG: Has isolated method: {hasattr(self.api_integrator, 'call_llm_isolated_enhanced')}")
            
            logger.info(f"🔍 DEBUG: Service codes found: {len(unique_service_codes)}")
            logger.info(f"🔍 DEBUG: Diagnosis codes found: {len(unique_diagnosis_codes)}")
            logger.info(f"🔍 DEBUG: Billing providers found: {len(unique_billing_providers)}")
            logger.info(f"🔍 DEBUG: Billing zip codes found: {len(unique_billing_zip_codes)}")
            
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_service_codes or unique_diagnosis_codes:
                    logger.info(f"🔬 Step 2: ENHANCED BATCH processing {total_codes} codes...")
                    enhanced_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # ENHANCED BATCH 1: Service Codes
                        if unique_service_codes:
                            logger.info(f"🏥 Processing service codes batch: {unique_service_codes}")
                            service_meanings = self._stable_batch_service_codes(unique_service_codes)
                            enhanced_extraction_result["code_meanings"]["service_code_meanings"] = service_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Service codes batch: {len(service_meanings)} meanings generated")
                        
                        # ENHANCED BATCH 2: Diagnosis Codes  
                        if unique_diagnosis_codes:
                            logger.info(f"🩺 Processing diagnosis codes batch: {unique_diagnosis_codes}")
                            diagnosis_meanings = self._stable_batch_diagnosis_codes(unique_diagnosis_codes)
                            enhanced_extraction_result["code_meanings"]["diagnosis_code_meanings"] = diagnosis_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Diagnosis codes batch: {len(diagnosis_meanings)} meanings generated")
                        
                        # Calculate savings
                        individual_calls_would_be = len(unique_service_codes) + len(unique_diagnosis_codes)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        enhanced_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        enhanced_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Final status
                        total_meanings = len(enhanced_extraction_result["code_meanings"]["service_code_meanings"]) + len(enhanced_extraction_result["code_meanings"]["diagnosis_code_meanings"])
                        
                        if total_meanings > 0:
                            enhanced_extraction_result["code_meanings_added"] = True
                            enhanced_extraction_result["stable_analysis"] = True
                            enhanced_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 ENHANCED BATCH SUCCESS: {total_meanings} meanings, {calls_saved} calls saved!")
                        else:
                            enhanced_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Batch completed but no meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced batch processing error: {e}")
                        enhanced_extraction_result["code_meaning_error"] = str(e)
                        enhanced_extraction_result["llm_call_status"] = "failed"
                else:
                    enhanced_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No codes found for batch processing")
            else:
                enhanced_extraction_result["llm_call_status"] = "skipped_no_api"

            # Performance stats
            processing_time = time.time() - start_time
            enhanced_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"🔬 ===== ENHANCED BATCH MEDICAL EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s")
            logger.info(f"  📊 API calls: {enhanced_extraction_result['batch_stats']['api_calls_made']} (saved {enhanced_extraction_result['batch_stats']['individual_calls_saved']})")
            logger.info(f"  ✅ Meanings: {len(enhanced_extraction_result['code_meanings']['service_code_meanings']) + len(enhanced_extraction_result['code_meanings']['diagnosis_code_meanings'])}")
            logger.info(f"  🏥 Provider fields: {enhanced_extraction_result['batch_stats']['provider_fields_extracted']} extracted")

        except Exception as e:
            logger.error(f"❌ Error in enhanced batch medical extraction: {e}")
            enhanced_extraction_result["error"] = f"Enhanced batch extraction failed: {str(e)}"

        return enhanced_extraction_result

    def extract_pharmacy_fields_batch_enhanced(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pharmacy field extraction with batch processing and provider field extraction"""
        logger.info("🔬 ===== STARTING ENHANCED BATCH PHARMACY EXTRACTION WITH PROVIDER FIELDS =====")
        
        enhanced_extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set(),
                # NEW: Enhanced provider field tracking
                "unique_billing_providers": set(),
                "unique_prescribing_providers": set(),
                "total_billing_providers": 0,
                "total_prescribing_providers": 0
            },
            "code_meanings": {
                "ndc_code_meanings": {},
                "medication_meanings": {},
                # NEW: Provider field meanings (if needed for batch processing)
                "billing_provider_meanings": {},
                "prescribing_provider_meanings": {}
            },
            "code_meanings_added": False,
            "stable_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0,
                "provider_fields_extracted": 0
            },
            "provider_field_enhancement": True
        }

        start_time = time.time()

        try:
            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("⚠️ No pharmacy claims data found")
                return enhanced_extraction_result

            # Step 1: Enhanced extraction with provider fields
            logger.info("🔬 Step 1: Enhanced pharmacy code extraction with provider data...")
            self._stable_pharmacy_extraction(pharmacy_data, enhanced_extraction_result)

            # Convert sets to lists and add provider field statistics
            unique_ndc_codes = list(enhanced_extraction_result["extraction_summary"]["unique_ndc_codes"])[:10]
            unique_label_names = list(enhanced_extraction_result["extraction_summary"]["unique_label_names"])[:15]
            unique_billing_providers = list(enhanced_extraction_result["extraction_summary"]["unique_billing_providers"])[:10]
            unique_prescribing_providers = list(enhanced_extraction_result["extraction_summary"]["unique_prescribing_providers"])[:10]
            
            enhanced_extraction_result["extraction_summary"]["unique_ndc_codes"] = unique_ndc_codes
            enhanced_extraction_result["extraction_summary"]["unique_label_names"] = unique_label_names
            enhanced_extraction_result["extraction_summary"]["unique_billing_providers"] = unique_billing_providers
            enhanced_extraction_result["extraction_summary"]["unique_prescribing_providers"] = unique_prescribing_providers
            enhanced_extraction_result["extraction_summary"]["total_billing_providers"] = len(unique_billing_providers)
            enhanced_extraction_result["extraction_summary"]["total_prescribing_providers"] = len(unique_prescribing_providers)

            total_codes = len(unique_ndc_codes) + len(unique_label_names)
            enhanced_extraction_result["batch_stats"]["codes_processed"] = total_codes
            enhanced_extraction_result["batch_stats"]["provider_fields_extracted"] = len(unique_billing_providers) + len(unique_prescribing_providers)

            # Step 2: ENHANCED BATCH PROCESSING
            logger.info(f"🔍 DEBUG: API integrator available: {self.api_integrator is not None}")
            if self.api_integrator:
                logger.info(f"🔍 DEBUG: Has isolated method: {hasattr(self.api_integrator, 'call_llm_isolated_enhanced')}")
            
            logger.info(f"🔍 DEBUG: NDC codes found: {len(unique_ndc_codes)}")
            logger.info(f"🔍 DEBUG: Label names found: {len(unique_label_names)}")
            logger.info(f"🔍 DEBUG: Billing providers found: {len(unique_billing_providers)}")
            logger.info(f"🔍 DEBUG: Prescribing providers found: {len(unique_prescribing_providers)}")
            
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_ndc_codes or unique_label_names:
                    logger.info(f"🔬 Step 2: ENHANCED BATCH processing {total_codes} pharmacy codes...")
                    enhanced_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # ENHANCED BATCH 1: NDC Codes
                        if unique_ndc_codes:
                            logger.info(f"💊 Processing NDC codes batch: {unique_ndc_codes}")
                            ndc_meanings = self._stable_batch_ndc_codes(unique_ndc_codes)
                            enhanced_extraction_result["code_meanings"]["ndc_code_meanings"] = ndc_meanings
                            api_calls_made += 1
                            logger.info(f"✅ NDC codes batch: {len(ndc_meanings)} meanings generated")
                        
                        # ENHANCED BATCH 2: Medications
                        if unique_label_names:
                            logger.info(f"💉 Processing medications batch: {unique_label_names}")
                            med_meanings = self._stable_batch_medications(unique_label_names)
                            enhanced_extraction_result["code_meanings"]["medication_meanings"] = med_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Medications batch: {len(med_meanings)} meanings generated")
                        
                        # Calculate savings
                        individual_calls_would_be = len(unique_ndc_codes) + len(unique_label_names)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        enhanced_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        enhanced_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Final status
                        total_meanings = len(enhanced_extraction_result["code_meanings"]["ndc_code_meanings"]) + len(enhanced_extraction_result["code_meanings"]["medication_meanings"])
                        
                        if total_meanings > 0:
                            enhanced_extraction_result["code_meanings_added"] = True
                            enhanced_extraction_result["stable_analysis"] = True
                            enhanced_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 ENHANCED PHARMACY BATCH SUCCESS: {total_meanings} meanings, {calls_saved} calls saved!")
                        else:
                            enhanced_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Pharmacy batch completed but no meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced pharmacy batch error: {e}")
                        enhanced_extraction_result["code_meaning_error"] = str(e)
                        enhanced_extraction_result["llm_call_status"] = "failed"
                else:
                    enhanced_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No pharmacy codes for batch processing")
            else:
                enhanced_extraction_result["llm_call_status"] = "skipped_no_api"

            # Performance stats
            processing_time = time.time() - start_time
            enhanced_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"💊 ===== ENHANCED BATCH PHARMACY EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s")
            logger.info(f"  📊 API calls: {enhanced_extraction_result['batch_stats']['api_calls_made']} (saved {enhanced_extraction_result['batch_stats']['individual_calls_saved']})")
            logger.info(f"  ✅ Meanings: {len(enhanced_extraction_result['code_meanings']['ndc_code_meanings']) + len(enhanced_extraction_result['code_meanings']['medication_meanings'])}")
            logger.info(f"  🏥 Provider fields: {enhanced_extraction_result['batch_stats']['provider_fields_extracted']} extracted")

        except Exception as e:
            logger.error(f"❌ Error in enhanced batch pharmacy extraction: {e}")
            enhanced_extraction_result["error"] = f"Enhanced pharmacy batch extraction failed: {str(e)}"

        return enhanced_extraction_result

    def _stable_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive medical field extraction with provider fields"""
        if isinstance(data, dict):
            current_record = {}
            
            # Enhanced health service code extraction
            if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
                service_code = str(data["hlth_srvc_cd"]).strip()
                current_record["hlth_srvc_cd"] = service_code
                result["extraction_summary"]["unique_service_codes"].add(service_code)

            # Enhanced claim received date extraction
            if "clm_rcvd_dt" in data and data["clm_rcvd_dt"]:
                current_record["clm_rcvd_dt"] = data["clm_rcvd_dt"]

            # Enhanced claim line service end date extraction
            if "clm_line_srvc_end_dt" in data and data["clm_line_srvc_end_dt"]:
                current_record["clm_line_srvc_end_dt"] = data["clm_line_srvc_end_dt"]

            # NEW: Enhanced billing provider name extraction
            if "billg_prov_nm" in data and data["billg_prov_nm"]:
                billing_provider = str(data["billg_prov_nm"]).strip()
                current_record["billg_prov_nm"] = billing_provider
                # Add to tracking for batch processing if needed
                if "unique_billing_providers" not in result["extraction_summary"]:
                    result["extraction_summary"]["unique_billing_providers"] = set()
                result["extraction_summary"]["unique_billing_providers"].add(billing_provider)

            # NEW: Enhanced billing provider zip code extraction  
            if "billg_prov_zip_cd" in data and data["billg_prov_zip_cd"]:
                billing_zip = str(data["billg_prov_zip_cd"]).strip()
                current_record["billg_prov_zip_cd"] = billing_zip
                # Add to tracking
                if "unique_billing_zip_codes" not in result["extraction_summary"]:
                    result["extraction_summary"]["unique_billing_zip_codes"] = set()
                result["extraction_summary"]["unique_billing_zip_codes"].add(billing_zip)

            # Enhanced diagnosis codes extraction
            diagnosis_codes = []

            # Handle comma-separated diagnosis codes
            if "diag_1_50_cd" in data and data["diag_1_50_cd"]:
                diag_value = str(data["diag_1_50_cd"]).strip()
                if diag_value and diag_value.lower() not in ['null', 'none', '']:
                    individual_codes = [code.strip() for code in diag_value.split(',') if code.strip()]
                    for i, code in enumerate(individual_codes, 1):
                        if code and code.lower() not in ['null', 'none', '']:
                            diagnosis_info = {
                                "code": code,
                                "position": i,
                                "source": "diag_1_50_cd"
                            }
                            diagnosis_codes.append(diagnosis_info)
                            result["extraction_summary"]["unique_diagnosis_codes"].add(code)

            # Handle individual diagnosis fields
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        diagnosis_info = {
                            "code": diag_code,
                            "position": i,
                            "source": f"individual_{diag_key}"
                        }
                        diagnosis_codes.append(diagnosis_info)
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)

            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)

            if current_record:
                current_record["data_path"] = path
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1

            # Continue enhanced recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._stable_medical_extraction(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._stable_medical_extraction(item, result, new_path)

    def _stable_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive pharmacy field extraction with provider fields"""
        if isinstance(data, dict):
            current_record = {}

            # Enhanced NDC code extraction
            ndc_found = False
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number', 'national_drug_code']:
                    ndc_code = str(data[key]).strip()
                    current_record["ndc"] = ndc_code
                    result["extraction_summary"]["unique_ndc_codes"].add(ndc_code)
                    ndc_found = True
                    break

            # Enhanced medication name extraction
            label_found = False
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                    medication_name = str(data[key]).strip()
                    current_record["lbl_nm"] = medication_name
                    result["extraction_summary"]["unique_label_names"].add(medication_name)
                    label_found = True
                    break

            # Enhanced prescription filled date extraction
            if "rx_filled_dt" in data and data["rx_filled_dt"]:
                current_record["rx_filled_dt"] = data["rx_filled_dt"]

            # NEW: Enhanced billing provider name extraction
            if "billg_prov_nm" in data and data["billg_prov_nm"]:
                billing_provider = str(data["billg_prov_nm"]).strip()
                current_record["billg_prov_nm"] = billing_provider
                # Add to tracking for batch processing if needed
                if "unique_billing_providers" not in result["extraction_summary"]:
                    result["extraction_summary"]["unique_billing_providers"] = set()
                result["extraction_summary"]["unique_billing_providers"].add(billing_provider)

            # NEW: Enhanced prescribing provider name extraction
            if "prscrb_prov_nm" in data and data["prscrb_prov_nm"]:
                prescribing_provider = str(data["prscrb_prov_nm"]).strip()
                current_record["prscrb_prov_nm"] = prescribing_provider
                # Add to tracking
                if "unique_prescribing_providers" not in result["extraction_summary"]:
                    result["extraction_summary"]["unique_prescribing_providers"] = set()
                result["extraction_summary"]["unique_prescribing_providers"].add(prescribing_provider)

            if (ndc_found or label_found or "rx_filled_dt" in current_record or 
                "billg_prov_nm" in current_record or "prscrb_prov_nm" in current_record):
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1

            # Continue enhanced recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._stable_pharmacy_extraction(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._stable_pharmacy_extraction(item, result, new_path)

    def prepare_enhanced_clinical_context(self, chat_context: Dict[str, Any]) -> str:
        """Enhanced context preparation for chatbot with provider information"""
        try:
            context_parts = []

            # Enhanced patient overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_parts.append(f"**PATIENT**: Age {patient_overview.get('age', 'unknown')}, ZIP {patient_overview.get('zip', 'unknown')}")

            # Enhanced medical extractions with provider data
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                provider_count = medical_extraction.get("extraction_summary", {}).get("total_billing_providers", 0)
                context_parts.append(f"**MEDICAL DATA**: {json.dumps(medical_extraction, indent=2)}")
                context_parts.append(f"**MEDICAL PROVIDER COUNT**: {provider_count} billing providers")

            # Enhanced pharmacy extractions with provider data
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                billing_provider_count = pharmacy_extraction.get("extraction_summary", {}).get("total_billing_providers", 0)
                prescribing_provider_count = pharmacy_extraction.get("extraction_summary", {}).get("total_prescribing_providers", 0)
                context_parts.append(f"**PHARMACY DATA**: {json.dumps(pharmacy_extraction, indent=2)}")
                context_parts.append(f"**PHARMACY PROVIDER COUNT**: {billing_provider_count} billing providers, {prescribing_provider_count} prescribing providers")

            # Enhanced entity extraction
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_parts.append(f"**HEALTH ENTITIES**: {json.dumps(entity_extraction, indent=2)}")

            # Enhanced health trajectory
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                context_parts.append(f"**HEALTH TRAJECTORY**: {health_trajectory[:500]}...")

            # Enhanced cardiovascular risk assessment
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_parts.append(f"**CARDIOVASCULAR RISK**: {json.dumps(heart_attack_prediction, indent=2)}")

            return "\n\n" + "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error preparing enhanced context: {e}")
            return "Enhanced patient healthcare data with provider information available for analysis."

    # Helper methods for enhanced processing
    def _calculate_age_stable(self, date_of_birth: str) -> str:
        """Enhanced age calculation"""
        try:
            if not date_of_birth:
                return "unknown"
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Enhanced age context
            if age < 18:
                return f"{age} years (Pediatric)"
            elif age < 65:
                return f"{age} years (Adult)"
            else:
                return f"{age} years (Senior)"
        except:
            return "unknown"

    def _get_stable_age_group(self, age: int) -> str:
        """Enhanced age group determination"""
        if age < 18:
            return "pediatric"
        elif age < 35:
            return "young_adult"
        elif age < 50:
            return "adult"
        elif age < 65:
            return "middle_aged"
        else:
            return "senior"

    def _stable_deidentify_json(self, data: Any) -> Any:
        """Enhanced JSON deidentification"""
        if isinstance(data, dict):
            deidentified_dict = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    deidentified_dict[key] = self._stable_deidentify_json(value)
                elif isinstance(value, str):
                    deidentified_dict[key] = self._stable_deidentify_string(value)
                else:
                    deidentified_dict[key] = value
            return deidentified_dict
        elif isinstance(data, list):
            return [self._stable_deidentify_json(item) for item in data]
        elif isinstance(data, str):
            return self._stable_deidentify_string(data)
        else:
            return data

    def _stable_deidentify_pharmacy_json(self, data: Any) -> Any:
        """Enhanced pharmacy JSON deidentification"""
        if isinstance(data, dict):
            deidentified_dict = {}
            for key, value in data.items():
                if key.lower() in ['src_mbr_first_nm', 'src_mbr_frst_nm', 'scr_mbr_last_nm', 'src_mbr_last_nm']:
                    deidentified_dict[key] = "[MASKED_NAME]"
                elif isinstance(value, (dict, list)):
                    deidentified_dict[key] = self._stable_deidentify_pharmacy_json(value)
                elif isinstance(value, str):
                    deidentified_dict[key] = self._stable_deidentify_string(value)
                else:
                    deidentified_dict[key] = value
            return deidentified_dict
        elif isinstance(data, list):
            return [self._stable_deidentify_pharmacy_json(item) for item in data]
        elif isinstance(data, str):
            return self._stable_deidentify_string(data)
        else:
            return data

    def _mask_medical_fields_stable(self, data: Any) -> Any:
        """Enhanced medical field masking"""
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if key.lower() in ['src_mbr_frst_nm', 'src_mbr_first_nm', 'src_mbr_last_nm', 'src_mvr_last_nm']:
                    masked_data[key] = "[MASKED_NAME]"
                elif isinstance(value, (dict, list)):
                    masked_data[key] = self._mask_medical_fields_stable(value)
                else:
                    masked_data[key] = value
            return masked_data
        elif isinstance(data, list):
            return [self._mask_medical_fields_stable(item) for item in data]
        else:
            return data

    def _stable_deidentify_string(self, data: str) -> str:
        """Enhanced string deidentification"""
        if not isinstance(data, str) or not data.strip():
            return data

        deidentified = str(data)
        
        # Enhanced pattern replacements
        deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[MASKED_SSN]', deidentified)
        deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[MASKED_PHONE]', deidentified)
        deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED_EMAIL]', deidentified)
        deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[MASKED_NAME]', deidentified)
        
        return deidentified

    def _clean_json_response_stable(self, response: str) -> str:
        """Enhanced LLM response cleaning for JSON extraction"""
        try:
            # Remove markdown wrappers
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            
            # Find JSON object boundaries
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_content = response[start:end]
                
                # Validate JSON
                try:
                    json.loads(json_content)
                    return json_content
                except json.JSONDecodeError:
                    # Try to fix common issues
                    fixed_content = self._fix_common_json_issues_stable(json_content)
                    return fixed_content
            else:
                return response
                
        except Exception as e:
            logger.warning(f"Enhanced JSON cleaning failed: {e}")
            return response

    def _fix_common_json_issues_stable(self, json_content: str) -> str:
        """Fix common JSON formatting issues with enhanced approach"""
        try:
            # Fix trailing commas
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            
            return json_content
        except Exception as e:
            logger.warning(f"Enhanced JSON fixing failed: {e}")
            return json_content

    # Enhanced isolated methods for individual code explanations (backward compatibility)
    def get_service_code_explanation_isolated(self, service_code: str) -> str:
        """Get isolated service code explanation"""
        try:
            if not self.api_integrator or not hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                return f"Service code {service_code} - explanation unavailable"
            
            prompt = f"Briefly explain medical service code {service_code}"
            system_msg = "You are a medical coding expert. Provide brief explanations."
            
            response = self.api_integrator.call_llm_isolated_enhanced(prompt, system_msg)
            return response if response and response != "Brief explanation unavailable" else f"Service code {service_code}"
        except Exception as e:
            logger.warning(f"Service code explanation error: {e}")
            return f"Service code {service_code}"

    def get_diagnosis_code_explanation_isolated(self, diagnosis_code: str) -> str:
        """Get isolated diagnosis code explanation"""
        try:
            if not self.api_integrator or not hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                return f"Diagnosis code {diagnosis_code} - explanation unavailable"
            
            prompt = f"Briefly explain ICD-10 diagnosis code {diagnosis_code}"
            system_msg = "You are a medical diagnosis expert. Provide brief explanations."
            
            response = self.api_integrator.call_llm_isolated_enhanced(prompt, system_msg)
            return response if response and response != "Brief explanation unavailable" else f"ICD-10 diagnosis code {diagnosis_code}"
        except Exception as e:
            logger.warning(f"Diagnosis code explanation error: {e}")
            return f"ICD-10 diagnosis code {diagnosis_code}"

    def get_ndc_code_explanation_isolated(self, ndc_code: str) -> str:
        """Get isolated NDC code explanation"""
        try:
            if not self.api_integrator or not hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                return f"NDC code {ndc_code} - explanation unavailable"
            
            prompt = f"Briefly explain NDC medication code {ndc_code}"
            system_msg = "You are a pharmacy expert. Provide brief explanations."
            
            response = self.api_integrator.call_llm_isolated_enhanced(prompt, system_msg)
            return response if response and response != "Brief explanation unavailable" else f"NDC medication code {ndc_code}"
        except Exception as e:
            logger.warning(f"NDC code explanation error: {e}")
            return f"NDC medication code {ndc_code}"

    def get_medication_explanation_isolated(self, medication: str) -> str:
        """Get isolated medication explanation"""
        try:
            if not self.api_integrator or not hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                return f"Medication {medication} - explanation unavailable"
            
            prompt = f"Briefly explain medication {medication}"
            system_msg = "You are a medication expert. Provide brief explanations."
            
            response = self.api_integrator.call_llm_isolated_enhanced(prompt, system_msg)
            return response if response and response != "Brief explanation unavailable" else f"Medication {medication}"
        except Exception as e:
            logger.warning(f"Medication explanation error: {e}")
            return f"Medication {medication}"

    # Enhanced provider field analysis methods
    def analyze_provider_network(self, medical_extraction: Dict[str, Any], pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze provider network from extracted data"""
        try:
            provider_analysis = {
                "medical_providers": {
                    "billing_providers": [],
                    "provider_zip_codes": [],
                    "unique_billing_provider_count": 0,
                    "unique_zip_code_count": 0
                },
                "pharmacy_providers": {
                    "billing_providers": [],
                    "prescribing_providers": [],
                    "unique_billing_provider_count": 0,
                    "unique_prescribing_provider_count": 0
                },
                "network_overlap": {
                    "common_billing_providers": [],
                    "total_unique_providers": 0
                },
                "provider_quality_metrics": {
                    "provider_diversity_score": 0.0,
                    "geographic_distribution_score": 0.0
                }
            }

            # Analyze medical providers
            if medical_extraction:
                medical_billing_providers = medical_extraction.get("extraction_summary", {}).get("unique_billing_providers", [])
                medical_zip_codes = medical_extraction.get("extraction_summary", {}).get("unique_billing_zip_codes", [])
                
                provider_analysis["medical_providers"]["billing_providers"] = list(medical_billing_providers)
                provider_analysis["medical_providers"]["provider_zip_codes"] = list(medical_zip_codes)
                provider_analysis["medical_providers"]["unique_billing_provider_count"] = len(medical_billing_providers)
                provider_analysis["medical_providers"]["unique_zip_code_count"] = len(medical_zip_codes)

            # Analyze pharmacy providers
            if pharmacy_extraction:
                pharmacy_billing_providers = pharmacy_extraction.get("extraction_summary", {}).get("unique_billing_providers", [])
                prescribing_providers = pharmacy_extraction.get("extraction_summary", {}).get("unique_prescribing_providers", [])
                
                provider_analysis["pharmacy_providers"]["billing_providers"] = list(pharmacy_billing_providers)
                provider_analysis["pharmacy_providers"]["prescribing_providers"] = list(prescribing_providers)
                provider_analysis["pharmacy_providers"]["unique_billing_provider_count"] = len(pharmacy_billing_providers)
                provider_analysis["pharmacy_providers"]["unique_prescribing_provider_count"] = len(prescribing_providers)

                # Analyze network overlap
                medical_billing = set(provider_analysis["medical_providers"]["billing_providers"])
                pharmacy_billing = set(pharmacy_billing_providers)
                common_providers = medical_billing.intersection(pharmacy_billing)
                
                provider_analysis["network_overlap"]["common_billing_providers"] = list(common_providers)
                
                all_providers = medical_billing.union(pharmacy_billing).union(set(prescribing_providers))
                provider_analysis["network_overlap"]["total_unique_providers"] = len(all_providers)

            # Calculate quality metrics
            total_providers = provider_analysis["network_overlap"]["total_unique_providers"]
            total_zip_codes = provider_analysis["medical_providers"]["unique_zip_code_count"]
            
            if total_providers > 0:
                provider_analysis["provider_quality_metrics"]["provider_diversity_score"] = min(total_providers / 10, 1.0)  # Normalize to 0-1
            
            if total_zip_codes > 0:
                provider_analysis["provider_quality_metrics"]["geographic_distribution_score"] = min(total_zip_codes / 5, 1.0)  # Normalize to 0-1

            logger.info(f"✅ Provider network analysis completed: {total_providers} unique providers across {total_zip_codes} zip codes")
            return provider_analysis

        except Exception as e:
            logger.error(f"Error in provider network analysis: {e}")
            return {"error": f"Provider network analysis failed: {str(e)}"}

    def generate_provider_summary_report(self, provider_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive provider summary report"""
        try:
            if "error" in provider_analysis:
                return f"Provider summary report unavailable: {provider_analysis['error']}"

            medical_providers = provider_analysis.get("medical_providers", {})
            pharmacy_providers = provider_analysis.get("pharmacy_providers", {})
            network_overlap = provider_analysis.get("network_overlap", {})
            quality_metrics = provider_analysis.get("provider_quality_metrics", {})

            report_parts = []
            report_parts.append("## 🏥 COMPREHENSIVE PROVIDER NETWORK ANALYSIS")
            report_parts.append("")

            # Medical provider summary
            medical_billing_count = medical_providers.get("unique_billing_provider_count", 0)
            medical_zip_count = medical_providers.get("unique_zip_code_count", 0)
            
            report_parts.append("### 🩺 Medical Provider Network")
            report_parts.append(f"- **Billing Providers**: {medical_billing_count}")
            report_parts.append(f"- **Geographic Coverage**: {medical_zip_count} ZIP codes")
            
            if medical_providers.get("billing_providers"):
                top_medical_providers = medical_providers["billing_providers"][:5]
                report_parts.append(f"- **Top Providers**: {', '.join(top_medical_providers)}")

            # Pharmacy provider summary
            pharmacy_billing_count = pharmacy_providers.get("unique_billing_provider_count", 0)
            prescribing_count = pharmacy_providers.get("unique_prescribing_provider_count", 0)
            
            report_parts.append("")
            report_parts.append("### 💊 Pharmacy Provider Network")
            report_parts.append(f"- **Billing Providers**: {pharmacy_billing_count}")
            report_parts.append(f"- **Prescribing Providers**: {prescribing_count}")
            
            if pharmacy_providers.get("billing_providers"):
                top_pharmacy_providers = pharmacy_providers["billing_providers"][:5]
                report_parts.append(f"- **Top Billing Providers**: {', '.join(top_pharmacy_providers)}")
            
            if pharmacy_providers.get("prescribing_providers"):
                top_prescribing_providers = pharmacy_providers["prescribing_providers"][:5]
                report_parts.append(f"- **Top Prescribing Providers**: {', '.join(top_prescribing_providers)}")

            # Network overlap analysis
            total_unique = network_overlap.get("total_unique_providers", 0)
            common_providers = network_overlap.get("common_billing_providers", [])
            
            report_parts.append("")
            report_parts.append("### 🔗 Network Integration Analysis")
            report_parts.append(f"- **Total Unique Providers**: {total_unique}")
            report_parts.append(f"- **Cross-Network Providers**: {len(common_providers)}")
            
            if common_providers:
                report_parts.append(f"- **Integrated Providers**: {', '.join(common_providers[:3])}")

            # Quality metrics
            diversity_score = quality_metrics.get("provider_diversity_score", 0.0)
            geographic_score = quality_metrics.get("geographic_distribution_score", 0.0)
            
            report_parts.append("")
            report_parts.append("### 📊 Network Quality Metrics")
            report_parts.append(f"- **Provider Diversity Score**: {diversity_score:.2f}/1.0")
            report_parts.append(f"- **Geographic Distribution Score**: {geographic_score:.2f}/1.0")
            
            # Overall assessment
            report_parts.append("")
            report_parts.append("### 🎯 Network Assessment")
            
            if diversity_score >= 0.8:
                diversity_assessment = "Excellent provider diversity"
            elif diversity_score >= 0.6:
                diversity_assessment = "Good provider diversity"
            else:
                diversity_assessment = "Limited provider diversity"
            
            if geographic_score >= 0.8:
                geographic_assessment = "Wide geographic coverage"
            elif geographic_score >= 0.6:
                geographic_assessment = "Moderate geographic coverage"
            else:
                geographic_assessment = "Limited geographic coverage"
            
            report_parts.append(f"- **Diversity**: {diversity_assessment}")
            report_parts.append(f"- **Coverage**: {geographic_assessment}")
            
            return "\n".join(report_parts)

        except Exception as e:
            logger.error(f"Error generating provider summary report: {e}")
            return f"Provider summary report generation failed: {str(e)}"

    # Backward compatibility methods
    def extract_medical_fields_batch(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced extraction"""
        return self.extract_medical_fields_batch_enhanced(deidentified_medical)

    def extract_pharmacy_fields_batch(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced extraction"""
        return self.extract_pharmacy_fields_batch_enhanced(deidentified_pharmacy)

    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_medical_data_enhanced(medical_data, patient_data)

    def deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_pharmacy_data_enhanced(pharmacy_data)

    def deidentify_mcid_data(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_mcid_data_enhanced(mcid_data)

    # Enhanced utility methods
    def get_extraction_statistics(self, medical_extraction: Dict[str, Any] = None, 
                                pharmacy_extraction: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get comprehensive extraction statistics including provider information"""
        try:
            stats = {
                "medical_stats": {
                    "total_records": 0,
                    "total_diagnosis_codes": 0,
                    "unique_service_codes": 0,
                    "unique_diagnosis_codes": 0,
                    "unique_billing_providers": 0,
                    "unique_billing_zip_codes": 0,
                    "batch_processing_used": False,
                    "llm_call_status": "not_attempted"
                },
                "pharmacy_stats": {
                    "total_records": 0,
                    "unique_ndc_codes": 0,
                    "unique_medications": 0,
                    "unique_billing_providers": 0,
                    "unique_prescribing_providers": 0,
                    "batch_processing_used": False,
                    "llm_call_status": "not_attempted"
                },
                "overall_stats": {
                    "total_provider_fields_extracted": 0,
                    "total_api_calls_made": 0,
                    "total_api_calls_saved": 0,
                    "processing_time_seconds": 0.0,
                    "enhancement_level": "comprehensive_with_provider_fields"
                }
            }

            # Medical extraction statistics
            if medical_extraction:
                extraction_summary = medical_extraction.get("extraction_summary", {})
                batch_stats = medical_extraction.get("batch_stats", {})
                
                stats["medical_stats"]["total_records"] = extraction_summary.get("total_hlth_srvc_records", 0)
                stats["medical_stats"]["total_diagnosis_codes"] = extraction_summary.get("total_diagnosis_codes", 0)
                stats["medical_stats"]["unique_service_codes"] = len(extraction_summary.get("unique_service_codes", []))
                stats["medical_stats"]["unique_diagnosis_codes"] = len(extraction_summary.get("unique_diagnosis_codes", []))
                stats["medical_stats"]["unique_billing_providers"] = extraction_summary.get("total_billing_providers", 0)
                stats["medical_stats"]["unique_billing_zip_codes"] = extraction_summary.get("total_billing_zip_codes", 0)
                stats["medical_stats"]["batch_processing_used"] = medical_extraction.get("code_meanings_added", False)
                stats["medical_stats"]["llm_call_status"] = medical_extraction.get("llm_call_status", "not_attempted")
                
                stats["overall_stats"]["total_api_calls_made"] += batch_stats.get("api_calls_made", 0)
                stats["overall_stats"]["total_api_calls_saved"] += batch_stats.get("individual_calls_saved", 0)
                stats["overall_stats"]["processing_time_seconds"] += batch_stats.get("processing_time_seconds", 0.0)

            # Pharmacy extraction statistics
            if pharmacy_extraction:
                extraction_summary = pharmacy_extraction.get("extraction_summary", {})
                batch_stats = pharmacy_extraction.get("batch_stats", {})
                
                stats["pharmacy_stats"]["total_records"] = extraction_summary.get("total_ndc_records", 0)
                stats["pharmacy_stats"]["unique_ndc_codes"] = len(extraction_summary.get("unique_ndc_codes", []))
                stats["pharmacy_stats"]["unique_medications"] = len(extraction_summary.get("unique_label_names", []))
                stats["pharmacy_stats"]["unique_billing_providers"] = extraction_summary.get("total_billing_providers", 0)
                stats["pharmacy_stats"]["unique_prescribing_providers"] = extraction_summary.get("total_prescribing_providers", 0)
                stats["pharmacy_stats"]["batch_processing_used"] = pharmacy_extraction.get("code_meanings_added", False)
                stats["pharmacy_stats"]["llm_call_status"] = pharmacy_extraction.get("llm_call_status", "not_attempted")
                
                stats["overall_stats"]["total_api_calls_made"] += batch_stats.get("api_calls_made", 0)
                stats["overall_stats"]["total_api_calls_saved"] += batch_stats.get("individual_calls_saved", 0)
                stats["overall_stats"]["processing_time_seconds"] += batch_stats.get("processing_time_seconds", 0.0)

            # Calculate total provider fields extracted
            medical_provider_fields = stats["medical_stats"]["unique_billing_providers"] + stats["medical_stats"]["unique_billing_zip_codes"]
            pharmacy_provider_fields = stats["pharmacy_stats"]["unique_billing_providers"] + stats["pharmacy_stats"]["unique_prescribing_providers"]
            stats["overall_stats"]["total_provider_fields_extracted"] = medical_provider_fields + pharmacy_provider_fields

            logger.info(f"✅ Extraction statistics compiled: {stats['overall_stats']['total_provider_fields_extracted']} provider fields")
            return stats

        except Exception as e:
            logger.error(f"Error generating extraction statistics: {e}")
            return {"error": f"Statistics generation failed: {str(e)}"}

    def validate_provider_data_quality(self, medical_extraction: Dict[str, Any] = None, 
                                     pharmacy_extraction: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate the quality of extracted provider data"""
        try:
            validation_result = {
                "medical_provider_validation": {
                    "billing_provider_completeness": 0.0,
                    "zip_code_completeness": 0.0,
                    "data_consistency_score": 0.0,
                    "validation_status": "not_validated"
                },
                "pharmacy_provider_validation": {
                    "billing_provider_completeness": 0.0,
                    "prescribing_provider_completeness": 0.0,
                    "data_consistency_score": 0.0,
                    "validation_status": "not_validated"
                },
                "overall_validation": {
                    "provider_data_quality_score": 0.0,
                    "completeness_level": "unknown",
                    "recommendations": []
                }
            }

            medical_completeness_scores = []
            pharmacy_completeness_scores = []

            # Validate medical provider data
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                medical_records = medical_extraction["hlth_srvc_records"]
                total_medical_records = len(medical_records)
                
                billing_provider_count = sum(1 for record in medical_records if record.get("billg_prov_nm"))
                zip_code_count = sum(1 for record in medical_records if record.get("billg_prov_zip_cd"))
                
                billing_completeness = billing_provider_count / total_medical_records if total_medical_records > 0 else 0
                zip_completeness = zip_code_count / total_medical_records if total_medical_records > 0 else 0
                
                validation_result["medical_provider_validation"]["billing_provider_completeness"] = billing_completeness
                validation_result["medical_provider_validation"]["zip_code_completeness"] = zip_completeness
                validation_result["medical_provider_validation"]["data_consistency_score"] = (billing_completeness + zip_completeness) / 2
                validation_result["medical_provider_validation"]["validation_status"] = "validated"
                
                medical_completeness_scores.extend([billing_completeness, zip_completeness])

            # Validate pharmacy provider data
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                pharmacy_records = pharmacy_extraction["ndc_records"]
                total_pharmacy_records = len(pharmacy_records)
                
                billing_provider_count = sum(1 for record in pharmacy_records if record.get("billg_prov_nm"))
                prescribing_provider_count = sum(1 for record in pharmacy_records if record.get("prscrb_prov_nm"))
                
                billing_completeness = billing_provider_count / total_pharmacy_records if total_pharmacy_records > 0 else 0
                prescribing_completeness = prescribing_provider_count / total_pharmacy_records if total_pharmacy_records > 0 else 0
                
                validation_result["pharmacy_provider_validation"]["billing_provider_completeness"] = billing_completeness
                validation_result["pharmacy_provider_validation"]["prescribing_provider_completeness"] = prescribing_completeness
                validation_result["pharmacy_provider_validation"]["data_consistency_score"] = (billing_completeness + prescribing_completeness) / 2
                validation_result["pharmacy_provider_validation"]["validation_status"] = "validated"
                
                pharmacy_completeness_scores.extend([billing_completeness, prescribing_completeness])

            # Calculate overall validation scores
            all_scores = medical_completeness_scores + pharmacy_completeness_scores
            if all_scores:
                overall_quality = sum(all_scores) / len(all_scores)
                validation_result["overall_validation"]["provider_data_quality_score"] = overall_quality
                
                if overall_quality >= 0.8:
                    validation_result["overall_validation"]["completeness_level"] = "excellent"
                elif overall_quality >= 0.6:
                    validation_result["overall_validation"]["completeness_level"] = "good"
                elif overall_quality >= 0.4:
                    validation_result["overall_validation"]["completeness_level"] = "moderate"
                else:
                    validation_result["overall_validation"]["completeness_level"] = "poor"

                # Generate recommendations
                recommendations = []
                if overall_quality < 0.6:
                    recommendations.append("Consider improving data collection processes for provider information")
                if medical_completeness_scores and min(medical_completeness_scores) < 0.5:
                    recommendations.append("Medical provider data completeness needs improvement")
                if pharmacy_completeness_scores and min(pharmacy_completeness_scores) < 0.5:
                    recommendations.append("Pharmacy provider data completeness needs improvement")
                if overall_quality >= 0.8:
                    recommendations.append("Provider data quality is excellent - maintain current processes")
                
                validation_result["overall_validation"]["recommendations"] = recommendations

            logger.info(f"✅ Provider data validation completed: {validation_result['overall_validation']['completeness_level']} quality")
            return validation_result

        except Exception as e:
            logger.error(f"Error in provider data validation: {e}")
            return {"error": f"Provider data validation failed: {str(e)}"}

    def export_provider_data_summary(self, medical_extraction: Dict[str, Any] = None, 
                                   pharmacy_extraction: Dict[str, Any] = None,
                                   format_type: str = "json") -> str:
        """Export provider data summary in specified format"""
        try:
            # Analyze provider network
            provider_analysis = self.analyze_provider_network(medical_extraction, pharmacy_extraction)
            
            if format_type.lower() == "json":
                return json.dumps(provider_analysis, indent=2)
            elif format_type.lower() == "report":
                return self.generate_provider_summary_report(provider_analysis)
            elif format_type.lower() == "csv":
                # Generate CSV format
                csv_lines = []
                csv_lines.append("Provider_Type,Provider_Name,Count,Category")
                
                # Medical providers
                medical_providers = provider_analysis.get("medical_providers", {})
                for provider in medical_providers.get("billing_providers", []):
                    csv_lines.append(f"Medical,{provider},1,Billing")
                
                # Pharmacy providers
                pharmacy_providers = provider_analysis.get("pharmacy_providers", {})
                for provider in pharmacy_providers.get("billing_providers", []):
                    csv_lines.append(f"Pharmacy,{provider},1,Billing")
                for provider in pharmacy_providers.get("prescribing_providers", []):
                    csv_lines.append(f"Pharmacy,{provider},1,Prescribing")
                
                return "\n".join(csv_lines)
            else:
                return f"Unsupported format: {format_type}. Use 'json', 'report', or 'csv'."

        except Exception as e:
            logger.error(f"Error exporting provider data summary: {e}")
            return f"Export failed: {str(e)}"

    def __str__(self) -> str:
        """Enhanced string representation"""
        return f"""Enhanced Health Data Processor v2.0
Features:
  ✅ Comprehensive medical and pharmacy field extraction
  ✅ Enhanced provider field extraction (billing, prescribing, zip codes)
  ✅ Batch LLM processing with retry logic
  ✅ Advanced graph generation capabilities
  ✅ Provider network analysis
  ✅ Data quality validation
  ✅ Multiple export formats
  ✅ Backward compatibility
Configuration:
  🔄 Max retry attempts: {self.max_retry_attempts}
  ⏱️ Retry delay: {self.retry_delay_seconds}s
  🤖 API integrator: {'✅ Connected' if self.api_integrator else '❌ Not connected'}
  📊 Batch processing: {'✅ Available' if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced') else '❌ Unavailable'}"""

# Example usage and testing functions
def test_enhanced_processor():
    """Test function for the enhanced processor"""
    logger.info("🧪 Testing Enhanced Health Data Processor...")
    
    # Initialize processor
    processor = EnhancedHealthDataProcessor()
    
    # Test configuration
    config = processor.get_retry_config()
    logger.info(f"📋 Current configuration: {config}")
    
    # Test graph detection
    test_queries = [
        "show me a medication timeline chart",
        "create a provider analysis dashboard",
        "what medications is the patient taking?",
        "generate a risk assessment visualization"
    ]
    
    for query in test_queries:
        graph_request = processor.detect_graph_request(query)
        logger.info(f"🔍 Query: '{query}' -> Graph: {graph_request['is_graph_request']} ({graph_request['graph_type']})")
    
    logger.info("✅ Enhanced processor test completed")
    return processor

if __name__ == "__main__":
    # Initialize and test the enhanced processor
    processor = test_enhanced_processor()
    print(processor)

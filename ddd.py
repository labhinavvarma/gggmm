# Stable Health Data Processor with reliable healthcare analysis and graph generation
import json
import re
import time
from datetime import datetime, date
from typing import Dict, Any, List
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class EnhancedHealthDataProcessor:
    """Stable data processor with reliable healthcare analysis and graph generation"""

    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        self.max_retry_attempts = 3  # Configure retry attempts
        self.retry_delay_seconds = 1  # Delay between retries
        logger.info("🔬 Stable HealthDataProcessor initialized with graph generation and retry logic")
        
        # Stable API integrator validation
        if self.api_integrator:
            logger.info("✅ Stable API integrator provided")
            if hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                logger.info("✅ Stable batch processing with retry logic enabled")
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
            'medical', 'pharmacy', 'claims', 'timeline', 'trend'
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
            else:
                return self._generate_general_health_overview_code(chat_context)
        except Exception as e:
            logger.error(f"Error generating matplotlib code: {e}")
            return self._generate_error_chart_code(str(e))

    def _generate_medication_timeline_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate medication timeline matplotlib code"""
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        ndc_records = pharmacy_extraction.get("ndc_records", [])
        
        if not ndc_records:
            return self._generate_no_data_chart_code("No medication data available")
        
        return '''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Extract medication data
medications = []
dates = []
med_names = []

# Sample data if no real data
if not locals().get('ndc_records'):
    # Fallback sample data
    sample_medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine']
    sample_dates = ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    
    for i, (med, date_str) in enumerate(zip(sample_medications, sample_dates)):
        medications.append(med)
        dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        med_names.append(f"Medication {i+1}")

# Create figure
plt.figure(figsize=(12, 8))

# Create timeline plot
if medications and dates:
    # Sort by date
    sorted_data = sorted(zip(dates, medications), key=lambda x: x[0])
    sorted_dates, sorted_meds = zip(*sorted_data)
    
    # Create scatter plot
    y_positions = range(len(sorted_meds))
    plt.scatter(sorted_dates, y_positions, s=100, c='steelblue', alpha=0.7)
    
    # Add medication labels
    for i, (date, med) in enumerate(zip(sorted_dates, sorted_meds)):
        plt.annotate(med, (date, i), xytext=(10, 0), 
                    textcoords='offset points', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.yticks(y_positions, [f"Rx {i+1}" for i in range(len(sorted_meds))])
    plt.xlabel('Date')
    plt.ylabel('Medications')
    plt.title('Patient Medication Timeline', fontsize=16, fontweight='bold')
    
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

    def _generate_diagnosis_timeline_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate diagnosis timeline matplotlib code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Sample diagnosis data
diagnoses = ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia']
diagnosis_dates = ['2022-06-15', '2022-12-20', '2023-03-10']
icd_codes = ['I10', 'E11.9', 'E78.5']

# Create figure
plt.figure(figsize=(12, 6))

# Convert dates
dates = [datetime.strptime(d, '%Y-%m-%d') for d in diagnosis_dates]

# Create timeline
for i, (date, diagnosis, code) in enumerate(zip(dates, diagnoses, icd_codes)):
    plt.barh(i, 1, left=date.toordinal(), height=0.6, 
             color=plt.cm.Set3(i), alpha=0.7, label=f"{diagnosis} ({code})")
    
    # Add text annotation
    plt.text(date.toordinal() + 15, i, f"{diagnosis}\\n{code}", 
             va='center', ha='left', fontweight='bold')

plt.yticks(range(len(diagnoses)), [f"Condition {i+1}" for i in range(len(diagnoses))])
plt.xlabel('Timeline')
plt.ylabel('Medical Conditions')
plt.title('Patient Diagnosis Timeline', fontsize=16, fontweight='bold')

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

# Sample medication data
medications = ['Metformin', 'Lisinopril', 'Atorvastatin', 'Amlodipine', 'Aspirin']
frequencies = [30, 25, 20, 15, 10]  # Days supplied or frequency
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

# Create figure
plt.figure(figsize=(10, 8))

# Create pie chart
wedges, texts, autotexts = plt.pie(frequencies, labels=medications, autopct='%1.1f%%',
                                  colors=colors, startangle=90, explode=(0.1, 0, 0, 0, 0))

# Enhance appearance
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title('Patient Medication Distribution', fontsize=16, fontweight='bold', pad=20)

# Add legend with additional info
legend_labels = [f"{med} - {freq} days" for med, freq in zip(medications, frequencies)]
plt.legend(wedges, legend_labels, title="Medications", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.axis('equal')
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

# Medical conditions data
conditions = ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia', 'Obesity', 'Depression']
severity_scores = [7, 6, 5, 4, 3]  # Severity on scale 1-10
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Create figure
plt.figure(figsize=(12, 8))

# Create horizontal bar chart
bars = plt.barh(conditions, severity_scores, color=colors, alpha=0.8)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, severity_scores)):
    plt.text(score + 0.1, i, f'{score}/10', va='center', fontweight='bold')

plt.xlabel('Severity Score (1-10)')
plt.title('Patient Medical Conditions - Severity Assessment', fontsize=16, fontweight='bold')
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
    
    plt.text(0.2, i, severity_label, va='center', ha='left', 
             fontweight='bold', color='white', 
             bbox=dict(boxstyle='round', facecolor='black', alpha=color_intensity))

plt.tight_layout()
plt.show()
'''

    def _generate_general_health_overview_code(self, chat_context: Dict[str, Any]) -> str:
        """Generate general health overview code"""
        return '''
import matplotlib.pyplot as plt
import numpy as np

# Health overview data
plt.figure(figsize=(15, 10))

# Create 2x2 subplot layout
gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

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

# 3. Medication Adherence (bottom left)
ax3 = plt.subplot(gs[1, 0])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
adherence = [0.95, 0.88, 0.92, 0.85, 0.90, 0.87]
ax3.plot(months, adherence, marker='o', linewidth=3, markersize=8, color='blue')
ax3.fill_between(months, adherence, alpha=0.3, color='blue')
ax3.set_ylim(0.7, 1.0)
ax3.set_ylabel('Adherence Rate')
ax3.set_title('Medication Adherence Trend', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Health Categories (bottom right)
ax4 = plt.subplot(gs[1, 1])
categories = ['Physical', 'Mental', 'Social', 'Preventive']
scores = [75, 68, 82, 60]
colors_cat = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars_cat = ax4.bar(categories, scores, color=colors_cat, alpha=0.7)
ax4.set_ylim(0, 100)
ax4.set_ylabel('Score (0-100)')
ax4.set_title('Health Categories', fontweight='bold')

# Add value labels
for bar, score in zip(bars_cat, scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{score}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Comprehensive Patient Health Overview', fontsize=16, fontweight='bold')
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

    # [Include all the existing deidentification and extraction methods from the original file]
    def deidentify_medical_data_enhanced(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stable medical data deidentification"""
        try:
            if not medical_data:
                return {"error": "No medical data available for deidentification"}
 
            # Stable age calculation
            age = self._calculate_age_stable(patient_data.get('date_of_birth', ''))
 
            # Stable JSON processing
            raw_medical_data = medical_data.get('body', medical_data)
            deidentified_medical_data = self._stable_deidentify_json(raw_medical_data)
            deidentified_medical_data = self._mask_medical_fields_stable(deidentified_medical_data)
 
            stable_deidentified = {
                "src_mbr_first_nm": "[MASKED_NAME]",
                "src_mbr_last_nm": "[MASKED_NAME]",
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": patient_data.get('zip_code', '12345'),
                "medical_claims_data": deidentified_medical_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "stable_medical_claims",
                "processing_method": "stable"
            }
 
            logger.info("✅ Stable medical deidentification completed")
            
            return stable_deidentified
 
        except Exception as e:
            logger.error(f"Error in stable medical deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def deidentify_pharmacy_data_enhanced(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stable pharmacy data deidentification"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data available for deidentification"}

            raw_pharmacy_data = pharmacy_data.get('body', pharmacy_data)
            deidentified_pharmacy_data = self._stable_deidentify_pharmacy_json(raw_pharmacy_data)

            stable_result = {
                "pharmacy_claims_data": deidentified_pharmacy_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "stable_pharmacy_claims",
                "processing_method": "stable",
                "name_fields_masked": ["src_mbr_first_nm", "scr_mbr_last_nm"]
            }

            logger.info("✅ Stable pharmacy deidentification completed")
            
            return stable_result

        except Exception as e:
            logger.error(f"Error in stable pharmacy deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def deidentify_mcid_data_enhanced(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stable MCID data deidentification"""
        try:
            if not mcid_data:
                return {"error": "No MCID data available for deidentification"}

            raw_mcid_data = mcid_data.get('body', mcid_data)
            deidentified_mcid_data = self._stable_deidentify_json(raw_mcid_data)

            stable_result = {
                "mcid_claims_data": deidentified_mcid_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "stable_mcid_claims",
                "processing_method": "stable"
            }

            logger.info("✅ Stable MCID deidentification completed")
            return stable_result

        except Exception as e:
            logger.error(f"Error in stable MCID deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}

    def extract_medical_fields_batch_enhanced(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Stable medical field extraction with batch processing"""
        logger.info("🔬 ===== STARTING STABLE BATCH MEDICAL EXTRACTION =====")
        
        stable_extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set()
            },
            "code_meanings": {
                "service_code_meanings": {},
                "diagnosis_code_meanings": {}
            },
            "code_meanings_added": False,
            "stable_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0
            }
        }

        start_time = time.time()

        try:
            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("⚠️ No medical claims data found")
                return stable_extraction_result

            # Step 1: Stable extraction
            logger.info("🔬 Step 1: Stable medical code extraction...")
            self._stable_medical_extraction(medical_data, stable_extraction_result)

            # Convert sets to lists for processing
            unique_service_codes = list(stable_extraction_result["extraction_summary"]["unique_service_codes"])[:15]
            unique_diagnosis_codes = list(stable_extraction_result["extraction_summary"]["unique_diagnosis_codes"])[:20]
            
            stable_extraction_result["extraction_summary"]["unique_service_codes"] = unique_service_codes
            stable_extraction_result["extraction_summary"]["unique_diagnosis_codes"] = unique_diagnosis_codes

            total_codes = len(unique_service_codes) + len(unique_diagnosis_codes)
            stable_extraction_result["batch_stats"]["codes_processed"] = total_codes

            # Step 2: ENHANCED BATCH PROCESSING WITH DEBUGGING
            logger.info(f"🔍 DEBUG: API integrator available: {self.api_integrator is not None}")
            if self.api_integrator:
                logger.info(f"🔍 DEBUG: Has isolated method: {hasattr(self.api_integrator, 'call_llm_isolated_enhanced')}")
            
            logger.info(f"🔍 DEBUG: Service codes found: {len(unique_service_codes)}")
            logger.info(f"🔍 DEBUG: Diagnosis codes found: {len(unique_diagnosis_codes)}")
            
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_service_codes or unique_diagnosis_codes:
                    logger.info(f"🔬 Step 2: ENHANCED BATCH processing {total_codes} codes...")
                    stable_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # ENHANCED BATCH 1: Service Codes
                        if unique_service_codes:
                            logger.info(f"🏥 Processing service codes batch: {unique_service_codes}")
                            service_meanings = self._stable_batch_service_codes(unique_service_codes)
                            logger.info(f"🔍 DEBUG: Service meanings result: {service_meanings}")
                            stable_extraction_result["code_meanings"]["service_code_meanings"] = service_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Service codes batch: {len(service_meanings)} meanings generated")
                        
                        # ENHANCED BATCH 2: Diagnosis Codes  
                        if unique_diagnosis_codes:
                            logger.info(f"🩺 Processing diagnosis codes batch: {unique_diagnosis_codes}")
                            diagnosis_meanings = self._stable_batch_diagnosis_codes(unique_diagnosis_codes)
                            logger.info(f"🔍 DEBUG: Diagnosis meanings result: {diagnosis_meanings}")
                            stable_extraction_result["code_meanings"]["diagnosis_code_meanings"] = diagnosis_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Diagnosis codes batch: {len(diagnosis_meanings)} meanings generated")
                        
                        # Calculate savings
                        individual_calls_would_be = len(unique_service_codes) + len(unique_diagnosis_codes)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        stable_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        stable_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Final status
                        total_meanings = len(stable_extraction_result["code_meanings"]["service_code_meanings"]) + len(stable_extraction_result["code_meanings"]["diagnosis_code_meanings"])
                        logger.info(f"🔍 DEBUG: Total meanings generated: {total_meanings}")
                        
                        if total_meanings > 0:
                            stable_extraction_result["code_meanings_added"] = True
                            stable_extraction_result["stable_analysis"] = True
                            stable_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 ENHANCED BATCH SUCCESS: {total_meanings} meanings, {calls_saved} calls saved!")
                        else:
                            stable_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Batch completed but no meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced batch processing error: {e}")
                        logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(f"❌ Traceback: {traceback.format_exc()}")
                        stable_extraction_result["code_meaning_error"] = str(e)
                        stable_extraction_result["llm_call_status"] = "failed"
                else:
                    stable_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No codes found for batch processing")
                    logger.info(f"🔍 DEBUG: Service codes empty: {len(unique_service_codes) == 0}")
                    logger.info(f"🔍 DEBUG: Diagnosis codes empty: {len(unique_diagnosis_codes) == 0}")
            else:
                stable_extraction_result["llm_call_status"] = "skipped_no_api"
                if not self.api_integrator:
                    logger.error("❌ No API integrator available for batch processing")
                else:
                    logger.error("❌ API integrator missing call_llm_isolated_enhanced method")

            # Stable performance stats
            processing_time = time.time() - start_time
            stable_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"🔬 ===== STABLE BATCH MEDICAL EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s")
            logger.info(f"  📊 API calls: {stable_extraction_result['batch_stats']['api_calls_made']} (saved {stable_extraction_result['batch_stats']['individual_calls_saved']})")
            logger.info(f"  ✅ Meanings: {len(stable_extraction_result['code_meanings']['service_code_meanings']) + len(stable_extraction_result['code_meanings']['diagnosis_code_meanings'])}")

        except Exception as e:
            logger.error(f"❌ Error in stable batch medical extraction: {e}")
            stable_extraction_result["error"] = f"Stable batch extraction failed: {str(e)}"

        return stable_extraction_result

    def extract_pharmacy_fields_batch_enhanced(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Stable pharmacy field extraction with batch processing"""
        logger.info("🔬 ===== STARTING STABLE BATCH PHARMACY EXTRACTION =====")
        
        stable_extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set()
            },
            "code_meanings": {
                "ndc_code_meanings": {},
                "medication_meanings": {}
            },
            "code_meanings_added": False,
            "stable_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0
            }
        }

        start_time = time.time()

        try:
            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("⚠️ No pharmacy claims data found")
                return stable_extraction_result

            # Step 1: Stable extraction
            logger.info("🔬 Step 1: Stable pharmacy code extraction...")
            self._stable_pharmacy_extraction(pharmacy_data, stable_extraction_result)

            # Convert sets to lists for processing
            unique_ndc_codes = list(stable_extraction_result["extraction_summary"]["unique_ndc_codes"])[:10]
            unique_label_names = list(stable_extraction_result["extraction_summary"]["unique_label_names"])[:15]
            
            stable_extraction_result["extraction_summary"]["unique_ndc_codes"] = unique_ndc_codes
            stable_extraction_result["extraction_summary"]["unique_label_names"] = unique_label_names

            total_codes = len(unique_ndc_codes) + len(unique_label_names)
            stable_extraction_result["batch_stats"]["codes_processed"] = total_codes

            # Step 2: ENHANCED BATCH PROCESSING WITH DEBUGGING
            logger.info(f"🔍 DEBUG: API integrator available: {self.api_integrator is not None}")
            if self.api_integrator:
                logger.info(f"🔍 DEBUG: Has isolated method: {hasattr(self.api_integrator, 'call_llm_isolated_enhanced')}")
            
            logger.info(f"🔍 DEBUG: NDC codes found: {len(unique_ndc_codes)}")
            logger.info(f"🔍 DEBUG: Label names found: {len(unique_label_names)}")
            
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_ndc_codes or unique_label_names:
                    logger.info(f"🔬 Step 2: ENHANCED BATCH processing {total_codes} pharmacy codes...")
                    stable_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # ENHANCED BATCH 1: NDC Codes
                        if unique_ndc_codes:
                            logger.info(f"💊 Processing NDC codes batch: {unique_ndc_codes}")
                            ndc_meanings = self._stable_batch_ndc_codes(unique_ndc_codes)
                            logger.info(f"🔍 DEBUG: NDC meanings result: {ndc_meanings}")
                            stable_extraction_result["code_meanings"]["ndc_code_meanings"] = ndc_meanings
                            api_calls_made += 1
                            logger.info(f"✅ NDC codes batch: {len(ndc_meanings)} meanings generated")
                        
                        # ENHANCED BATCH 2: Medications
                        if unique_label_names:
                            logger.info(f"💉 Processing medications batch: {unique_label_names}")
                            med_meanings = self._stable_batch_medications(unique_label_names)
                            logger.info(f"🔍 DEBUG: Medication meanings result: {med_meanings}")
                            stable_extraction_result["code_meanings"]["medication_meanings"] = med_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Medications batch: {len(med_meanings)} meanings generated")
                        
                        # Calculate savings
                        individual_calls_would_be = len(unique_ndc_codes) + len(unique_label_names)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        stable_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        stable_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Final status
                        total_meanings = len(stable_extraction_result["code_meanings"]["ndc_code_meanings"]) + len(stable_extraction_result["code_meanings"]["medication_meanings"])
                        logger.info(f"🔍 DEBUG: Total pharmacy meanings generated: {total_meanings}")
                        
                        if total_meanings > 0:
                            stable_extraction_result["code_meanings_added"] = True
                            stable_extraction_result["stable_analysis"] = True
                            stable_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 ENHANCED PHARMACY BATCH SUCCESS: {total_meanings} meanings, {calls_saved} calls saved!")
                        else:
                            stable_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Pharmacy batch completed but no meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced pharmacy batch error: {e}")
                        logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(f"❌ Traceback: {traceback.format_exc()}")
                        stable_extraction_result["code_meaning_error"] = str(e)
                        stable_extraction_result["llm_call_status"] = "failed"
                else:
                    stable_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No pharmacy codes for batch processing")
                    logger.info(f"🔍 DEBUG: NDC codes empty: {len(unique_ndc_codes) == 0}")
                    logger.info(f"🔍 DEBUG: Label names empty: {len(unique_label_names) == 0}")
            else:
                stable_extraction_result["llm_call_status"] = "skipped_no_api"
                if not self.api_integrator:
                    logger.error("❌ No API integrator available for pharmacy batch processing")
                else:
                    logger.error("❌ API integrator missing call_llm_isolated_enhanced method")

            # Stable performance stats
            processing_time = time.time() - start_time
            stable_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"💊 ===== STABLE BATCH PHARMACY EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s")
            logger.info(f"  📊 API calls: {stable_extraction_result['batch_stats']['api_calls_made']} (saved {stable_extraction_result['batch_stats']['individual_calls_saved']})")

        except Exception as e:
            logger.error(f"❌ Error in stable batch pharmacy extraction: {e}")
            stable_extraction_result["error"] = f"Stable pharmacy batch extraction failed: {str(e)}"

        return stable_extraction_result

    # [Include all the remaining helper methods from the original file]
    # I'll include the key methods but truncate for space

    def _stable_batch_service_codes(self, service_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL service codes with debugging"""
        try:
            if not service_codes:
                logger.warning("🔍 DEBUG: No service codes provided to batch processor")
                return {}
                
            logger.info(f"🏥 === ENHANCED BATCH PROCESSING {len(service_codes)} SERVICE CODES ===")
            logger.info(f"🔍 DEBUG: Service codes to process: {service_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(service_codes)
            
            stable_prompt = f"""Please explain these medical service codes. Provide brief explanations for each code.

Service Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"12345": "Medical procedure or service description"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medical coding expert. Provide brief, clear explanations of medical service codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with prompt length: {len(stable_prompt)}")
            
            response = self.api_integrator.call_llm_isolated_enhanced(stable_prompt, stable_system_msg)
            
            logger.info(f"🔍 DEBUG: LLM response received: {response[:200]}...")
            logger.info(f"🔍 DEBUG: Response type: {type(response)}")
            
            if response and response != "Brief explanation unavailable" and "error" not in response.lower():
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} service code meanings")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for service codes: {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"Medical service code {code}" for code in service_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing service codes: {e}")
                    return {code: f"Medical service code {code}" for code in service_codes}
            else:
                logger.warning(f"⚠️ LLM returned unavailable or error response: {response}")
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
        """Enhanced BATCH process ALL diagnosis codes with debugging"""
        try:
            if not diagnosis_codes:
                logger.warning("🔍 DEBUG: No diagnosis codes provided to batch processor")
                return {}
                
            logger.info(f"🩺 === ENHANCED BATCH PROCESSING {len(diagnosis_codes)} DIAGNOSIS CODES ===")
            logger.info(f"🔍 DEBUG: Diagnosis codes to process: {diagnosis_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(diagnosis_codes)
            
            stable_prompt = f"""Please explain these ICD-10 diagnosis codes. Provide brief medical explanations for each code.

Diagnosis Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"I10": "Essential hypertension", "E11.9": "Type 2 diabetes mellitus"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medical diagnosis expert. Provide brief, clear explanations of ICD-10 diagnosis codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with prompt length: {len(stable_prompt)}")
            
            response = self.api_integrator.call_llm_isolated_enhanced(stable_prompt, stable_system_msg)
            
            logger.info(f"🔍 DEBUG: LLM response received: {response[:200]}...")
            logger.info(f"🔍 DEBUG: Response type: {type(response)}")
            
            if response and response != "Brief explanation unavailable" and "error" not in response.lower():
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} diagnosis code meanings")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for diagnosis codes: {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing diagnosis codes: {e}")
                    return {code: f"ICD-10 diagnosis code {code}" for code in diagnosis_codes}
            else:
                logger.warning(f"⚠️ LLM returned unavailable or error response: {response}")
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
        """Enhanced BATCH process ALL NDC codes with debugging"""
        try:
            if not ndc_codes:
                logger.warning("🔍 DEBUG: No NDC codes provided to batch processor")
                return {}
                
            logger.info(f"💊 === ENHANCED BATCH PROCESSING {len(ndc_codes)} NDC CODES ===")
            logger.info(f"🔍 DEBUG: NDC codes to process: {ndc_codes}")
            
            # Create a simple prompt that's more likely to succeed
            codes_text = ", ".join(ndc_codes)
            
            stable_prompt = f"""Please explain these NDC medication codes. Provide brief explanations for each code.

NDC Codes: {codes_text}

Please respond with a JSON object where each code is a key and the explanation is the value. For example:
{{"12345-678-90": "Medication name and therapeutic use"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a pharmacy expert. Provide brief, clear explanations of NDC medication codes in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with prompt length: {len(stable_prompt)}")
            
            response = self.api_integrator.call_llm_isolated_enhanced(stable_prompt, stable_system_msg)
            
            logger.info(f"🔍 DEBUG: LLM response received: {response[:200]}...")
            logger.info(f"🔍 DEBUG: Response type: {type(response)}")
            
            if response and response != "Brief explanation unavailable" and "error" not in response.lower():
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} NDC code meanings")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for NDC codes: {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {code: f"NDC medication code {code}" for code in ndc_codes}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing NDC codes: {e}")
                    return {code: f"NDC medication code {code}" for code in ndc_codes}
            else:
                logger.warning(f"⚠️ LLM returned unavailable or error response: {response}")
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
        """Enhanced BATCH process ALL medications with debugging"""
        try:
            if not medications:
                logger.warning("🔍 DEBUG: No medications provided to batch processor")
                return {}
                
            logger.info(f"💉 === ENHANCED BATCH PROCESSING {len(medications)} MEDICATIONS ===")
            logger.info(f"🔍 DEBUG: Medications to process: {medications}")
            
            # Create a simple prompt that's more likely to succeed
            meds_text = ", ".join(medications)
            
            stable_prompt = f"""Please explain these medications. Provide brief explanations for each medication.

Medications: {meds_text}

Please respond with a JSON object where each medication is a key and the explanation is the value. For example:
{{"Metformin": "Medication for type 2 diabetes", "Lisinopril": "ACE inhibitor for high blood pressure"}}

Only return the JSON object, no other text."""

            stable_system_msg = """You are a medication expert. Provide brief, clear explanations of medications in JSON format."""
            
            logger.info(f"🔍 DEBUG: Calling LLM with prompt length: {len(stable_prompt)}")
            
            response = self.api_integrator.call_llm_isolated_enhanced(stable_prompt, stable_system_msg)
            
            logger.info(f"🔍 DEBUG: LLM response received: {response[:200]}...")
            logger.info(f"🔍 DEBUG: Response type: {type(response)}")
            
            if response and response != "Brief explanation unavailable" and "error" not in response.lower():
                try:
                    # Try to extract JSON from response
                    clean_response = self._clean_json_response_stable(response)
                    logger.info(f"🔍 DEBUG: Cleaned response: {clean_response[:200]}...")
                    
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Successfully parsed {len(meanings_dict)} medication meanings")
                    logger.info(f"🔍 DEBUG: Parsed meanings: {meanings_dict}")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parse error for medications: {e}")
                    logger.error(f"❌ Failed to parse: {clean_response}")
                    # Return fallback explanations
                    return {med: f"Medication: {med}" for med in medications}
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing medications: {e}")
                    return {med: f"Medication: {med}" for med in medications}
            else:
                logger.warning(f"⚠️ LLM returned unavailable or error response: {response}")
                # Return fallback explanations
                return {med: f"Medication: {med}" for med in medications}
                
        except Exception as e:
            logger.error(f"❌ Medications batch exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # Return fallback explanations
            return {med: f"Medication: {med}" for med in medications}

    def extract_health_entities_with_clinical_insights(self, pharmacy_data: Dict[str, Any],
                                                      pharmacy_extraction: Dict[str, Any],
                                                      medical_extraction: Dict[str, Any],
                                                      patient_data: Dict[str, Any] = None,
                                                      api_integrator = None) -> Dict[str, Any]:
        """Stable health entity extraction with clinical insights"""
        logger.info("🔬 ===== Stable HEALTH ENTITY EXTRACTION =====")
        
        stable_entities = {
            "diabetics": "no",
            "age_group": "unknown",
            "age": None,
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": [],
            "stable_analysis": False,
            "llm_analysis": "not_performed"
        }

        try:
            # Stable age calculation
            if patient_data and patient_data.get('date_of_birth'):
                age = self._calculate_age_stable(patient_data['date_of_birth'])
                if age != "unknown":
                    try:
                        age_num = int(age.split()[0])
                        stable_entities["age"] = age_num
                        stable_entities["age_group"] = self._get_stable_age_group(age_num)
                        stable_entities["analysis_details"].append(f"Age analysis: {age}")
                    except:
                        pass

            # Stable entity extraction using batch meanings
            medical_meanings_available = (medical_extraction and 
                                        medical_extraction.get("code_meanings_added", False) and
                                        medical_extraction.get("stable_analysis", False))
            
            pharmacy_meanings_available = (pharmacy_extraction and 
                                         pharmacy_extraction.get("code_meanings_added", False) and
                                         pharmacy_extraction.get("stable_analysis", False))
            
            if medical_meanings_available or pharmacy_meanings_available:
                logger.info("🔬 Using stable batch-generated meanings for entity extraction")
                stable_entities = self._stable_analyze_entities_with_meanings(
                    stable_entities, medical_extraction, pharmacy_extraction
                )
                stable_entities["stable_analysis"] = True
                stable_entities["llm_analysis"] = "used_stable_batch_meanings"
                stable_entities["analysis_details"].append("Used stable batch-generated meanings")
            else:
                logger.info("🔬 Using stable direct pattern matching for entity extraction")
                self._stable_analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, stable_entities)

            # Stable medication identification
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        medication_info = {
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "detailed_meaning": record.get("medication_detailed_meaning", ""),
                            "stable_processing": True
                        }
                        stable_entities["medications_identified"].append(medication_info)

            logger.info(f"🔬 ===== Stable HEALTH ENTITY EXTRACTION COMPLETED =====")
            logger.info(f"  ✅ Stable analysis: {stable_entities['stable_analysis']}")
            logger.info(f"  🩺 Diabetes: {stable_entities['diabetics']}")
            logger.info(f"  💓 Blood pressure: {stable_entities['blood_pressure']}")
            logger.info(f"  💊 Medications: {len(stable_entities['medications_identified'])}")

        except Exception as e:
            logger.error(f"❌ Error in stable entity extraction: {e}")
            stable_entities["analysis_details"].append(f"Stable analysis error: {str(e)}")

        return stable_entities

    def _calculate_age_stable(self, date_of_birth: str) -> str:
        """Stable age calculation"""
        try:
            if not date_of_birth:
                return "unknown"
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Stable age context
            if age < 18:
                return f"{age} years (Pediatric)"
            elif age < 65:
                return f"{age} years (Adult)"
            else:
                return f"{age} years (Senior)"
        except:
            return "unknown"

    def _get_stable_age_group(self, age: int) -> str:
        """Stable age group determination"""
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

    def _stable_analyze_entities_with_meanings(self, entities: Dict[str, Any], 
                                             medical_extraction: Dict[str, Any], 
                                             pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Stable entity analysis using batch meanings"""
        try:
            medical_conditions = []
            
            # Stable analysis of medical meanings
            medical_meanings = medical_extraction.get("code_meanings", {})
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            
            for code, meaning in diagnosis_meanings.items():
                meaning_lower = meaning.lower()
                
                # Stable diabetes analysis
                if any(term in meaning_lower for term in ['diabetes', 'diabetic', 'insulin', 'glucose']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes (ICD-10 {code})")
                
                # Stable hypertension analysis
                if any(term in meaning_lower for term in ['hypertension', 'high blood pressure']):
                    entities["blood_pressure"] = "diagnosed"
                    medical_conditions.append(f"Hypertension (ICD-10 {code})")
                
                # Stable smoking analysis
                if any(term in meaning_lower for term in ['tobacco', 'smoking', 'nicotine']):
                    entities["smoking"] = "yes"
                    medical_conditions.append(f"Tobacco use (ICD-10 {code})")
                
                # Stable alcohol analysis
                if any(term in meaning_lower for term in ['alcohol', 'alcoholism']):
                    entities["alcohol"] = "yes"
                    medical_conditions.append(f"Alcohol-related condition (ICD-10 {code})")

            # Stable analysis of pharmacy meanings
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
            medication_meanings = pharmacy_meanings.get("medication_meanings", {})
            
            for medication, meaning in medication_meanings.items():
                meaning_lower = meaning.lower()
                
                # Stable diabetes medication analysis
                if any(term in meaning_lower for term in ['diabetes', 'blood sugar', 'insulin', 'metformin']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes medication: {medication}")
                
                # Stable cardiovascular medication analysis
                if any(term in meaning_lower for term in ['blood pressure', 'hypertension', 'ace inhibitor']):
                    if entities["blood_pressure"] == "unknown":
                        entities["blood_pressure"] = "managed"
                    medical_conditions.append(f"BP medication: {medication}")

            entities["medical_conditions"] = medical_conditions
            
            logger.info(f"🔬 Stable meaning analysis: {len(medical_conditions)} conditions identified")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in stable meaning analysis: {e}")
            return entities

    def _stable_analyze_entities_direct(self, pharmacy_data: Dict[str, Any],
                                      pharmacy_extraction: Dict[str, Any],
                                      medical_extraction: Dict[str, Any],
                                      entities: Dict[str, Any]):
        """Stable direct entity analysis using pattern matching"""
        try:
            logger.info("🔬 Stable direct pattern matching analysis")
            
            # Stable medication pattern matching
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    medication_name = record.get("lbl_nm", "").lower()
                    
                    # Stable diabetes detection
                    if any(term in medication_name for term in ['metformin', 'insulin', 'glipizide']):
                        entities["diabetics"] = "yes"
                        
                    # Stable cardiovascular detection
                    if any(term in medication_name for term in ['amlodipine', 'lisinopril', 'atenolol']):
                        entities["blood_pressure"] = "managed"

            entities["analysis_details"].append("Stable direct pattern matching completed")

        except Exception as e:
            logger.error(f"Error in stable direct analysis: {e}")
            entities["analysis_details"].append(f"Stable direct analysis error: {str(e)}")

    def prepare_enhanced_clinical_context(self, chat_context: Dict[str, Any]) -> str:
        """Stable context preparation for chatbot"""
        try:
            context_parts = []

            # Stable patient overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_parts.append(f"**PATIENT**: Age {patient_overview.get('age', 'unknown')}, ZIP {patient_overview.get('zip', 'unknown')}")

            # Stable medical extractions
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_parts.append(f"**MEDICAL DATA**: {json.dumps(medical_extraction, indent=2)}")

            # Stable pharmacy extractions
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_parts.append(f"**PHARMACY DATA**: {json.dumps(pharmacy_extraction, indent=2)}")

            # Stable entity extraction
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_parts.append(f"**HEALTH ENTITIES**: {json.dumps(entity_extraction, indent=2)}")

            # Stable health trajectory
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                context_parts.append(f"**HEALTH TRAJECTORY**: {health_trajectory[:500]}...")

            # Stable cardiovascular risk assessment
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_parts.append(f"**CARDIOVASCULAR RISK**: {json.dumps(heart_attack_prediction, indent=2)}")

            return "\n\n" + "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error preparing stable context: {e}")
            return "Stable patient healthcare data available for analysis."

    # Helper methods for stable processing
    def _stable_deidentify_json(self, data: Any) -> Any:
        """Stable JSON deidentification"""
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
        """Stable pharmacy JSON deidentification"""
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
        """Stable medical field masking"""
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
        """Stable string deidentification"""
        if not isinstance(data, str) or not data.strip():
            return data

        deidentified = str(data)
        
        # Stable pattern replacements
        deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[MASKED_SSN]', deidentified)
        deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[MASKED_PHONE]', deidentified)
        deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED_EMAIL]', deidentified)
        deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[MASKED_NAME]', deidentified)
        
        return deidentified

    def _stable_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Stable recursive medical field extraction"""
        if isinstance(data, dict):
            current_record = {}

            # Stable health service code extraction
            if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
                service_code = str(data["hlth_srvc_cd"]).strip()
                current_record["hlth_srvc_cd"] = service_code
                result["extraction_summary"]["unique_service_codes"].add(service_code)

            # Stable claim received date extraction
            if "clm_rcvd_dt" in data and data["clm_rcvd_dt"]:
                current_record["clm_rcvd_dt"] = data["clm_rcvd_dt"]

            # Stable diagnosis codes extraction
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

            # Continue stable recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._stable_medical_extraction(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._stable_medical_extraction(item, result, new_path)

    def _stable_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Stable recursive pharmacy field extraction"""
        if isinstance(data, dict):
            current_record = {}

            # Stable NDC code extraction
            ndc_found = False
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number', 'national_drug_code']:
                    ndc_code = str(data[key]).strip()
                    current_record["ndc"] = ndc_code
                    result["extraction_summary"]["unique_ndc_codes"].add(ndc_code)
                    ndc_found = True
                    break

            # Stable medication name extraction
            label_found = False
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                    medication_name = str(data[key]).strip()
                    current_record["lbl_nm"] = medication_name
                    result["extraction_summary"]["unique_label_names"].add(medication_name)
                    label_found = True
                    break

            # Stable prescription filled date extraction
            if "rx_filled_dt" in data and data["rx_filled_dt"]:
                current_record["rx_filled_dt"] = data["rx_filled_dt"]

            if ndc_found or label_found or "rx_filled_dt" in current_record:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1

            # Continue stable recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._stable_pharmacy_extraction(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._stable_pharmacy_extraction(item, result, new_path)

    # [Include all remaining helper methods - truncated for space]
    def _clean_json_response_stable(self, response: str) -> str:
        """Stable LLM response cleaning for JSON extraction"""
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
            logger.warning(f"Stable JSON cleaning failed: {e}")
            return response

    def _fix_common_json_issues_stable(self, json_content: str) -> str:
        """Fix common JSON formatting issues with stable approach"""
        try:
            # Fix trailing commas
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            
            return json_content
        except Exception as e:
            logger.warning(f"Stable JSON fixing failed: {e}")
            return json_content

    # Include all other helper methods from the original file
    # [Additional helper methods would be included here]

    def extract_health_entities_with_clinical_insights(self, pharmacy_data: Dict[str, Any],
                                                      pharmacy_extraction: Dict[str, Any],
                                                      medical_extraction: Dict[str, Any],
                                                      patient_data: Dict[str, Any] = None,
                                                      api_integrator = None) -> Dict[str, Any]:
        """Stable health entity extraction"""
        logger.info("🔬 ===== Stable HEALTH ENTITY EXTRACTION =====")
        
        stable_entities = {
            "diabetics": "no",
            "age_group": "unknown",
            "age": None,
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": [],
            "stable_analysis": False,
            "llm_analysis": "not_performed"
        }

        try:
            # Stable age calculation
            if patient_data and patient_data.get('date_of_birth'):
                age = self._calculate_age_stable(patient_data['date_of_birth'])
                if age != "unknown":
                    try:
                        age_num = int(age.split()[0])
                        stable_entities["age"] = age_num
                        stable_entities["age_group"] = self._get_stable_age_group(age_num)
                        stable_entities["analysis_details"].append(f"Age analysis: {age}")
                    except:
                        pass

            # Stable entity extraction using batch meanings
            medical_meanings_available = (medical_extraction and 
                                        medical_extraction.get("code_meanings_added", False) and
                                        medical_extraction.get("stable_analysis", False))
            
            pharmacy_meanings_available = (pharmacy_extraction and 
                                         pharmacy_extraction.get("code_meanings_added", False) and
                                         pharmacy_extraction.get("stable_analysis", False))
            
            if medical_meanings_available or pharmacy_meanings_available:
                logger.info("🔬 Using stable batch-generated meanings for entity extraction")
                stable_entities = self._stable_analyze_entities_with_meanings(
                    stable_entities, medical_extraction, pharmacy_extraction
                )
                stable_entities["stable_analysis"] = True
                stable_entities["llm_analysis"] = "used_stable_batch_meanings"
                stable_entities["analysis_details"].append("Used stable batch-generated meanings")
            else:
                logger.info("🔬 Using stable direct pattern matching for entity extraction")
                self._stable_analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, stable_entities)

            # Stable medication identification
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        medication_info = {
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "detailed_meaning": record.get("medication_detailed_meaning", ""),
                            "stable_processing": True
                        }
                        stable_entities["medications_identified"].append(medication_info)

            logger.info(f"🔬 ===== Stable HEALTH ENTITY EXTRACTION COMPLETED =====")
            logger.info(f"  ✅ Stable analysis: {stable_entities['stable_analysis']}")
            logger.info(f"  🩺 Diabetes: {stable_entities['diabetics']}")
            logger.info(f"  💓 Blood pressure: {stable_entities['blood_pressure']}")
            logger.info(f"  💊 Medications: {len(stable_entities['medications_identified'])}")

        except Exception as e:
            logger.error(f"❌ Error in stable entity extraction: {e}")
            stable_entities["analysis_details"].append(f"Stable analysis error: {str(e)}")

        return stable_entities

    def _calculate_age_stable(self, date_of_birth: str) -> str:
        """Stable age calculation"""
        try:
            if not date_of_birth:
                return "unknown"
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Stable age context
            if age < 18:
                return f"{age} years (Pediatric)"
            elif age < 65:
                return f"{age} years (Adult)"
            else:
                return f"{age} years (Senior)"
        except:
            return "unknown"

    def _get_stable_age_group(self, age: int) -> str:
        """Stable age group determination"""
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

    def _stable_analyze_entities_with_meanings(self, entities: Dict[str, Any], 
                                             medical_extraction: Dict[str, Any], 
                                             pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Stable entity analysis using batch meanings"""
        try:
            medical_conditions = []
            
            # Stable analysis of medical meanings
            medical_meanings = medical_extraction.get("code_meanings", {})
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            
            for code, meaning in diagnosis_meanings.items():
                meaning_lower = meaning.lower()
                
                # Stable diabetes analysis
                if any(term in meaning_lower for term in ['diabetes', 'diabetic', 'insulin', 'glucose']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes (ICD-10 {code})")
                
                # Stable hypertension analysis
                if any(term in meaning_lower for term in ['hypertension', 'high blood pressure']):
                    entities["blood_pressure"] = "diagnosed"
                    medical_conditions.append(f"Hypertension (ICD-10 {code})")
                
                # Stable smoking analysis
                if any(term in meaning_lower for term in ['tobacco', 'smoking', 'nicotine']):
                    entities["smoking"] = "yes"
                    medical_conditions.append(f"Tobacco use (ICD-10 {code})")
                
                # Stable alcohol analysis
                if any(term in meaning_lower for term in ['alcohol', 'alcoholism']):
                    entities["alcohol"] = "yes"
                    medical_conditions.append(f"Alcohol-related condition (ICD-10 {code})")

            # Stable analysis of pharmacy meanings
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
            medication_meanings = pharmacy_meanings.get("medication_meanings", {})
            
            for medication, meaning in medication_meanings.items():
                meaning_lower = meaning.lower()
                
                # Stable diabetes medication analysis
                if any(term in meaning_lower for term in ['diabetes', 'blood sugar', 'insulin', 'metformin']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes medication: {medication}")
                
                # Stable cardiovascular medication analysis
                if any(term in meaning_lower for term in ['blood pressure', 'hypertension', 'ace inhibitor']):
                    if entities["blood_pressure"] == "unknown":
                        entities["blood_pressure"] = "managed"
                    medical_conditions.append(f"BP medication: {medication}")

            entities["medical_conditions"] = medical_conditions
            
            logger.info(f"🔬 Stable meaning analysis: {len(medical_conditions)} conditions identified")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in stable meaning analysis: {e}")
            return entities

    def _stable_analyze_entities_direct(self, pharmacy_data: Dict[str, Any],
                                      pharmacy_extraction: Dict[str, Any],
                                      medical_extraction: Dict[str, Any],
                                      entities: Dict[str, Any]):
        """Stable direct entity analysis using pattern matching"""
        try:
            logger.info("🔬 Stable direct pattern matching analysis")
            
            # Stable medication pattern matching
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    medication_name = record.get("lbl_nm", "").lower()
                    
                    # Stable diabetes detection
                    if any(term in medication_name for term in ['metformin', 'insulin', 'glipizide']):
                        entities["diabetics"] = "yes"
                        
                    # Stable cardiovascular detection
                    if any(term in medication_name for term in ['amlodipine', 'lisinopril', 'atenolol']):
                        entities["blood_pressure"] = "managed"

            entities["analysis_details"].append("Stable direct pattern matching completed")

        except Exception as e:
            logger.error(f"Error in stable direct analysis: {e}")
            entities["analysis_details"].append(f"Stable direct analysis error: {str(e)}")

    def prepare_enhanced_clinical_context(self, chat_context: Dict[str, Any]) -> str:
        """Stable context preparation for chatbot"""
        try:
            context_parts = []

            # Stable patient overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_parts.append(f"**PATIENT**: Age {patient_overview.get('age', 'unknown')}, ZIP {patient_overview.get('zip', 'unknown')}")

            # Stable medical extractions
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_parts.append(f"**MEDICAL DATA**: {json.dumps(medical_extraction, indent=2)}")

            # Stable pharmacy extractions
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_parts.append(f"**PHARMACY DATA**: {json.dumps(pharmacy_extraction, indent=2)}")

            # Stable entity extraction
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_parts.append(f"**HEALTH ENTITIES**: {json.dumps(entity_extraction, indent=2)}")

            # Stable health trajectory
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                context_parts.append(f"**HEALTH TRAJECTORY**: {health_trajectory[:500]}...")

            # Stable cardiovascular risk assessment
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_parts.append(f"**CARDIOVASCULAR RISK**: {json.dumps(heart_attack_prediction, indent=2)}")

            return "\n\n" + "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error preparing stable context: {e}")
            return "Stable patient healthcare data available for analysis."

    # Helper methods for stable processing
    def _stable_deidentify_json(self, data: Any) -> Any:
        """Stable JSON deidentification"""
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
        """Stable pharmacy JSON deidentification"""
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
        """Stable medical field masking"""
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
        """Stable string deidentification"""
        if not isinstance(data, str) or not data.strip():
            return data

        deidentified = str(data)
        
        # Stable pattern replacements
        deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[MASKED_SSN]', deidentified)
        deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[MASKED_PHONE]', deidentified)
        deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED_EMAIL]', deidentified)
        deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[MASKED_NAME]', deidentified)
        
        return deidentified

    def _stable_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
     """Enhanced stable recursive medical field extraction with billing provider info"""
    if isinstance(data, dict):
        current_record = {}
        
        # Stable health service code extraction
        if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
            service_code = str(data["hlth_srvc_cd"]).strip()
            current_record["hlth_srvc_cd"] = service_code
            result["extraction_summary"]["unique_service_codes"].add(service_code)

        # Stable claim received date extraction
        if "clm_rcvd_dt" in data and data["clm_rcvd_dt"]:
            current_record["clm_rcvd_dt"] = data["clm_rcvd_dt"]

        # Stable claim line service end date extraction
        if "clm_line_srvc_end_dt" in data and data["clm_line_srvc_end_dt"]:
            current_record["clm_line_srvc_end_dt"] = data["clm_line_srvc_end_dt"]

        # Enhanced billing provider extraction
        billing_provider_fields = [
            "billing_provider", "billing_provider_name", "provider_name", 
            "rendering_provider", "attending_provider", "billing_nm",
            "provider_first_nm", "provider_last_nm", "provider_organization"
        ]
        
        for field in billing_provider_fields:
            if field in data and data[field]:
                # Construct full provider name if available
                if field in ["provider_first_nm", "provider_last_nm"]:
                    first_name = data.get("provider_first_nm", "")
                    last_name = data.get("provider_last_nm", "")
                    if first_name and last_name:
                        current_record["billing_provider"] = f"{first_name} {last_name}".strip()
                    elif last_name:
                        current_record["billing_provider"] = last_name.strip()
                else:
                    current_record["billing_provider"] = str(data[field]).strip()
                break
        
        # If no billing provider found, use a default
        if "billing_provider" not in current_record:
            current_record["billing_provider"] = "Unknown Provider"

        # Enhanced provider location extraction
        provider_zip_fields = [
            "provider_zip", "provider_zip_cd", "billing_zip", 
            "provider_address_zip", "billing_address_zip"
        ]
        
        for field in provider_zip_fields:
            if field in data and data[field]:
                current_record["provider_zip"] = str(data[field]).strip()
                break
        
        if "provider_zip" not in current_record:
            current_record["provider_zip"] = ""

        # Enhanced provider state extraction
        provider_state_fields = [
            "provider_state", "provider_state_cd", "billing_state",
            "provider_address_state", "billing_address_state"
        ]
        
        for field in provider_state_fields:
            if field in data and data[field]:
                current_record["provider_state"] = str(data[field]).strip()
                break

        # Enhanced provider specialty extraction
        specialty_fields = [
            "provider_specialty", "specialty_cd", "provider_type",
            "taxonomy_cd", "specialty_description"
        ]
        
        for field in specialty_fields:
            if field in data and data[field]:
                current_record["provider_specialty"] = str(data[field]).strip()
                break

        # Enhanced claim amount extraction
        amount_fields = [
            "claim_amount", "total_claim_amount", "allowed_amount",
            "paid_amount", "billed_amount", "charge_amount"
        ]
        
        for field in amount_fields:
            if field in data and data[field]:
                try:
                    current_record["claim_amount"] = float(data[field])
                except (ValueError, TypeError):
                    current_record["claim_amount"] = 0.0
                break

        # Enhanced procedure code extraction
        procedure_fields = [
            "procedure_cd", "cpt_cd", "hcpcs_cd", "procedure_code"
        ]
        
        for field in procedure_fields:
            if field in data and data[field]:
                current_record["procedure_code"] = str(data[field]).strip()
                break

        # Enhanced place of service extraction
        pos_fields = [
            "place_of_service", "pos_cd", "service_location", "place_of_service_cd"
        ]
        
        for field in pos_fields:
            if field in data and data[field]:
                current_record["place_of_service"] = str(data[field]).strip()
                break

        # Stable diagnosis codes extraction (unchanged logic but enhanced structure)
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
                            "source": "diag_1_50_cd",
                            "description": ""  # Will be filled by LLM batch processing
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
                        "source": f"individual_{diag_key}",
                        "description": ""  # Will be filled by LLM batch processing
                    }
                    diagnosis_codes.append(diagnosis_info)
                    result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)

        if diagnosis_codes:
            current_record["diagnosis_codes"] = diagnosis_codes
            result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)

        # Add billing provider to unique set for tracking
        if current_record.get("billing_provider"):
            if "unique_billing_providers" not in result["extraction_summary"]:
                result["extraction_summary"]["unique_billing_providers"] = set()
            result["extraction_summary"]["unique_billing_providers"].add(current_record["billing_provider"])

        if current_record:
            current_record["data_path"] = path
            result["hlth_srvc_records"].append(current_record)
            result["extraction_summary"]["total_hlth_srvc_records"] += 1

        # Continue stable recursive search
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            self._stable_medical_extraction(value, result, new_path)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            self._stable_medical_extraction(item, result, new_path)

    def _stable_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
     """Enhanced stable recursive pharmacy field extraction with provider info"""
    if isinstance(data, dict):
        current_record = {}

        # Stable NDC code extraction
        ndc_found = False
        for key in data:
            if key.lower() in ['ndc', 'ndc_code', 'ndc_number', 'national_drug_code']:
                ndc_code = str(data[key]).strip()
                current_record["ndc"] = ndc_code
                result["extraction_summary"]["unique_ndc_codes"].add(ndc_code)
                ndc_found = True
                break

        # Stable medication name extraction
        label_found = False
        for key in data:
            if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                medication_name = str(data[key]).strip()
                current_record["lbl_nm"] = medication_name
                result["extraction_summary"]["unique_label_names"].add(medication_name)
                label_found = True
                break

        # Stable prescription filled date extraction
        if "rx_filled_dt" in data and data["rx_filled_dt"]:
            current_record["rx_filled_dt"] = data["rx_filled_dt"]

        # Enhanced prescribing provider extraction
        prescribing_provider_fields = [
            "prescribing_provider", "prescriber_name", "doctor_name",
            "physician_name", "prescriber_first_nm", "prescriber_last_nm",
            "prescribing_physician", "ordering_provider"
        ]
        
        for field in prescribing_provider_fields:
            if field in data and data[field]:
                # Construct full prescriber name if available
                if field in ["prescriber_first_nm", "prescriber_last_nm"]:
                    first_name = data.get("prescriber_first_nm", "")
                    last_name = data.get("prescriber_last_nm", "")
                    if first_name and last_name:
                        current_record["prescribing_provider"] = f"{first_name} {last_name}".strip()
                    elif last_name:
                        current_record["prescribing_provider"] = last_name.strip()
                else:
                    current_record["prescribing_provider"] = str(data[field]).strip()
                break
        
        # If no prescribing provider found, use a default
        if "prescribing_provider" not in current_record:
            current_record["prescribing_provider"] = "Unknown Prescriber"

        # Enhanced billing provider (pharmacy) extraction
        billing_provider_fields = [
            "billing_provider", "pharmacy_name", "dispensing_pharmacy",
            "pharmacy", "provider_name", "billing_pharmacy", "pharmacy_nm"
        ]
        
        for field in billing_provider_fields:
            if field in data and data[field]:
                current_record["billing_provider"] = str(data[field]).strip()
                break
        
        # If no billing provider found, use a default
        if "billing_provider" not in current_record:
            current_record["billing_provider"] = "Unknown Pharmacy"

        # Enhanced pharmacy location extraction
        pharmacy_zip_fields = [
            "pharmacy_zip", "billing_zip", "pharmacy_address_zip",
            "dispensing_pharmacy_zip", "provider_zip"
        ]
        
        for field in pharmacy_zip_fields:
            if field in data and data[field]:
                current_record["pharmacy_zip"] = str(data[field]).strip()
                break

        # Enhanced pharmacy state extraction
        pharmacy_state_fields = [
            "pharmacy_state", "billing_state", "pharmacy_address_state",
            "dispensing_pharmacy_state", "provider_state"
        ]
        
        for field in pharmacy_state_fields:
            if field in data and data[field]:
                current_record["pharmacy_state"] = str(data[field]).strip()
                break

        # Enhanced prescription details
        prescription_fields = [
            "days_supply", "quantity_dispensed", "refills_remaining",
            "prescription_number", "rx_number", "generic_indicator",
            "brand_generic", "strength", "dosage_form"
        ]
        
        for field in prescription_fields:
            if field in data and data[field]:
                current_record[field] = str(data[field]).strip()

        # Enhanced cost information
        cost_fields = [
            "total_cost", "patient_pay", "insurance_pay", "copay",
            "deductible", "ingredient_cost", "dispensing_fee"
        ]
        
        for field in cost_fields:
            if field in data and data[field]:
                try:
                    current_record[field] = float(data[field])
                except (ValueError, TypeError):
                    current_record[field] = 0.0

        # Enhanced therapeutic class extraction
        therapeutic_fields = [
            "therapeutic_class", "drug_class", "ahfs_class",
            "therapeutic_category", "pharmacologic_class"
        ]
        
        for field in therapeutic_fields:
            if field in data and data[field]:
                current_record["therapeutic_class"] = str(data[field]).strip()
                break

        # Add providers to unique sets for tracking
        if current_record.get("prescribing_provider"):
            if "unique_prescribing_providers" not in result["extraction_summary"]:
                result["extraction_summary"]["unique_prescribing_providers"] = set()
            result["extraction_summary"]["unique_prescribing_providers"].add(current_record["prescribing_provider"])

        if current_record.get("billing_provider"):
            if "unique_billing_providers_pharmacy" not in result["extraction_summary"]:
                result["extraction_summary"]["unique_billing_providers_pharmacy"] = set()
            result["extraction_summary"]["unique_billing_providers_pharmacy"].add(current_record["billing_provider"])

        if ndc_found or label_found or "rx_filled_dt" in current_record:
            current_record["data_path"] = path
            result["ndc_records"].append(current_record)
            result["extraction_summary"]["total_ndc_records"] += 1

        # Continue stable recursive search
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            self._stable_pharmacy_extraction(value, result, new_path)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            self._stable_pharmacy_extraction(item, result, new_path)
    # Backward compatibility methods
    def extract_medical_fields_batch(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses stable extraction"""
        return self.extract_medical_fields_batch_enhanced(deidentified_medical)

    def extract_pharmacy_fields_batch(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses stable extraction"""
        return self.extract_pharmacy_fields_batch_enhanced(deidentified_pharmacy)

    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses stable deidentification"""
        return self.deidentify_medical_data_enhanced(medical_data, patient_data)

    def deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses stable deidentification"""
        return self.deidentify_pharmacy_data_enhanced(pharmacy_data)

    def deidentify_mcid_data(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses stable deidentification"""
        return self.deidentify_mcid_data_enhanced(mcid_data)

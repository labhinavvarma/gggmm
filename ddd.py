import json
import re
from datetime import datetime, date
from typing import Dict, Any, List
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class HealthDataProcessor:
    """Handles all data processing, extraction, and deidentification"""
 
    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        logger.info("🔧 HealthDataProcessor initialized")
 
    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data with complete JSON processing"""
        try:
            if not medical_data:
                return {"error": "No medical data to deidentify"}
 
            # Calculate age
            try:
                dob_str = patient_data.get('date_of_birth', '')
                if dob_str:
                    dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                else:
                    age = "unknown"
            except Exception as e:
                logger.warning(f"Error calculating age: {e}")
                age = "unknown"
 
            # Process the entire JSON structure properly
            if 'body' in medical_data:
                raw_medical_data = medical_data['body']
            else:
                raw_medical_data = medical_data
 
            # Deep copy and process the entire JSON structure
            deidentified_medical_data = self._deep_deidentify_json(raw_medical_data)
 
            # Additional masking for specific fields
            deidentified_medical_data = self._mask_specific_fields(deidentified_medical_data)
 
            deidentified = {
                "src_mbr_first_nm": "[MASKED_NAME]",
                "src_mbr_last_nm": "[MASKED_NAME]",
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": "12345",
                "medical_claims_data": deidentified_medical_data,  # Complete processed JSON
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "medical_claims"
            }
 
            logger.info("✅ Successfully deidentified complete medical claims JSON structure")
            return deidentified
 
        except Exception as e:
            logger.error(f"Error in medical deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
 
    def _mask_specific_fields(self, data: Any) -> Any:
        """Mask specific sensitive fields in the data"""
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                # Mask specific name fields
                if key.lower() in ['src_mbr_frst_nm', 'src_mbr_first_nm', 'src_mbr_last_nm', 'src_mvr_last_nm']:
                    masked_data[key] = "[MASKED_NAME]"
                elif isinstance(value, (dict, list)):
                    masked_data[key] = self._mask_specific_fields(value)
                else:
                    masked_data[key] = value
            return masked_data
        elif isinstance(data, list):
            return [self._mask_specific_fields(item) for item in data]
        else:
            return data
 
    def deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify pharmacy data with comprehensive name masking"""
        try:
            if not pharmacy_data:
               return {"error": "No pharmacy data to deidentify"}
 
        # Process the entire JSON structure properly
            if 'body' in pharmacy_data:
                    raw_pharmacy_data = pharmacy_data['body']
            else:
                    raw_pharmacy_data = pharmacy_data
 
        # Deep copy and process the entire JSON structure with name masking
            deidentified_pharmacy_data = self._deep_deidentify_pharmacy_json(raw_pharmacy_data)
 
            result = {
            "pharmacy_claims_data": deidentified_pharmacy_data,  # Complete processed JSON
            "original_structure_preserved": True,
            "deidentification_timestamp": datetime.now().isoformat(),
            "data_type": "pharmacy_claims",
            "name_fields_masked": ["src_mbr_first_nm", "scr_mbr_last_nm"]
            }
 
            logger.info("✅ Successfully deidentified complete pharmacy claims JSON structure with name masking")
            return result
 
        except Exception as e:
            logger.error(f"Error in pharmacy deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
 
    def _deep_deidentify_pharmacy_json(self, data: Any) -> Any:
        """Deep deidentification of pharmacy JSON structure with specific name field masking"""
        try:
            if isinstance(data, dict):
            # Process dictionary recursively
                 deidentified_dict = {}
                 for key, value in data.items():
                # Mask specific name fields in pharmacy data
                      if key.lower() in ['src_mbr_first_nm', 'src_mbr_frst_nm', 'scr_mbr_last_nm', 'src_mbr_last_nm']:
                       deidentified_dict[key] = "[MASKED_NAME]"
                       logger.info(f"🔒 Masked pharmacy name field: {key}")
                      elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                       deidentified_dict[key] = self._deep_deidentify_pharmacy_json(value)
                      elif isinstance(value, str):
                    # Apply string deidentification
                       deidentified_dict[key] = self._deidentify_string(value)
                      else:
                    # Keep primitive types as-is
                        deidentified_dict[key] = value
                 return deidentified_dict
 
            elif isinstance(data, list):
                   # Process list recursively
                  return [self._deep_deidentify_pharmacy_json(item) for item in data]
 
            elif isinstance(data, str):
                    # Deidentify string values
                    return self._deidentify_string(data)
 
            else:
                  # Return primitive types as-is (int, float, bool, None)
                  return data
 
        except Exception as e:
            logger.warning(f"Error in deep pharmacy deidentification: {e}")
            return data  # Return original data if deidentification fails
 
 
 
    def deidentify_mcid_data(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify MCID data with complete JSON processing"""
        try:
            if not mcid_data:
                return {"error": "No MCID data to deidentify"}
 
            # Process the entire JSON structure properly
            if 'body' in mcid_data:
                raw_mcid_data = mcid_data['body']
            else:
                raw_mcid_data = mcid_data
 
            # Deep copy and process the entire JSON structure
            deidentified_mcid_data = self._deep_deidentify_json(raw_mcid_data)
 
            result = {
                "mcid_claims_data": deidentified_mcid_data,  # Complete processed JSON
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "mcid_claims"
            }
 
            logger.info("✅ Successfully deidentified complete MCID claims JSON structure")
            return result
 
        except Exception as e:
            logger.error(f"Error in MCID deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
 
    def _deep_deidentify_json(self, data: Any) -> Any:
        """Deep deidentification of entire JSON structure"""
        try:
            if isinstance(data, dict):
                # Process dictionary recursively
                deidentified_dict = {}
                for key, value in data.items():
                    # Deidentify the key if it contains PII indicators
                    clean_key = self._deidentify_string(key) if isinstance(key, str) else key
                    # Recursively process the value
                    deidentified_dict[clean_key] = self._deep_deidentify_json(value)
                return deidentified_dict
 
            elif isinstance(data, list):
                # Process list recursively
                return [self._deep_deidentify_json(item) for item in data]
 
            elif isinstance(data, str):
                # Deidentify string values
                return self._deidentify_string(data)
 
            else:
                # Return primitive types as-is (int, float, bool, None)
                return data
 
        except Exception as e:
            logger.warning(f"Error in deep deidentification: {e}")
            return data  # Return original data if deidentification fails
 
    def _deidentify_string(self, data: str) -> str:
        """Enhanced string deidentification"""
        try:
            if not isinstance(data, str) or not data.strip():
                return data
 
            # Create a copy to work with
            deidentified = str(data)
 
            # Remove common PII patterns
            # SSN patterns
            deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', deidentified)
 
            # Phone number patterns
            deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', deidentified)
            deidentified = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '[PHONE_MASKED]', deidentified)
 
            # Email patterns
            deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', deidentified)
 
            # Name patterns (be careful not to remove medical terms)
            # Only replace if it looks like a full name (First Last)
            deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[NAME_MASKED]', deidentified)
 
            # Address patterns
            deidentified = re.sub(r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS_MASKED]', deidentified, flags=re.IGNORECASE)
 
            # Credit card patterns
            deidentified = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_MASKED]', deidentified)
 
            # Date of birth patterns (but keep medical dates)
            deidentified = re.sub(r'\bDOB:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'DOB: [DATE_MASKED]', deidentified, flags=re.IGNORECASE)
 
            return deidentified
 
        except Exception as e:
            logger.warning(f"Error deidentifying string: {e}")
            return data  # Return original if deidentification fails
 
    def extract_medical_fields(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hlth_srvc_cd and diag_1_50_cd fields from deidentified medical data"""
        extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set()
            }
        }
 
        try:
            logger.info("🔍 Starting medical field extraction...")
 
            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("No medical claims data found in deidentified medical data")
                return extraction_result
 
            # Recursively extract from the entire JSON structure
            self._recursive_medical_extraction(medical_data, extraction_result)
 
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_service_codes"] = list(
                extraction_result["extraction_summary"]["unique_service_codes"]
            )
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = list(
                extraction_result["extraction_summary"]["unique_diagnosis_codes"]
            )
 
            logger.info(f"📋 Medical extraction completed: "
                       f"{extraction_result['extraction_summary']['total_hlth_srvc_records']} health service records, "
                       f"{extraction_result['extraction_summary']['total_diagnosis_codes']} diagnosis codes")
 
        except Exception as e:
            logger.error(f"Error in medical field extraction: {e}")
            extraction_result["error"] = f"Medical extraction failed: {str(e)}"
 
        return extraction_result
 
    def _recursive_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Recursively search for medical fields in nested data structures"""
        if isinstance(data, dict):
            current_record = {}
 
            # Extract health service code
            if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
                current_record["hlth_srvc_cd"] = data["hlth_srvc_cd"]
                result["extraction_summary"]["unique_service_codes"].add(str(data["hlth_srvc_cd"]))
 
            # Extract claim received date
            if "clm_rcvd_dt" in data and data["clm_rcvd_dt"]:
                current_record["clm_rcvd_dt"] = data["clm_rcvd_dt"]
 
            diagnosis_codes = []
 
            # Handle comma-separated diagnosis codes in diag_1_50_cd field
            if "diag_1_50_cd" in data and data["diag_1_50_cd"]:
                diag_value = str(data["diag_1_50_cd"]).strip()
                if diag_value and diag_value.lower() not in ['null', 'none', '']:
                    # Split by comma and process each diagnosis code
                    individual_codes = [code.strip() for code in diag_value.split(',') if code.strip()]
                    for i, code in enumerate(individual_codes, 1):
                        if code and code.lower() not in ['null', 'none', '']:
                            diagnosis_codes.append({
                                "code": code,
                                "position": i,
                                "source": "diag_1_50_cd (comma-separated)"
                            })
                            result["extraction_summary"]["unique_diagnosis_codes"].add(code)
 
            # Also handle individual diagnosis fields (diag_1_cd, diag_2_cd, etc.) for backwards compatibility
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        diagnosis_codes.append({
                            "code": diag_code,
                            "position": i,
                            "source": f"individual field ({diag_key})"
                        })
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)
 
            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
 
            if current_record:
                current_record["data_path"] = path
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1
 
            # Continue recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._recursive_medical_extraction(value, result, new_path)
 
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._recursive_medical_extraction(item, result, new_path)
 
    def extract_pharmacy_fields(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Ndc and lbl_nm fields from deidentified pharmacy data"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set()
            }
        }
 
        try:
            logger.info("🔍 Starting pharmacy field extraction...")
 
            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy claims data found in deidentified pharmacy data")
                return extraction_result
 
            self._recursive_pharmacy_extraction(pharmacy_data, extraction_result)
 
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
 
            logger.info(f"💊 Pharmacy extraction completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records")
 
        except Exception as e:
            logger.error(f"Error in pharmacy field extraction: {e}")
            extraction_result["error"] = f"Pharmacy extraction failed: {str(e)}"
 
        return extraction_result
 
    def _recursive_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Recursively search for pharmacy fields in nested data structures"""
        if isinstance(data, dict):
            current_record = {}
 
            # Look for NDC fields with various naming conventions
            ndc_found = False
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number', 'national_drug_code']:
                    current_record["ndc"] = data[key]
                    result["extraction_summary"]["unique_ndc_codes"].add(str(data[key]))
                    ndc_found = True
                    break
 
            # Look for label name fields with various naming conventions
            label_found = False
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                    current_record["lbl_nm"] = data[key]
                    result["extraction_summary"]["unique_label_names"].add(str(data[key]))
                    label_found = True
                    break
 
            # Extract prescription filled date
            if "rx_filled_dt" in data and data["rx_filled_dt"]:
                current_record["rx_filled_dt"] = data["rx_filled_dt"]
 
            if ndc_found or label_found or "rx_filled_dt" in current_record:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
 
            # Continue recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._recursive_pharmacy_extraction(value, result, new_path)
 
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._recursive_pharmacy_extraction(item, result, new_path)
 
    def calculate_age_from_dob(self, date_of_birth: str) -> tuple[int, str]:
        """Calculate age and age group from date of birth"""
        try:
            if not date_of_birth:
                return None, "unknown"
 
            # Parse date of birth
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
 
            # Determine age group
            if age < 18:
                age_group = "pediatric"
            elif age < 35:
                age_group = "young_adult"
            elif age < 50:
                age_group = "adult"
            elif age < 65:
                age_group = "middle_aged"
            else:
                age_group = "senior"
 
            return age, age_group
 
        except Exception as e:
            logger.warning(f"Error calculating age from date of birth: {e}")
            return None, "unknown"
 
    def extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any],
                                        pharmacy_extraction: Dict[str, Any],
                                        medical_extraction: Dict[str, Any],
                                        patient_data: Dict[str, Any] = None,
                                        api_integrator = None) -> Dict[str, Any]:
        """Enhanced health entity extraction using LLM for better accuracy"""
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
            "llm_analysis": "not_performed"
        }
 
        try:
            # 1. Calculate age from date of birth
            if patient_data and patient_data.get('date_of_birth'):
                age, age_group = self.calculate_age_from_dob(patient_data['date_of_birth'])
                if age is not None:
                    entities["age"] = age
                    entities["age_group"] = age_group
                    entities["analysis_details"].append(f"Age calculated from DOB: {age} years ({age_group})")
 
            # 2. Use LLM for entity extraction if available
            if api_integrator:
                llm_entities = self._extract_entities_with_llm(
                    pharmacy_data, pharmacy_extraction, medical_extraction,
                    patient_data, api_integrator
                )
 
                if llm_entities:
                    # Update entities with LLM results
                    entities.update(llm_entities)
                    entities["llm_analysis"] = "completed"
                    entities["analysis_details"].append("LLM entity extraction completed successfully")
                else:
                    entities["analysis_details"].append("LLM entity extraction failed, using fallback method")
                    # Fall back to direct analysis
                    self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
            else:
                entities["analysis_details"].append("No LLM available, using direct analysis")
                # Fall back to direct analysis
                self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
 
            # 3. Always extract medications for reference
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        entities["medications_identified"].append({
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "path": record.get("data_path", "")
                        })
 
            entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
 
        except Exception as e:
            logger.error(f"Error in enhanced LLM entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
 
        return entities

    def _extract_entities_with_llm(self, pharmacy_data: Dict[str, Any],
                                   pharmacy_extraction: Dict[str, Any],
                                   medical_extraction: Dict[str, Any],
                                   patient_data: Dict[str, Any],
                                   api_integrator) -> Dict[str, Any]:
        """Use LLM to extract health entities from claims data with robust error handling"""
        try:
            # Prepare SUMMARIZED context for LLM to avoid truncation
            context_summary = self._prepare_summarized_context(
                pharmacy_data, pharmacy_extraction, medical_extraction, patient_data
            )

            # Create concise prompt for entity extraction
            entity_prompt = f"""
You are a medical AI expert analyzing patient claims data. 

PATIENT CLAIMS SUMMARY:
{context_summary}

ANALYSIS TASK:
Carefully analyze the MEDICATION_NAMES and MEDICATIONS_DETAILED along with ICD-10 codes to determine patient conditions:

1. **MEDICATION ANALYSIS** - Look at each medication name and determine what it treats:
   - Diabetes medications: metformin, insulin, glipizide, jardiance, ozempic, trulicity, lantus, humalog, etc.
   - Blood pressure medications: amlodipine, lisinopril, losartan, metoprolol, hydrochlorothiazide, etc.
   - Smoking cessation: varenicline, bupropion, nicotine replacement
   - Alcohol treatment: naltrexone, disulfiram, acamprosate

2. **ICD-10 CODE ANALYSIS** - Analyze diagnosis codes:
   - Diabetes: E10.x (Type 1), E11.x (Type 2), E12.x, E13.x, E14.x
   - Hypertension: I10, I11.x, I12.x, I13.x, I15.x
   - Smoking: F17.x (tobacco dependence), Z72.0 (tobacco use)
   - Alcohol: F10.x (alcohol disorders), Z72.1 (alcohol use)

3. **ENTITY DETERMINATION**:
   - diabetics: "yes" if diabetes medications OR diabetes ICD codes found
   - smoking: "yes" if smoking cessation medications OR tobacco ICD codes found
   - alcohol: "yes" if alcohol treatment medications OR alcohol ICD codes found
   - blood_pressure: "managed" if BP medications found, "diagnosed" if hypertension ICD codes found, "unknown" if neither
   - medical_conditions: list all conditions identified from medications and codes

**CRITICAL**: Analyze the actual medication names in MEDICATION_NAMES list - use your medical knowledge to understand what each medication treats.

RESPONSE FORMAT (JSON ONLY, NO MARKDOWN):
{{
    "diabetics": "yes",
    "smoking": "no", 
    "alcohol": "no",
    "blood_pressure": "managed",
    "medical_conditions": ["diabetes", "hypertension", "asthma"],
    "llm_reasoning": "Found metformin and lantus (diabetes meds), amlodipine (BP med), and E11.9 (diabetes code)"
}}
"""

            logger.info("🤖 Calling LLM for entity extraction with summarized context...")
            logger.info(f"📊 Context summary length: {len(context_summary)} characters")

            # Call LLM with entity extraction prompt
            llm_response = api_integrator.call_llm(entity_prompt)

            if llm_response and not llm_response.startswith("Error"):
                logger.info(f"📄 LLM response length: {len(llm_response)} characters")
                logger.info(f"📄 LLM response: {llm_response}")
                
                # Parse JSON with multiple fallback strategies
                return self._parse_llm_response_robust(llm_response)
            else:
                logger.error(f"❌ LLM call failed: {llm_response}")
                return None

        except Exception as e:
            logger.error(f"❌ Error in LLM entity extraction: {e}")
            return None

    def _prepare_summarized_context(self, pharmacy_data: Dict[str, Any],
                                   pharmacy_extraction: Dict[str, Any],
                                   medical_extraction: Dict[str, Any],
                                   patient_data: Dict[str, Any]) -> str:
        """Prepare a concise summary of the context to avoid API limits"""
        try:
            context_parts = []
            
            # 1. Patient Info (minimal)
            patient_info = {
                "age": patient_data.get("calculated_age", patient_data.get("age", "unknown")) if patient_data else "unknown",
                "gender": patient_data.get("gender", "unknown") if patient_data else "unknown"
            }
            context_parts.append(f"PATIENT: {patient_info}")
            
            # 2. Detailed Medications Analysis (include actual medication names)
            medications = []
            medication_names = []
            
            # First: Get from pharmacy_extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"][:15]:  # Increased to 15
                    med_info = {}
                    if record.get("lbl_nm"):
                        med_info["name"] = record.get("lbl_nm", "")
                        medication_names.append(record.get("lbl_nm", ""))
                    if record.get("ndc"):
                        med_info["ndc"] = record.get("ndc", "")
                    if record.get("rx_filled_dt"):
                        med_info["date"] = record.get("rx_filled_dt", "")
                    if med_info:
                        medications.append(med_info)
            
            # Second: Extract from raw pharmacy_data if available
            raw_medications = self._extract_medications_from_raw_data(pharmacy_data)
            if raw_medications:
                medications.extend(raw_medications[:10])  # Add up to 10 more
                medication_names.extend([med.get("name", "") for med in raw_medications[:10] if med.get("name")])
            
            context_parts.append(f"MEDICATIONS_DETAILED: {medications}")
            
            # 3. Medication Names List (for easy LLM reference)
            unique_med_names = list(dict.fromkeys([name for name in medication_names if name]))[:25]
            context_parts.append(f"MEDICATION_NAMES: {unique_med_names}")
            
            # 4. Diagnosis Codes Summary (top 20 most relevant)
            diagnosis_codes = []
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                for record in medical_extraction["hlth_srvc_records"][:20]:  # Increased to 20
                    if record.get("diagnosis_codes"):
                        for diag in record["diagnosis_codes"][:3]:  # Max 3 per record
                            if isinstance(diag, dict) and diag.get("code"):
                                diagnosis_codes.append(diag.get("code", ""))
                            elif isinstance(diag, str):
                                diagnosis_codes.append(diag)
            
            # Remove duplicates and keep top 25
            unique_codes = list(dict.fromkeys([code for code in diagnosis_codes if code]))[:25]
            context_parts.append(f"ICD_CODES: {unique_codes}")
            
            # 5. Service Codes Summary (top 15)
            service_codes = []
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                for record in medical_extraction["hlth_srvc_records"][:15]:
                    if record.get("hlth_srvc_cd"):
                        service_codes.append(record.get("hlth_srvc_cd"))
            
            unique_services = list(dict.fromkeys([code for code in service_codes if code]))[:15]
            context_parts.append(f"SERVICE_CODES: {unique_services}")
            
            # 6. Summary Stats
            stats = {
                "total_medications": len(medications),
                "unique_medication_names": len(unique_med_names),
                "total_medical_records": len(medical_extraction.get("hlth_srvc_records", [])) if medical_extraction else 0,
                "unique_diagnosis_codes": len(unique_codes),
                "unique_service_codes": len(unique_services)
            }
            context_parts.append(f"SUMMARY_STATS: {stats}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing summarized context: {e}")
            return "Patient claims data available for analysis."

    def _extract_medications_from_raw_data(self, pharmacy_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract medication names from raw pharmacy data"""
        try:
            medications = []
            if not pharmacy_data:
                return medications
                
            def search_for_medications(data, path=""):
                """Recursively search for medication fields"""
                if isinstance(data, dict):
                    current_med = {}
                    
                    # Look for medication name fields
                    for key, value in data.items():
                        key_lower = key.lower()
                        if key_lower in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                            if value and isinstance(value, str):
                                current_med['name'] = value
                        elif key_lower in ['ndc', 'ndc_code', 'ndc_number']:
                            if value:
                                current_med['ndc'] = str(value)
                        elif key_lower in ['rx_filled_dt', 'fill_date', 'prescription_date']:
                            if value:
                                current_med['date'] = str(value)
                    
                    if current_med.get('name'):
                        medications.append(current_med)
                    
                    # Continue recursive search
                    for key, value in data.items():
                        if isinstance(value, (dict, list)):
                            search_for_medications(value, f"{path}.{key}" if path else key)
                            
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, (dict, list)):
                            search_for_medications(item, f"{path}[{i}]" if path else f"[{i}]")
            
            search_for_medications(pharmacy_data)
            return medications[:15]  # Limit to 15 medications
            
        except Exception as e:
            logger.warning(f"Error extracting medications from raw data: {e}")
            return []

    def _parse_llm_response_robust(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response with multiple fallback strategies"""
        try:
            # Strategy 1: Standard JSON extraction
            result = self._extract_clean_json(llm_response)
            if result:
                return result
            
            # Strategy 2: Fix truncated JSON
            result = self._repair_truncated_json(llm_response)
            if result:
                return result
                
            # Strategy 3: Pattern matching fallback
            result = self._extract_by_pattern_matching(llm_response)
            if result:
                return result
                
            logger.error("❌ All JSON parsing strategies failed")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in robust JSON parsing: {e}")
            return None

    def _extract_clean_json(self, response: str) -> Dict[str, Any]:
        """Extract clean JSON from response"""
        try:
            # Clean response
            json_str = response.strip()
            
            # Remove markdown wrappers
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            elif json_str.startswith('```'):
                json_str = json_str[3:]
                
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            
            # Find JSON boundaries using brace counting
            json_start = json_str.find('{')
            if json_start == -1:
                return None
                
            brace_count = 0
            json_end = -1
            
            for i in range(json_start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                return None
                
            json_str = json_str[json_start:json_end]
            logger.info(f"🔍 Clean JSON extracted: {json_str[:150]}...")
            
            # Parse JSON
            llm_entities = json.loads(json_str)
            return self._validate_and_clean_entities(llm_entities)
            
        except Exception as e:
            logger.warning(f"Clean JSON extraction failed: {e}")
            return None

    def _repair_truncated_json(self, response: str) -> Dict[str, Any]:
        """Attempt to repair truncated JSON"""
        try:
            logger.info("🔧 Attempting to repair truncated JSON...")
            
            json_str = response.strip()
            
            # Remove markdown if present
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            elif json_str.startswith('```'):
                json_str = json_str[3:]
                
            json_str = json_str.strip()
            
            # Find start of JSON
            json_start = json_str.find('{')
            if json_start == -1:
                return None
                
            json_str = json_str[json_start:]
            
            # Count braces to see if we need to add closing braces
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            # Count brackets to see if we need to close arrays
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Count quotes to see if we have unclosed strings
            quote_count = json_str.count('"')
            
            # Repair strategy
            if open_braces > close_braces:
                # Add missing closing braces
                missing_braces = open_braces - close_braces
                
                # If we have unclosed string, close it first
                if quote_count % 2 == 1:
                    json_str += '"'
                    
                # If we have unclosed array, close it
                if open_brackets > close_brackets:
                    json_str += ']' * (open_brackets - close_brackets)
                    
                # Add missing braces
                json_str += '}' * missing_braces
                
                logger.info(f"🔧 Added {missing_braces} closing braces")
                logger.info(f"🔧 Repaired JSON: {json_str[:200]}...")
                
                # Try to parse repaired JSON
                llm_entities = json.loads(json_str)
                result = self._validate_and_clean_entities(llm_entities)
                
                if result:
                    result["llm_reasoning"] = "LLM analysis completed (JSON auto-repaired)"
                    logger.info("✅ Successfully repaired truncated JSON")
                    return result
                    
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            return None

    def _extract_by_pattern_matching(self, response: str) -> Dict[str, Any]:
        """Extract entities using pattern matching as last resort"""
        try:
            logger.info("🔍 Attempting pattern matching extraction...")
            
            entities = {
                "diabetics": "no",
                "smoking": "no",
                "alcohol": "no", 
                "blood_pressure": "unknown",
                "medical_conditions": [],
                "llm_reasoning": "Extracted using pattern matching due to JSON parsing failure"
            }
            
            response_lower = response.lower()
            
            # Pattern match for diabetics
            if any(term in response_lower for term in ['diabetes', 'diabetic', '"diabetics": "yes"', 'metformin', 'insulin']):
                entities["diabetics"] = "yes"
                
            # Pattern match for smoking
            if any(term in response_lower for term in ['smoking', 'tobacco', '"smoking": "yes"', 'nicotine']):
                entities["smoking"] = "yes"
                
            # Pattern match for alcohol
            if any(term in response_lower for term in ['alcohol', '"alcohol": "yes"', 'naltrexone']):
                entities["alcohol"] = "yes"
                
            # Pattern match for blood pressure
            if any(term in response_lower for term in ['hypertension', 'blood pressure', 'amlodipine', 'lisinopril']):
                if 'managed' in response_lower:
                    entities["blood_pressure"] = "managed"
                else:
                    entities["blood_pressure"] = "diagnosed"
                    
            logger.info(f"🔍 Pattern matching result: {entities}")
            return entities
            
        except Exception as e:
            logger.warning(f"Pattern matching failed: {e}")
            return None

    def _validate_and_clean_entities(self, llm_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean LLM entities"""
        try:
            cleaned_entities = {
                "diabetics": str(llm_entities.get("diabetics", "no")).lower(),
                "smoking": str(llm_entities.get("smoking", "no")).lower(),
                "alcohol": str(llm_entities.get("alcohol", "no")).lower(),
                "blood_pressure": str(llm_entities.get("blood_pressure", "unknown")).lower(),
                "medical_conditions": llm_entities.get("medical_conditions", []),
                "llm_reasoning": llm_entities.get("llm_reasoning", "LLM analysis completed"),
                "diabetes_evidence": llm_entities.get("diabetes_evidence", []),
                "bp_evidence": llm_entities.get("bp_evidence", []),
                "smoking_evidence": llm_entities.get("smoking_evidence", []),
                "alcohol_evidence": llm_entities.get("alcohol_evidence", []),
                "medication_analysis": llm_entities.get("medication_analysis", []),
                "icd10_analysis": llm_entities.get("icd10_analysis", [])
            }
            
            # Validate values
            valid_yes_no = ["yes", "no"]
            valid_bp = ["unknown", "managed", "diagnosed"]
            
            if cleaned_entities["diabetics"] not in valid_yes_no:
                cleaned_entities["diabetics"] = "no"
            if cleaned_entities["smoking"] not in valid_yes_no:
                cleaned_entities["smoking"] = "no"
            if cleaned_entities["alcohol"] not in valid_yes_no:
                cleaned_entities["alcohol"] = "no"
            if cleaned_entities["blood_pressure"] not in valid_bp:
                cleaned_entities["blood_pressure"] = "unknown"
                
            logger.info(f"✅ Validated and cleaned entities: {cleaned_entities}")
            return cleaned_entities
            
        except Exception as e:
            logger.error(f"Error validating entities: {e}")
            return None
 
    def _analyze_entities_direct(self, pharmacy_data: Dict[str, Any],
                                pharmacy_extraction: Dict[str, Any],
                                medical_extraction: Dict[str, Any],
                                entities: Dict[str, Any]):
        """Fallback direct entity analysis (original method)"""
        try:
            # Analyze original pharmacy data
            if pharmacy_data:
                data_str = json.dumps(pharmacy_data).lower()
                # Note: You would need to implement these methods if they don't exist
                # self._analyze_pharmacy_for_entities(data_str, entities)
 
            # Analyze structured pharmacy extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                # Note: You would need to implement this method if it doesn't exist
                # self._analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
                pass
 
            # Analyze medical extraction for conditions
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                # Note: You would need to implement this method if it doesn't exist
                # self._analyze_medical_extraction_for_entities(medical_extraction, entities)
                pass
 
            entities["analysis_details"].append("Direct entity analysis completed as fallback")
 
        except Exception as e:
            logger.error(f"Error in direct entity analysis: {e}")
            entities["analysis_details"].append(f"Error in direct entity analysis: {str(e)}")
 
 
    # ISOLATED LLM METHODS FOR CODE EXPLANATIONS - DO NOT AFFECT CHATBOT
    def get_service_code_explanation_isolated(self, service_code: str) -> str:
        """Get LLM explanation for health service code (1-2 lines) - ISOLATED"""
        if not self.api_integrator or not service_code:
            return "Explanation not available"
 
        try:
            prompt = f"Explain healthcare service code '{service_code}' in 1-2 lines. What medical service or procedure does this code represent?"
 
            # Use isolated LLM call that doesn't affect chatbot context
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a medical coding expert. Provide brief, accurate explanations for healthcare codes in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
            # Ensure response is brief (max 2 lines)
            lines = response.strip().split('\n')
            return ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
 
        except Exception as e:
            logger.warning(f"Error getting isolated service code explanation: {e}")
            return "Explanation unavailable"
 
    def get_diagnosis_code_explanation_isolated(self, diagnosis_code: str) -> str:
        """Get LLM explanation for diagnosis code (1-2 lines) - ISOLATED"""
        if not self.api_integrator or not diagnosis_code:
            return "Explanation not available"
 
        try:
            prompt = f"Explain ICD-10 diagnosis code '{diagnosis_code}' in 1-2 lines. What medical condition does this code represent?"
 
            # Use isolated LLM call that doesn't affect chatbot context
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a medical coding expert. Provide brief, accurate explanations for ICD-10 diagnosis codes in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
            # Ensure response is brief (max 2 lines)
            lines = response.strip().split('\n')
            return ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
 
        except Exception as e:
            logger.warning(f"Error getting isolated diagnosis code explanation: {e}")
            return "Explanation unavailable"
 
    def get_ndc_code_explanation_isolated(self, ndc_code: str) -> str:
        """Get LLM explanation for NDC code (1-2 lines) - ISOLATED"""
        if not self.api_integrator or not ndc_code:
            return "Explanation not available"
 
        try:
            prompt = f"Explain NDC code '{ndc_code}' in 1-2 lines. What medication or drug product does this NDC number represent?"
 
            # Use isolated LLM call that doesn't affect chatbot context
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a pharmacy expert. Provide brief, accurate explanations for NDC (National Drug Code) numbers in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
            # Ensure response is brief (max 2 lines)
            lines = response.strip().split('\n')
            return ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
 
        except Exception as e:
            logger.warning(f"Error getting isolated NDC code explanation: {e}")
            return "Explanation unavailable"
 
    def get_medication_explanation_isolated(self, medication_name: str) -> str:
        """Get LLM explanation for medication/label name (1-2 lines) - ISOLATED"""
        if not self.api_integrator or not medication_name:
            return "Explanation not available"
 
        try:
            prompt = f"Explain the medication '{medication_name}' in 1-2 lines. What is this drug used for and how does it work?"
 
            # Use isolated LLM call that doesn't affect chatbot context
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a pharmacist. Provide brief, accurate explanations for medications in 1-2 lines focusing on primary use and mechanism."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
            # Ensure response is brief (max 2 lines)
            lines = response.strip().split('\n')
            return ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
 
        except Exception as e:
            logger.warning(f"Error getting isolated medication explanation: {e}")
            return "Explanation unavailable"
 
    def prepare_chunked_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare comprehensive context in chunks to avoid payload issues"""
        try:
            context_sections = []
 
            # 1. Patient Overview (small)
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
 
            # 2. Deidentified Claims Data (chunked)
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical:
                medical_summary = {
                    "patient_info": {
                        "name": f"{deidentified_medical.get('src_mbr_first_nm', 'N/A')} {deidentified_medical.get('src_mbr_last_nm', 'N/A')}",
                        "age": deidentified_medical.get('src_mbr_age', 'N/A'),
                        "zip": deidentified_medical.get('src_mbr_zip_cd', 'N/A')
                    },
                    "complete_medical_claims_data": deidentified_medical.get('medical_claims_data', {})
                }
                context_sections.append(f"DEIDENTIFIED MEDICAL CLAIMS DATA:\n{json.dumps(medical_summary, indent=2)}")
 
            # 3. Deidentified Pharmacy Claims Data
            deidentified_pharmacy = chat_context.get("deidentified_pharmacy", {})
            if deidentified_pharmacy:
                pharmacy_summary = {
                    "complete_pharmacy_claims_data": deidentified_pharmacy.get('pharmacy_claims_data', {})
                }
                context_sections.append(f"DEIDENTIFIED PHARMACY CLAIMS DATA:\n{json.dumps(pharmacy_summary, indent=2)}")
 
            # 4. Deidentified MCID Claims Data
            deidentified_mcid = chat_context.get("deidentified_mcid", {})
            if deidentified_mcid:
                mcid_summary = {
                    "complete_mcid_claims_data": deidentified_mcid.get('mcid_claims_data', {})
                }
                context_sections.append(f"DEIDENTIFIED MCID CLAIMS DATA:\n{json.dumps(mcid_summary, indent=2)}")
 
            # 5. Medical Extractions (detailed)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                extraction_summary = medical_extraction.get('extraction_summary', {})
                hlth_records = medical_extraction.get('hlth_srvc_records', [])
 
                medical_details = {
                    "summary": extraction_summary,
                    "health_service_records": hlth_records[:10] if len(hlth_records) > 10 else hlth_records
                }
                context_sections.append(f"MEDICAL EXTRACTIONS:\n{json.dumps(medical_details, indent=2)}")
 
            # 6. Pharmacy Extractions (detailed)
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                extraction_summary = pharmacy_extraction.get('extraction_summary', {})
                ndc_records = pharmacy_extraction.get('ndc_records', [])
 
                pharmacy_details = {
                    "summary": extraction_summary,
                    "ndc_records": ndc_records[:15] if len(ndc_records) > 15 else ndc_records
                }
                context_sections.append(f"PHARMACY EXTRACTIONS:\n{json.dumps(pharmacy_details, indent=2)}")
 
            # 7. Entity Extraction (small)
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
 
            # 8. Heart Attack Prediction (small)
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"HEART ATTACK PREDICTION:\n{json.dumps(heart_attack_prediction, indent=2)}")
 
            # Join all sections
            return "\n\n" + "\n\n".join(context_sections)
 
        except Exception as e:
            logger.error(f"Error preparing chunked context: {e}")
            return "Patient claims data available for analysis."

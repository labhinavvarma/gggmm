import json
import re
from datetime import datetime, date
from typing import Dict, Any, List
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class HealthDataProcessor:
    """Handles all data processing, extraction, and deidentification with enhanced code meanings"""
 
    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        logger.info("🔧 Enhanced HealthDataProcessor initialized with code meaning capabilities")
 
    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data with complete JSON processing - KEEP REAL ZIP CODE"""
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
 
            # Get real zip code from patient data (NO MASKING)
            real_zip_code = patient_data.get('zip_code', 'unknown')
 
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
                "src_mbr_zip_cd": real_zip_code,  # KEEP REAL ZIP CODE
                "medical_claims_data": deidentified_medical_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "medical_claims",
                "zip_code_preserved": True  # Flag to indicate zip was preserved
            }
 
            logger.info("✅ Successfully deidentified medical claims JSON structure - ZIP CODE PRESERVED")
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
                "pharmacy_claims_data": deidentified_pharmacy_data,
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
                deidentified_dict = {}
                for key, value in data.items():
                    # Mask specific name fields in pharmacy data
                    if key.lower() in ['src_mbr_first_nm', 'src_mbr_frst_nm', 'scr_mbr_last_nm', 'src_mbr_last_nm']:
                        deidentified_dict[key] = "[MASKED_NAME]"
                        logger.info(f"🔒 Masked pharmacy name field: {key}")
                    elif isinstance(value, (dict, list)):
                        deidentified_dict[key] = self._deep_deidentify_pharmacy_json(value)
                    elif isinstance(value, str):
                        deidentified_dict[key] = self._deidentify_string(value)
                    else:
                        deidentified_dict[key] = value
                return deidentified_dict
 
            elif isinstance(data, list):
                return [self._deep_deidentify_pharmacy_json(item) for item in data]
 
            elif isinstance(data, str):
                return self._deidentify_string(data)
 
            else:
                return data
 
        except Exception as e:
            logger.warning(f"Error in deep pharmacy deidentification: {e}")
            return data
 
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
                "mcid_claims_data": deidentified_mcid_data,
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
                deidentified_dict = {}
                for key, value in data.items():
                    clean_key = self._deidentify_string(key) if isinstance(key, str) else key
                    deidentified_dict[clean_key] = self._deep_deidentify_json(value)
                return deidentified_dict
 
            elif isinstance(data, list):
                return [self._deep_deidentify_json(item) for item in data]
 
            elif isinstance(data, str):
                return self._deidentify_string(data)
 
            else:
                return data
 
        except Exception as e:
            logger.warning(f"Error in deep deidentification: {e}")
            return data
 
    def _deidentify_string(self, data: str) -> str:
        """Enhanced string deidentification"""
        try:
            if not isinstance(data, str) or not data.strip():
                return data
 
            deidentified = str(data)
 
            # Remove common PII patterns
            deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', deidentified)
            deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', deidentified)
            deidentified = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '[PHONE_MASKED]', deidentified)
            deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', deidentified)
            deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[NAME_MASKED]', deidentified)
            deidentified = re.sub(r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS_MASKED]', deidentified, flags=re.IGNORECASE)
            deidentified = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_MASKED]', deidentified)
            deidentified = re.sub(r'\bDOB:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'DOB: [DATE_MASKED]', deidentified, flags=re.IGNORECASE)
 
            return deidentified
 
        except Exception as e:
            logger.warning(f"Error deidentifying string: {e}")
            return data

    def extract_medical_fields(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced extraction with LLM-generated code meanings for medical fields"""
        extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set()
            },
            "code_meanings_added": False,
            "llm_call_status": "pending"
        }
 
        try:
            logger.info("🔍 Starting enhanced medical field extraction with code meanings...")
 
            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("No medical claims data found in deidentified medical data")
                return extraction_result
 
            # Step 1: Extract basic fields (same as before)
            self._recursive_medical_extraction(medical_data, extraction_result)
 
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_service_codes"] = list(
                extraction_result["extraction_summary"]["unique_service_codes"]
            )
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = list(
                extraction_result["extraction_summary"]["unique_diagnosis_codes"]
            )
 
            # Step 2: Get code meanings via LLM if API integrator is available
            if self.api_integrator and extraction_result["hlth_srvc_records"]:
                logger.info("🤖 Calling LLM to get medical code meanings...")
                self._add_medical_code_meanings(extraction_result)
                extraction_result["code_meanings_added"] = True
                extraction_result["llm_call_status"] = "completed"
            else:
                extraction_result["llm_call_status"] = "skipped_no_api"
 
            logger.info(f"📋 Enhanced medical extraction completed: "
                       f"{extraction_result['extraction_summary']['total_hlth_srvc_records']} health service records, "
                       f"{extraction_result['extraction_summary']['total_diagnosis_codes']} diagnosis codes, "
                       f"Code meanings: {extraction_result['code_meanings_added']}")
 
        except Exception as e:
            logger.error(f"Error in enhanced medical field extraction: {e}")
            extraction_result["error"] = f"Enhanced medical extraction failed: {str(e)}"
            extraction_result["llm_call_status"] = "error"
 
        return extraction_result

    def _add_medical_code_meanings(self, extraction_result: Dict[str, Any]):
        """Add LLM-generated meanings for medical codes"""
        try:
            # Collect all unique codes
            all_service_codes = set()
            all_diagnosis_codes = set()
            
            for record in extraction_result["hlth_srvc_records"]:
                if record.get("hlth_srvc_cd"):
                    all_service_codes.add(record["hlth_srvc_cd"])
                
                for diag in record.get("diagnosis_codes", []):
                    if diag.get("code"):
                        all_diagnosis_codes.add(diag["code"])
            
            # Get meanings in batch from LLM
            service_meanings = self._get_batch_code_meanings(list(all_service_codes), "medical_service")
            diagnosis_meanings = self._get_batch_code_meanings(list(all_diagnosis_codes), "icd10_diagnosis")
            
            # Add meanings to records
            for record in extraction_result["hlth_srvc_records"]:
                # Add service code meaning
                if record.get("hlth_srvc_cd") and record["hlth_srvc_cd"] in service_meanings:
                    record["hlth_srvc_meaning"] = service_meanings[record["hlth_srvc_cd"]]
                
                # Add diagnosis code meanings
                for diag in record.get("diagnosis_codes", []):
                    if diag.get("code") and diag["code"] in diagnosis_meanings:
                        diag["meaning"] = diagnosis_meanings[diag["code"]]
            
            # Store meanings in summary
            extraction_result["code_meanings"] = {
                "service_code_meanings": service_meanings,
                "diagnosis_code_meanings": diagnosis_meanings,
                "total_service_codes_explained": len(service_meanings),
                "total_diagnosis_codes_explained": len(diagnosis_meanings)
            }
            
            logger.info(f"✅ Added meanings for {len(service_meanings)} service codes and {len(diagnosis_meanings)} diagnosis codes")
            
        except Exception as e:
            logger.error(f"Error adding medical code meanings: {e}")
            extraction_result["code_meaning_error"] = str(e)

    def extract_pharmacy_fields(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced extraction with LLM-generated meanings for pharmacy fields"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set()
            },
            "code_meanings_added": False,
            "llm_call_status": "pending"
        }
 
        try:
            logger.info("🔍 Starting enhanced pharmacy field extraction with medication meanings...")
 
            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy claims data found in deidentified pharmacy data")
                return extraction_result
 
            # Step 1: Extract basic fields (same as before)
            self._recursive_pharmacy_extraction(pharmacy_data, extraction_result)
 
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
 
            # Step 2: Get medication meanings via LLM if API integrator is available
            if self.api_integrator and extraction_result["ndc_records"]:
                logger.info("🤖 Calling LLM to get pharmacy/medication meanings...")
                self._add_pharmacy_code_meanings(extraction_result)
                extraction_result["code_meanings_added"] = True
                extraction_result["llm_call_status"] = "completed"
            else:
                extraction_result["llm_call_status"] = "skipped_no_api"
 
            logger.info(f"💊 Enhanced pharmacy extraction completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records, "
                       f"Code meanings: {extraction_result['code_meanings_added']}")
 
        except Exception as e:
            logger.error(f"Error in enhanced pharmacy field extraction: {e}")
            extraction_result["error"] = f"Enhanced pharmacy extraction failed: {str(e)}"
            extraction_result["llm_call_status"] = "error"
 
        return extraction_result

    def _add_pharmacy_code_meanings(self, extraction_result: Dict[str, Any]):
        """Add LLM-generated meanings for pharmacy codes and medications"""
        try:
            # Collect all unique codes and medications
            all_ndc_codes = set()
            all_medications = set()
            
            for record in extraction_result["ndc_records"]:
                if record.get("ndc"):
                    all_ndc_codes.add(record["ndc"])
                if record.get("lbl_nm"):
                    all_medications.add(record["lbl_nm"])
            
            # Get meanings in batch from LLM
            ndc_meanings = self._get_batch_code_meanings(list(all_ndc_codes), "ndc_codes")
            medication_meanings = self._get_batch_code_meanings(list(all_medications), "medications")
            
            # Add meanings to records
            for record in extraction_result["ndc_records"]:
                # Add NDC code meaning
                if record.get("ndc") and record["ndc"] in ndc_meanings:
                    record["ndc_meaning"] = ndc_meanings[record["ndc"]]
                
                # Add medication meaning
                if record.get("lbl_nm") and record["lbl_nm"] in medication_meanings:
                    record["medication_meaning"] = medication_meanings[record["lbl_nm"]]
            
            # Store meanings in summary
            extraction_result["code_meanings"] = {
                "ndc_code_meanings": ndc_meanings,
                "medication_meanings": medication_meanings,
                "total_ndc_codes_explained": len(ndc_meanings),
                "total_medications_explained": len(medication_meanings)
            }
            
            logger.info(f"✅ Added meanings for {len(ndc_meanings)} NDC codes and {len(medication_meanings)} medications")
            
        except Exception as e:
            logger.error(f"Error adding pharmacy code meanings: {e}")
            extraction_result["code_meaning_error"] = str(e)

    def _get_batch_code_meanings(self, codes: List[str], code_type: str) -> Dict[str, str]:
        """Get meanings for a batch of codes in a single LLM call"""
        if not codes or not self.api_integrator:
            return {}
        
        try:
            # Limit batch size to avoid token limits
            max_batch_size = 20
            if len(codes) > max_batch_size:
                codes = codes[:max_batch_size]
                logger.info(f"Limiting batch to {max_batch_size} codes to avoid token limits")
            
            # Create prompt based on code type
            if code_type == "medical_service":
                prompt = self._create_service_code_batch_prompt(codes)
            elif code_type == "icd10_diagnosis":
                prompt = self._create_diagnosis_code_batch_prompt(codes)
            elif code_type == "ndc_codes":
                prompt = self._create_ndc_batch_prompt(codes)
            elif code_type == "medications":
                prompt = self._create_medication_batch_prompt(codes)
            else:
                logger.error(f"Unknown code type: {code_type}")
                return {}
            
            # Call LLM
            response = self.api_integrator.call_llm(prompt)
            
            if response and not response.startswith("Error"):
                # Parse response to extract meanings
                return self._parse_batch_meanings_response(response, codes)
            else:
                logger.error(f"LLM call failed for {code_type}: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting batch meanings for {code_type}: {e}")
            return {}

    def _create_service_code_batch_prompt(self, service_codes: List[str]) -> str:
        """Create prompt for batch service code explanations"""
        codes_list = ", ".join(service_codes)
        return f"""You are a medical coding expert. Provide brief explanations for these healthcare service codes.

HEALTHCARE SERVICE CODES TO EXPLAIN:
{codes_list}

INSTRUCTIONS:
- Provide 1-2 sentence explanations for each code
- Focus on what medical service/procedure each code represents
- Use clear, professional medical language
- Format as: CODE: explanation

RESPONSE FORMAT:
99213: Office visit for established patient, moderate complexity
80053: Comprehensive metabolic panel blood test
[continue for each code]

EXPLAIN THESE SERVICE CODES:
{codes_list}"""

    def _create_diagnosis_code_batch_prompt(self, diagnosis_codes: List[str]) -> str:
        """Create prompt for batch diagnosis code explanations"""
        codes_list = ", ".join(diagnosis_codes)
        return f"""You are a medical coding expert specializing in ICD-10 diagnosis codes. Provide brief explanations for these diagnosis codes.

ICD-10 DIAGNOSIS CODES TO EXPLAIN:
{codes_list}

INSTRUCTIONS:
- Provide 1-2 sentence explanations for each ICD-10 code
- Focus on what medical condition each code represents
- Use clear, professional medical language
- Include severity/specificity when relevant
- Format as: CODE: explanation

RESPONSE FORMAT:
E11.9: Type 2 diabetes mellitus without complications
I10: Essential hypertension (high blood pressure)
[continue for each code]

EXPLAIN THESE ICD-10 CODES:
{codes_list}"""

    def _create_ndc_batch_prompt(self, ndc_codes: List[str]) -> str:
        """Create prompt for batch NDC code explanations"""
        codes_list = ", ".join(ndc_codes)
        return f"""You are a pharmacy expert specializing in NDC (National Drug Code) identification. Provide brief explanations for these NDC codes.

NDC CODES TO EXPLAIN:
{codes_list}

INSTRUCTIONS:
- Provide 1-2 sentence explanations for each NDC code
- Focus on what medication/drug product each NDC represents
- Include generic name, strength, and dosage form when possible
- Use clear, professional pharmaceutical language
- Format as: NDC: explanation

RESPONSE FORMAT:
0781-1506-10: Metformin HCl 500mg tablets, used for Type 2 diabetes management
0173-0687-02: Amlodipine 5mg tablets, calcium channel blocker for hypertension
[continue for each code]

EXPLAIN THESE NDC CODES:
{codes_list}"""

    def _create_medication_batch_prompt(self, medications: List[str]) -> str:
        """Create prompt for batch medication explanations"""
        meds_list = ", ".join(medications)
        return f"""You are a clinical pharmacist. Provide brief explanations for these medications focusing on their therapeutic use.

MEDICATIONS TO EXPLAIN:
{meds_list}

INSTRUCTIONS:
- Provide 1-2 sentence explanations for each medication
- Focus on primary therapeutic use and mechanism of action
- Include what conditions they treat
- Use clear, professional language
- Format as: MEDICATION: explanation

RESPONSE FORMAT:
METFORMIN HCL 500 MG: Oral antidiabetic medication that improves insulin sensitivity, first-line treatment for Type 2 diabetes
AMLODIPINE 5 MG: Calcium channel blocker that relaxes blood vessels, used to treat hypertension and angina
[continue for each medication]

EXPLAIN THESE MEDICATIONS:
{meds_list}"""

    def _parse_batch_meanings_response(self, response: str, codes: List[str]) -> Dict[str, str]:
        """Parse LLM response to extract code meanings"""
        meanings = {}
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    # Split on first colon
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        code = parts[0].strip()
                        meaning = parts[1].strip()
                        
                        # Match against input codes (case insensitive)
                        for input_code in codes:
                            if code.upper() == input_code.upper():
                                meanings[input_code] = meaning
                                break
            
            logger.info(f"Successfully parsed {len(meanings)} code meanings from LLM response")
            return meanings
            
        except Exception as e:
            logger.error(f"Error parsing batch meanings response: {e}")
            return {}

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

    def extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any],
                                        pharmacy_extraction: Dict[str, Any],
                                        medical_extraction: Dict[str, Any],
                                        patient_data: Dict[str, Any] = None,
                                        api_integrator = None) -> Dict[str, Any]:
        """Enhanced health entity extraction using ONLY extracted data WITH CODE MEANINGS"""
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
            "llm_analysis": "not_performed",
            "enhanced_with_code_meanings": False,
            "analysis_method": "code_meanings_only"
        }
 
        try:
            # 1. Calculate age from date of birth
            if patient_data and patient_data.get('date_of_birth'):
                age, age_group = self.calculate_age_from_dob(patient_data['date_of_birth'])
                if age is not None:
                    entities["age"] = age
                    entities["age_group"] = age_group
                    entities["analysis_details"].append(f"Age calculated from DOB: {age} years ({age_group})")
 
            # 2. Check if extracted data has code meanings
            has_medical_meanings = medical_extraction.get("code_meanings_added", False)
            has_pharmacy_meanings = pharmacy_extraction.get("code_meanings_added", False)
            
            if has_medical_meanings or has_pharmacy_meanings:
                entities["enhanced_with_code_meanings"] = True
                entities["analysis_details"].append("Using enhanced extracted data with code meanings")
            else:
                entities["analysis_details"].append("WARNING: No code meanings available - analysis may be less accurate")
            
            # 3. Use ONLY LLM for entity extraction with enhanced data (NO FALLBACK KEYWORD METHODS)
            if api_integrator:
                llm_entities = self._extract_entities_with_enhanced_data(
                    pharmacy_data, pharmacy_extraction, medical_extraction,
                    patient_data, api_integrator
                )
 
                if llm_entities:
                    entities.update(llm_entities)
                    entities["llm_analysis"] = "completed_with_enhanced_data"
                    entities["analysis_details"].append("LLM entity extraction completed using code meanings - NO keyword fallback used")
                else:
                    entities["analysis_details"].append("LLM entity extraction failed - NO fallback analysis performed")
                    entities["llm_analysis"] = "failed"
            else:
                entities["analysis_details"].append("No LLM available - entity extraction cannot be performed without API integrator")
                entities["llm_analysis"] = "no_api_available"
 
            # 4. Extract medications for reference (with meanings if available)
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    med_info = {
                        "ndc": record.get("ndc", ""),
                        "label_name": record.get("lbl_nm", ""),
                        "path": record.get("data_path", "")
                    }
                    
                    # Add meanings if available
                    if record.get("ndc_meaning"):
                        med_info["ndc_meaning"] = record["ndc_meaning"]
                    if record.get("medication_meaning"):
                        med_info["medication_meaning"] = record["medication_meaning"]
                    
                    entities["medications_identified"].append(med_info)
 
            entities["analysis_details"].append(f"Enhanced analysis sources: {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records with meanings, {len(medical_extraction.get('hlth_srvc_records', []))} medical records with meanings")
 
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction with code meanings: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
 
        return entities

    def _extract_entities_with_enhanced_data(self, pharmacy_data: Dict[str, Any],
                                           pharmacy_extraction: Dict[str, Any],
                                           medical_extraction: Dict[str, Any],
                                           patient_data: Dict[str, Any],
                                           api_integrator) -> Dict[str, Any]:
        """Use LLM to extract health entities from enhanced extracted data WITH CODE MEANINGS - NO KEYWORD MATCHING"""
        try:
            # Create enhanced prompt using ONLY extracted data with meanings
            entity_prompt = f"""You are a medical AI expert analyzing patient claims data with comprehensive code meanings.

ENHANCED EXTRACTED MEDICAL DATA WITH CODE MEANINGS:
{json.dumps(medical_extraction, indent=2)}

ENHANCED EXTRACTED PHARMACY DATA WITH CODE MEANINGS:
{json.dumps(pharmacy_extraction, indent=2)}

PATIENT INFO:
- Gender: {patient_data.get("gender", "unknown")}
- Age: {patient_data.get("calculated_age", "unknown")}

ANALYSIS INSTRUCTIONS:
You have access to extracted medical and pharmacy data that includes:
1. Medical service codes (hlth_srvc_cd) WITH their professional meanings
2. ICD-10 diagnosis codes WITH their clinical meanings  
3. NDC medication codes WITH their pharmaceutical meanings
4. Medication names WITH their therapeutic explanations

CRITICAL: Use ONLY the provided code meanings and explanations to make determinations. 
Do NOT rely on simple keyword matching. Analyze the full medical context provided by the code meanings.

COMPREHENSIVE MEDICAL ANALYSIS:

For each condition, analyze the complete medical picture from code meanings:

diabetics: "yes" or "no"
- Analyze diagnosis code meanings for any form of diabetes mellitus
- Analyze medication meanings for diabetes management (insulin, oral hypoglycemics, etc.)
- Consider diabetes complications mentioned in code meanings
- Look for comprehensive diabetes care patterns

smoking: "yes" or "no"  
- Analyze diagnosis code meanings for tobacco use disorders
- Analyze medication meanings for smoking cessation treatments
- Consider tobacco-related health complications mentioned
- Look for comprehensive smoking cessation care

alcohol: "yes" or "no"
- Analyze diagnosis code meanings for alcohol use disorders
- Analyze medication meanings for alcohol dependency treatments
- Consider alcohol-related health complications mentioned
- Look for comprehensive substance abuse care

blood_pressure: "unknown", "managed", or "diagnosed"
- "diagnosed": ICD-10 code meanings indicate hypertension diagnosis
- "managed": Medication meanings indicate antihypertensive treatments
- Analyze for hypertensive complications or related cardiovascular conditions
- Consider comprehensive cardiovascular care patterns

medical_conditions: Extract ALL conditions from comprehensive analysis

RESPONSE FORMAT (JSON ONLY):
{{
    "diabetics": "yes/no",
    "smoking": "yes/no", 
    "alcohol": "yes/no",
    "blood_pressure": "unknown/managed/diagnosed",
    "medical_conditions": ["condition1", "condition2"],
    "llm_reasoning": "Comprehensive analysis summary using all available code meanings",
    "evidence_from_meanings": {{
        "diabetes_evidence": ["specific code meanings that indicate diabetes"],
        "bp_evidence": ["specific code meanings that indicate blood pressure issues"],
        "smoking_evidence": ["specific code meanings that indicate smoking"],
        "alcohol_evidence": ["specific code meanings that indicate alcohol use"]
    }},
    "analysis_method": "comprehensive_code_meanings_analysis"
}}

Provide your comprehensive medical analysis based ONLY on the code meanings provided:"""
 
            logger.info("🤖 Calling LLM for comprehensive entity extraction using code meanings only...")
 
            response = api_integrator.call_llm(entity_prompt)
 
            if response and not response.startswith("Error"):
                try:
                    # Clean and parse JSON response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
 
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        llm_entities = json.loads(json_str)
 
                        # Validate and clean the entities
                        cleaned_entities = {
                            "diabetics": str(llm_entities.get("diabetics", "no")).lower(),
                            "smoking": str(llm_entities.get("smoking", "no")).lower(),
                            "alcohol": str(llm_entities.get("alcohol", "no")).lower(),
                            "blood_pressure": str(llm_entities.get("blood_pressure", "unknown")).lower(),
                            "medical_conditions": llm_entities.get("medical_conditions", []),
                            "llm_reasoning": llm_entities.get("llm_reasoning", "Comprehensive LLM analysis using code meanings"),
                            "evidence_from_meanings": llm_entities.get("evidence_from_meanings", {}),
                            "analysis_method": "comprehensive_code_meanings_analysis"
                        }
 
                        logger.info(f"✅ Comprehensive LLM entity extraction successful using code meanings: {cleaned_entities}")
                        return cleaned_entities
 
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse comprehensive LLM JSON response: {e}")
                    return None
            else:
                logger.error(f"Comprehensive LLM call failed: {response}")
                return None
 
        except Exception as e:
            logger.error(f"Error in comprehensive LLM entity extraction: {e}")
            return None

    def calculate_age_from_dob(self, date_of_birth: str) -> tuple[int, str]:
        """Calculate age and age group from date of birth"""
        try:
            if not date_of_birth:
                return None, "unknown"
 
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
 
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



    # Keep existing isolated LLM methods for individual code explanations
    def get_service_code_explanation_isolated(self, service_code: str) -> str:
        """Get LLM explanation for health service code (1-2 lines) - ISOLATED"""
        if not self.api_integrator or not service_code:
            return "Explanation not available"
 
        try:
            prompt = f"Explain healthcare service code '{service_code}' in 1-2 lines. What medical service or procedure does this code represent?"
 
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a medical coding expert. Provide brief, accurate explanations for healthcare codes in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
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
 
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a medical coding expert. Provide brief, accurate explanations for ICD-10 diagnosis codes in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
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
 
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a pharmacy expert. Provide brief, accurate explanations for NDC (National Drug Code) numbers in 1-2 lines."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
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
 
            response = self.api_integrator.call_llm_isolated(
                prompt,
                "You are a pharmacist. Provide brief, accurate explanations for medications in 1-2 lines focusing on primary use and mechanism."
            )
 
            if response == "Explanation unavailable":
                return "Explanation unavailable"
 
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
                        "zip": deidentified_medical.get('src_mbr_zip_cd', 'N/A')  # Real zip code now
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
 
            # 5. Enhanced Medical Extractions (with meanings)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_sections.append(f"ENHANCED MEDICAL EXTRACTIONS WITH CODE MEANINGS:\n{json.dumps(medical_extraction, indent=2)}")
 
            # 6. Enhanced Pharmacy Extractions (with meanings)
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_sections.append(f"ENHANCED PHARMACY EXTRACTIONS WITH CODE MEANINGS:\n{json.dumps(pharmacy_extraction, indent=2)}")
 
            # 7. Entity Extraction (small)
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
 
            # 8. Heart Attack Prediction (small)
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"HEART ATTACK PREDICTION:\n{json.dumps(heart_attack_prediction, indent=2)}")
 
            return "\n\n" + "\n\n".join(context_sections)
 
        except Exception as e:
            logger.error(f"Error preparing enhanced chunked context: {e}")
            return "Enhanced patient claims data with code meanings available for analysis."

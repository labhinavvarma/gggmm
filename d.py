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
    
    def __init__(self):
        logger.info("🔧 HealthDataProcessor initialized")
    
    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify medical data with complete JSON processing - FIXED"""
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
            
            # FIXED: Process the entire JSON structure properly
            # Get the actual medical data - handle both 'body' and direct data
            if 'body' in medical_data:
                raw_medical_data = medical_data['body']
            else:
                raw_medical_data = medical_data
            
            # Deep copy and process the entire JSON structure
            deidentified_medical_data = self._deep_deidentify_json(raw_medical_data)
            
            deidentified = {
                "src_mbr_first_nm": "john",
                "src_mbr_last_nm": "smith", 
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": "12345",
                "medical_data": deidentified_medical_data,  # Complete processed JSON
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Successfully deidentified complete medical JSON structure")
            return deidentified
            
        except Exception as e:
            logger.error(f"Error in medical deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
    
    def deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deidentify pharmacy data with complete JSON processing - FIXED"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data to deidentify"}
            
            # FIXED: Process the entire JSON structure properly
            # Get the actual pharmacy data - handle both 'body' and direct data
            if 'body' in pharmacy_data:
                raw_pharmacy_data = pharmacy_data['body']
            else:
                raw_pharmacy_data = pharmacy_data
            
            # Deep copy and process the entire JSON structure
            deidentified_pharmacy_data = self._deep_deidentify_json(raw_pharmacy_data)
            
            result = {
                "pharmacy_data": deidentified_pharmacy_data,  # Complete processed JSON
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Successfully deidentified complete pharmacy JSON structure")
            return result
            
        except Exception as e:
            logger.error(f"Error in pharmacy deidentification: {e}")
            return {"error": f"Deidentification failed: {str(e)}"}
    
    def _deep_deidentify_json(self, data: Any) -> Any:
        """Deep deidentification of entire JSON structure - FIXED"""
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
        """Enhanced string deidentification - FIXED"""
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
        """Extract hlth_srvc_cd and diag_1_50_cd fields from deidentified medical data - FIXED for comma-separated"""
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
            
            medical_data = deidentified_medical.get("medical_data", {})
            if not medical_data:
                logger.warning("No medical data found in deidentified medical data")
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
        """Recursively search for medical fields in nested data structures - FIXED for comma-separated diagnosis codes"""
        if isinstance(data, dict):
            current_record = {}
            
            # Extract health service code
            if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
                current_record["hlth_srvc_cd"] = data["hlth_srvc_cd"]
                result["extraction_summary"]["unique_service_codes"].add(str(data["hlth_srvc_cd"]))
            
            diagnosis_codes = []
            
            # FIXED: Handle comma-separated diagnosis codes in diag_1_50_cd field
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
            
            pharmacy_data = deidentified_pharmacy.get("pharmacy_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy data found in deidentified pharmacy data")
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
            
            if ndc_found or label_found:
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
                                        medical_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced health entity extraction using pharmacy data, extractions, and medical codes"""
        entities = {
            "diabetics": "no",
            "age_group": "unknown", 
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": []
        }
        
        try:
            # Analyze original pharmacy data
            if pharmacy_data:
                data_str = json.dumps(pharmacy_data).lower()
                self._analyze_pharmacy_for_entities(data_str, entities)
            
            # Analyze structured pharmacy extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                self._analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
            
            # Analyze medical extraction for conditions
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                self._analyze_medical_extraction_for_entities(medical_extraction, entities)
            
            entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        
        return entities
    
    def _analyze_pharmacy_for_entities(self, data_str: str, entities: Dict[str, Any]):
        """Original pharmacy data analysis for entities"""
        diabetes_keywords = [
            'insulin', 'metformin', 'glipizide', 'diabetes', 'diabetic', 
            'glucophage', 'lantus', 'humalog', 'novolog', 'levemir'
        ]
        for keyword in diabetes_keywords:
            if keyword in data_str:
                entities["diabetics"] = "yes"
                entities["analysis_details"].append(f"Diabetes indicator found in pharmacy data: {keyword}")
                break
        
        senior_medications = [
            'aricept', 'warfarin', 'lisinopril', 'atorvastatin', 'metoprolol',
            'furosemide', 'amlodipine', 'simvastatin'
        ]
        adult_medications = [
            'adderall', 'vyvanse', 'accutane', 'birth control'
        ]
        
        for med in senior_medications:
            if med in data_str:
                entities["age_group"] = "senior"
                entities["analysis_details"].append(f"Senior medication found: {med}")
                break
        
        if entities["age_group"] == "unknown":
            for med in adult_medications:
                if med in data_str:
                    entities["age_group"] = "adult"
                    entities["analysis_details"].append(f"Adult medication found: {med}")
                    break
    
    def _analyze_pharmacy_extraction_for_entities(self, pharmacy_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze structured pharmacy extraction for health entities"""
        ndc_records = pharmacy_extraction.get("ndc_records", [])
        
        for record in ndc_records:
            ndc = record.get("ndc", "")
            lbl_nm = record.get("lbl_nm", "")
            
            if lbl_nm:
                entities["medications_identified"].append({
                    "ndc": ndc,
                    "label_name": lbl_nm,
                    "path": record.get("data_path", "")
                })
                
                lbl_lower = lbl_nm.lower()
                
                if any(word in lbl_lower for word in ['insulin', 'metformin', 'glucophage', 'diabetes']):
                    entities["diabetics"] = "yes"
                    entities["analysis_details"].append(f"Diabetes medication found in extraction: {lbl_nm}")
                
                if any(word in lbl_lower for word in ['lisinopril', 'amlodipine', 'metoprolol', 'blood pressure']):
                    entities["blood_pressure"] = "managed"
                    entities["analysis_details"].append(f"Blood pressure medication found in extraction: {lbl_nm}")
    
    def _analyze_medical_extraction_for_entities(self, medical_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Analyze medical codes for health conditions"""
        hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
        
        condition_mappings = {
            "diabetes": ["E10", "E11", "E12", "E13", "E14"],
            "hypertension": ["I10", "I11", "I12", "I13", "I15"],
            "smoking": ["Z72.0", "F17"],
            "alcohol": ["F10", "Z72.1"],
        }
        
        for record in hlth_srvc_records:
            diagnosis_codes = record.get("diagnosis_codes", [])
            for diag in diagnosis_codes:
                diag_code = diag.get("code", "")
                if diag_code:
                    for condition, code_prefixes in condition_mappings.items():
                        if any(diag_code.startswith(prefix) for prefix in code_prefixes):
                            if condition == "diabetes":
                                entities["diabetics"] = "yes"
                                entities["medical_conditions"].append(f"Diabetes (ICD-10: {diag_code})")
                            elif condition == "hypertension":
                                entities["blood_pressure"] = "diagnosed"
                                entities["medical_conditions"].append(f"Hypertension (ICD-10: {diag_code})")
                            elif condition == "smoking":
                                entities["smoking"] = "yes"
                                entities["medical_conditions"].append(f"Smoking (ICD-10: {diag_code})")
                            
                            entities["analysis_details"].append(f"Medical condition identified from ICD-10 {diag_code}: {condition}")
    
    def prepare_chunked_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare comprehensive context in chunks to avoid payload issues"""
        try:
            context_sections = []
            
            # 1. Patient Overview (small)
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
            
            # 2. Deidentified Medical Data (chunked) - FIXED to include entire JSON
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical:
                # Include complete medical data with summary
                medical_summary = {
                    "patient_info": {
                        "name": f"{deidentified_medical.get('src_mbr_first_nm', 'N/A')} {deidentified_medical.get('src_mbr_last_nm', 'N/A')}",
                        "age": deidentified_medical.get('src_mbr_age', 'N/A'),
                        "zip": deidentified_medical.get('src_mbr_zip_cd', 'N/A')
                    },
                    "complete_medical_data": deidentified_medical.get('medical_data', {})  # FIXED: Include entire JSON
                }
                context_sections.append(f"DEIDENTIFIED MEDICAL DATA:\n{json.dumps(medical_summary, indent=2)}")
            
            # 3. Medical Extractions (detailed)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                extraction_summary = medical_extraction.get('extraction_summary', {})
                hlth_records = medical_extraction.get('hlth_srvc_records', [])
                
                medical_details = {
                    "summary": extraction_summary,
                    "health_service_records": hlth_records[:10] if len(hlth_records) > 10 else hlth_records
                }
                context_sections.append(f"MEDICAL EXTRACTIONS:\n{json.dumps(medical_details, indent=2)}")
            
            # 4. Pharmacy Extractions (detailed)
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                extraction_summary = pharmacy_extraction.get('extraction_summary', {})
                ndc_records = pharmacy_extraction.get('ndc_records', [])
                
                pharmacy_details = {
                    "summary": extraction_summary,
                    "ndc_records": ndc_records[:15] if len(ndc_records) > 15 else ndc_records
                }
                context_sections.append(f"PHARMACY EXTRACTIONS:\n{json.dumps(pharmacy_details, indent=2)}")
            
            # 5. Entity Extraction (small)
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES:\n{json.dumps(entity_extraction, indent=2)}")
            
            # 6. Heart Attack Prediction (small)
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"HEART ATTACK PREDICTION:\n{json.dumps(heart_attack_prediction, indent=2)}")
            
            # 7. Health Analysis (text summaries)
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                trajectory_text = health_trajectory[:1000] + "..." if len(health_trajectory) > 1000 else health_trajectory
                context_sections.append(f"HEALTH TRAJECTORY ANALYSIS:\n{trajectory_text}")
            
            final_summary = chat_context.get("final_summary", "")
            if final_summary:
                summary_text = final_summary[:1000] + "..." if len(final_summary) > 1000 else final_summary
                context_sections.append(f"CLINICAL SUMMARY:\n{summary_text}")
            
            # Join all sections
            return "\n\n" + "\n\n".join(context_sections)
            
        except Exception as e:
            logger.error(f"Error preparing chunked context: {e}")
            return "Patient medical data available for analysis."

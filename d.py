import json
import re
import copy
from datetime import datetime, date
from typing import Dict, Any, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataProcessor:
    """Enhanced data processor with comprehensive nested JSON deidentification"""
    
    def __init__(self):
        logger.info("🔧 Enhanced HealthDataProcessor initialized")
        
        # Enhanced PII patterns for comprehensive deidentification
        self.pii_patterns = {
            'ssn': [
                r'\b\d{3}-?\d{2}-?\d{4}\b',
                r'\b\d{9}\b'
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',
                r'\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'names': [
                r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b',
                r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b'
            ],
            'addresses': [
                r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Circle|Cir|Court|Ct|Place|Pl)\b',
                r'\b[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'
            ],
            'credit_cards': [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ],
            'dates_of_birth': [
                r'\bDOB:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\bDate\s+of\s+Birth:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            ],
            'member_ids': [
                r'\b[A-Z]{2,3}\d{6,12}\b',
                r'\bMBR[-_]?ID:?\s*[A-Z0-9]{6,15}\b'
            ]
        }
        
        # Medical terms to preserve (don't deidentify)
        self.medical_preserve_patterns = [
            r'\bICD[-_]?\d+',
            r'\bCPT[-_]?\d+',
            r'\bNDC[-_]?\d+',
            r'\b[A-Z]\d{2}\.?\d*\b',  # ICD-10 codes
            r'\b\d{5}[-]?\d*\b'      # CPT codes
        ]
    
    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced medical data deidentification with comprehensive JSON processing"""
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
            
            # Enhanced JSON processing - handle both 'body' and direct data
            if 'body' in medical_data:
                raw_medical_data = medical_data['body']
            else:
                raw_medical_data = medical_data
            
            # Deep copy and comprehensively deidentify the entire JSON structure
            logger.info("🔒 Starting comprehensive medical data deidentification...")
            deidentified_medical_data = self._comprehensive_deidentify_json(
                copy.deepcopy(raw_medical_data), 
                data_type="medical"
            )
            
            deidentified = {
                "src_mbr_first_nm": "john",
                "src_mbr_last_nm": "smith", 
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": "12345",
                "medical_data": deidentified_medical_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "deidentification_level": "comprehensive_nested",
                "total_fields_processed": self._count_total_fields(deidentified_medical_data),
                "deidentification_stats": self._get_deidentification_stats()
            }
            
            logger.info("✅ Successfully deidentified comprehensive medical JSON structure")
            logger.info(f"📊 Processed {deidentified['total_fields_processed']} total fields")
            
            return deidentified
            
        except Exception as e:
            logger.error(f"Error in enhanced medical deidentification: {e}")
            return {"error": f"Enhanced deidentification failed: {str(e)}"}
    
    def deidentify_mcid_data(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced MCID data deidentification with comprehensive JSON processing"""
        try:
            if not mcid_data:
                return {"error": "No MCID data to deidentify"}
            
            # Enhanced JSON processing - handle both 'body' and direct data
            if 'body' in mcid_data:
                raw_mcid_data = mcid_data['body']
            else:
                raw_mcid_data = mcid_data
            
            # Deep copy and comprehensively deidentify the entire JSON structure
            logger.info("🔒 Starting comprehensive MCID data deidentification...")
            deidentified_mcid_data = self._comprehensive_deidentify_json(
                copy.deepcopy(raw_mcid_data), 
                data_type="mcid"
            )
            
            result = {
                "mcid_data": deidentified_mcid_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "deidentification_level": "comprehensive_nested",
                "total_fields_processed": self._count_total_fields(deidentified_mcid_data),
                "deidentification_stats": self._get_deidentification_stats()
            }
            
            logger.info("✅ Successfully deidentified comprehensive MCID JSON structure")
            logger.info(f"📊 Processed {result['total_fields_processed']} total fields")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced MCID deidentification: {e}")
            return {"error": f"Enhanced MCID deidentification failed: {str(e)}"}
        """Enhanced pharmacy data deidentification with comprehensive JSON processing"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data to deidentify"}
            
            # Enhanced JSON processing - handle both 'body' and direct data
            if 'body' in pharmacy_data:
                raw_pharmacy_data = pharmacy_data['body']
            else:
                raw_pharmacy_data = pharmacy_data
            
            # Deep copy and comprehensively deidentify the entire JSON structure
            logger.info("🔒 Starting comprehensive pharmacy data deidentification...")
            deidentified_pharmacy_data = self._comprehensive_deidentify_json(
                copy.deepcopy(raw_pharmacy_data), 
                data_type="pharmacy"
            )
            
            result = {
                "pharmacy_data": deidentified_pharmacy_data,
                "original_structure_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "deidentification_level": "comprehensive_nested",
                "total_fields_processed": self._count_total_fields(deidentified_pharmacy_data),
                "deidentification_stats": self._get_deidentification_stats()
            }
            
            logger.info("✅ Successfully deidentified comprehensive pharmacy JSON structure")
            logger.info(f"📊 Processed {result['total_fields_processed']} total fields")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced pharmacy deidentification: {e}")
            return {"error": f"Enhanced deidentification failed: {str(e)}"}
    
    def _comprehensive_deidentify_json(self, data: Any, data_type: str = "general", path: str = "") -> Any:
        """Comprehensive deidentification of entire JSON structure including deeply nested objects"""
        try:
            if isinstance(data, dict):
                deidentified_dict = {}
                for key, value in data.items():
                    # Deidentify the key if it contains PII indicators
                    clean_key = self._deidentify_string(str(key)) if isinstance(key, str) else key
                    
                    # Create new path for tracking
                    new_path = f"{path}.{key}" if path else str(key)
                    
                    # Recursively process the value
                    deidentified_dict[clean_key] = self._comprehensive_deidentify_json(
                        value, data_type, new_path
                    )
                    
                return deidentified_dict
                
            elif isinstance(data, list):
                # Process list recursively with index tracking
                deidentified_list = []
                for i, item in enumerate(data):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    deidentified_list.append(
                        self._comprehensive_deidentify_json(item, data_type, new_path)
                    )
                return deidentified_list
                
            elif isinstance(data, str):
                # Comprehensive string deidentification
                return self._advanced_deidentify_string(data, data_type, path)
                
            elif isinstance(data, (int, float)):
                # Handle numeric data that might be PII
                return self._deidentify_numeric_data(data, path)
                
            else:
                # Return primitive types as-is (bool, None, etc.)
                return data
                
        except Exception as e:
            logger.warning(f"Error in comprehensive deidentification at path {path}: {e}")
            return data  # Return original data if deidentification fails
    
    def _advanced_deidentify_string(self, data: str, data_type: str, path: str) -> str:
        """Advanced string deidentification with context awareness"""
        try:
            if not isinstance(data, str) or not data.strip():
                return data
            
            deidentified = str(data)
            
            # Track what was deidentified for stats
            original_length = len(deidentified)
            
            # Apply all PII patterns
            for pii_type, patterns in self.pii_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, deidentified, re.IGNORECASE):
                        mask = f"[{pii_type.upper()}_MASKED]"
                        deidentified = re.sub(pattern, mask, deidentified, flags=re.IGNORECASE)
                        self._update_deidentification_stats(pii_type, path)
            
            # Preserve medical codes and terms
            for preserve_pattern in self.medical_preserve_patterns:
                # This ensures medical codes are not accidentally masked
                matches = re.findall(preserve_pattern, data, re.IGNORECASE)
                for match in matches:
                    # If the match was masked, restore it
                    deidentified = deidentified.replace('[NAMES_MASKED]', match)
            
            # Context-specific deidentification
            if data_type == "medical":
                deidentified = self._deidentify_medical_specific(deidentified, path)
            elif data_type == "pharmacy":
                deidentified = self._deidentify_pharmacy_specific(deidentified, path)
            
            return deidentified
            
        except Exception as e:
            logger.warning(f"Error in advanced string deidentification: {e}")
            return data
    
    def _deidentify_medical_specific(self, data: str, path: str) -> str:
        """Medical-specific deidentification patterns"""
        # Provider names in medical data
        data = re.sub(r'\bDr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[PROVIDER_MASKED]', data)
        data = re.sub(r'\bPhysician:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b', 'Physician: [PROVIDER_MASKED]', data)
        
        # Hospital/facility names
        data = re.sub(r'\b[A-Z][a-z]+\s+(Hospital|Medical Center|Clinic|Healthcare)\b', '[FACILITY_MASKED]', data)
        
        return data
    
    def _deidentify_pharmacy_specific(self, data: str, path: str) -> str:
        """Pharmacy-specific deidentification patterns"""
        # Pharmacist names
        data = re.sub(r'\bPharmacist:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b', 'Pharmacist: [PHARMACIST_MASKED]', data)
        
        # Pharmacy names (but preserve drug names)
        if 'pharmacy' in path.lower() and 'drug' not in path.lower():
            data = re.sub(r'\b[A-Z][a-z]+\s+Pharmacy\b', '[PHARMACY_MASKED]', data)
        
        return data
    
    def _deidentify_numeric_data(self, data: Union[int, float], path: str) -> Union[int, float]:
        """Deidentify numeric data that might be PII"""
        # Check if this could be a SSN (9 digits)
        if isinstance(data, int) and 100000000 <= data <= 999999999:
            # If path suggests this is a SSN, mask it
            if any(ssn_indicator in path.lower() for ssn_indicator in ['ssn', 'social', 'security']):
                return 123456789  # Generic masked SSN
        
        # Check if this could be a phone number (10 digits)
        if isinstance(data, int) and 1000000000 <= data <= 9999999999:
            if any(phone_indicator in path.lower() for phone_indicator in ['phone', 'tel', 'mobile']):
                return 5551234567  # Generic masked phone
        
        return data
    
    def _count_total_fields(self, data: Any) -> int:
        """Count total number of fields processed"""
        if isinstance(data, dict):
            return len(data) + sum(self._count_total_fields(v) for v in data.values())
        elif isinstance(data, list):
            return sum(self._count_total_fields(item) for item in data)
        else:
            return 1
    
    def _update_deidentification_stats(self, pii_type: str, path: str):
        """Update deidentification statistics"""
        if not hasattr(self, '_deidentification_stats'):
            self._deidentification_stats = {}
        
        if pii_type not in self._deidentification_stats:
            self._deidentification_stats[pii_type] = {
                'count': 0,
                'paths': []
            }
        
        self._deidentification_stats[pii_type]['count'] += 1
        self._deidentification_stats[pii_type]['paths'].append(path)
    
    def _get_deidentification_stats(self) -> Dict[str, Any]:
        """Get deidentification statistics"""
        return getattr(self, '_deidentification_stats', {})
    
    def extract_medical_fields_with_llm_meanings(self, deidentified_medical: Dict[str, Any], llm_caller_func) -> Dict[str, Any]:
        """Enhanced medical field extraction with LLM-powered code meanings and date extraction"""
        extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set(),
                "dates_extracted": 0
            },
            "llm_enhanced": True
        }
        
        try:
            logger.info("🔍 Starting enhanced medical field extraction with LLM meanings...")
            
            medical_data = deidentified_medical.get("medical_data", {})
            if not medical_data:
                logger.warning("No medical data found in deidentified medical data")
                return extraction_result
            
            # Enhanced recursive extraction from the entire JSON structure
            self._enhanced_recursive_medical_extraction_with_llm(medical_data, extraction_result, llm_caller_func)
            
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_service_codes"] = list(
                extraction_result["extraction_summary"]["unique_service_codes"]
            )
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = list(
                extraction_result["extraction_summary"]["unique_diagnosis_codes"]
            )
            
            logger.info(f"📋 Enhanced medical extraction with LLM completed: "
                       f"{extraction_result['extraction_summary']['total_hlth_srvc_records']} health service records, "
                       f"{extraction_result['extraction_summary']['total_diagnosis_codes']} diagnosis codes, "
                       f"{extraction_result['extraction_summary']['dates_extracted']} dates extracted")
            
        except Exception as e:
            logger.error(f"Error in enhanced medical field extraction with LLM: {e}")
            extraction_result["error"] = f"Enhanced medical extraction with LLM failed: {str(e)}"
        
        return extraction_result
    
    def _enhanced_recursive_medical_extraction_with_llm(self, data: Any, result: Dict[str, Any], llm_caller_func, path: str = ""):
        """Enhanced recursive search with LLM meanings and date extraction"""
        if isinstance(data, dict):
            current_record = {}
            
            # Extract health service code with multiple possible field names
            service_code_fields = ['hlth_srvc_cd', 'health_service_code', 'service_code', 'procedure_code']
            for field in service_code_fields:
                if field in data and data[field]:
                    service_code = str(data[field])
                    current_record["hlth_srvc_cd"] = service_code
                    result["extraction_summary"]["unique_service_codes"].add(service_code)
                    
                    # Get LLM meaning for service code
                    try:
                        service_meaning = self._get_llm_code_meaning(service_code, "medical_service", llm_caller_func)
                        current_record["hlth_srvc_cd_meaning"] = service_meaning
                    except Exception as e:
                        current_record["hlth_srvc_cd_meaning"] = f"Error getting meaning: {str(e)}"
                    break
            
            # Extract claim received date
            date_fields = ['CLM_RCVD_DT', 'clm_rcvd_dt', 'claim_received_date', 'received_date']
            for field in date_fields:
                if field in data and data[field]:
                    current_record["claim_received_date"] = str(data[field])
                    result["extraction_summary"]["dates_extracted"] += 1
                    break
            
            diagnosis_codes = []
            
            # Enhanced diagnosis code extraction with LLM meanings
            diag_combo_fields = ['diag_1_50_cd', 'diagnosis_codes', 'icd_codes']
            for field in diag_combo_fields:
                if field in data and data[field]:
                    diag_value = str(data[field]).strip()
                    if diag_value and diag_value.lower() not in ['null', 'none', '']:
                        # Enhanced separation
                        separators = [',', ';', '|', '\n']
                        individual_codes = [diag_value]
                        
                        for sep in separators:
                            temp_codes = []
                            for code in individual_codes:
                                temp_codes.extend([c.strip() for c in code.split(sep) if c.strip()])
                            individual_codes = temp_codes
                        
                        for i, code in enumerate(individual_codes, 1):
                            if code and code.lower() not in ['null', 'none', '']:
                                # Get LLM meaning for diagnosis code
                                try:
                                    diag_meaning = self._get_llm_code_meaning(code, "diagnosis", llm_caller_func)
                                except Exception as e:
                                    diag_meaning = f"Error getting meaning: {str(e)}"
                                
                                diagnosis_codes.append({
                                    "code": code,
                                    "position": i,
                                    "source": f"{field} (separated)",
                                    "path": path,
                                    "llm_meaning": diag_meaning
                                })
                                result["extraction_summary"]["unique_diagnosis_codes"].add(code)
            
            # Individual diagnosis fields with LLM meanings
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        # Get LLM meaning
                        try:
                            diag_meaning = self._get_llm_code_meaning(diag_code, "diagnosis", llm_caller_func)
                        except Exception as e:
                            diag_meaning = f"Error getting meaning: {str(e)}"
                        
                        diagnosis_codes.append({
                            "code": diag_code,
                            "position": i,
                            "source": f"individual field ({diag_key})",
                            "path": path,
                            "llm_meaning": diag_meaning
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
                self._enhanced_recursive_medical_extraction_with_llm(value, result, llm_caller_func, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_recursive_medical_extraction_with_llm(item, result, llm_caller_func, new_path)
    
    def extract_pharmacy_fields_with_llm_meanings_from_raw(self, raw_pharmacy: Dict[str, Any], llm_caller_func) -> Dict[str, Any]:
        """Enhanced pharmacy field extraction with LLM-powered NDC/label meanings and date extraction from raw data"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set(),
                "dates_extracted": 0
            },
            "llm_enhanced": True,
            "data_source": "raw_pharmacy"
        }
        
        try:
            logger.info("🔍 Starting enhanced pharmacy field extraction with LLM meanings from raw data...")
            
            if not raw_pharmacy:
                logger.warning("No raw pharmacy data found")
                return extraction_result
            
            # Handle both 'body' and direct data from raw pharmacy
            if 'body' in raw_pharmacy:
                pharmacy_data = raw_pharmacy['body']
            else:
                pharmacy_data = raw_pharmacy
            
            self._enhanced_recursive_pharmacy_extraction_with_llm(pharmacy_data, extraction_result, llm_caller_func)
            
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
            
            logger.info(f"💊 Enhanced pharmacy extraction from raw data with LLM completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records, "
                       f"{extraction_result['extraction_summary']['dates_extracted']} dates extracted")
            
        except Exception as e:
            logger.error(f"Error in enhanced pharmacy field extraction from raw data with LLM: {e}")
            extraction_result["error"] = f"Enhanced pharmacy extraction from raw data with LLM failed: {str(e)}"
        
        return extraction_result
        """Enhanced pharmacy field extraction with LLM-powered NDC/label meanings and date extraction"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set(),
                "dates_extracted": 0
            },
            "llm_enhanced": True
        }
        
        try:
            logger.info("🔍 Starting enhanced pharmacy field extraction with LLM meanings...")
            
            pharmacy_data = deidentified_pharmacy.get("pharmacy_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy data found in deidentified pharmacy data")
                return extraction_result
            
            self._enhanced_recursive_pharmacy_extraction_with_llm(pharmacy_data, extraction_result, llm_caller_func)
            
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
            
            logger.info(f"💊 Enhanced pharmacy extraction with LLM completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records, "
                       f"{extraction_result['extraction_summary']['dates_extracted']} dates extracted")
            
        except Exception as e:
            logger.error(f"Error in enhanced pharmacy field extraction with LLM: {e}")
            extraction_result["error"] = f"Enhanced pharmacy extraction with LLM failed: {str(e)}"
        
        return extraction_result
    
    def _enhanced_recursive_pharmacy_extraction_with_llm(self, data: Any, result: Dict[str, Any], llm_caller_func, path: str = ""):
        """Enhanced recursive search for pharmacy fields with LLM meanings and date extraction"""
        if isinstance(data, dict):
            current_record = {}
            
            # Enhanced NDC field detection with LLM meaning
            ndc_field_names = [
                'ndc', 'ndc_code', 'ndc_number', 'national_drug_code', 
                'drug_code', 'product_ndc', 'ndc_id'
            ]
            ndc_found = False
            for field in ndc_field_names:
                if field in data and data[field]:
                    ndc_code = str(data[field])
                    current_record["ndc"] = ndc_code
                    result["extraction_summary"]["unique_ndc_codes"].add(ndc_code)
                    
                    # Get LLM meaning for NDC code
                    try:
                        ndc_meaning = self._get_llm_code_meaning(ndc_code, "ndc", llm_caller_func)
                        current_record["ndc_llm_meaning"] = ndc_meaning
                    except Exception as e:
                        current_record["ndc_llm_meaning"] = f"Error getting meaning: {str(e)}"
                    
                    ndc_found = True
                    break
            
            # Enhanced label name field detection with LLM description
            label_field_names = [
                'lbl_nm', 'label_name', 'drug_name', 'medication_name', 
                'product_name', 'brand_name', 'generic_name', 'drug_label',
                'medication', 'product_label'
            ]
            label_found = False
            for field in label_field_names:
                if field in data and data[field]:
                    label_name = str(data[field])
                    current_record["lbl_nm"] = label_name
                    result["extraction_summary"]["unique_label_names"].add(label_name)
                    
                    # Get LLM description for label name
                    try:
                        label_description = self._get_llm_code_meaning(label_name, "medication_label", llm_caller_func)
                        current_record["lbl_nm_llm_description"] = label_description
                    except Exception as e:
                        current_record["lbl_nm_llm_description"] = f"Error getting description: {str(e)}"
                    
                    label_found = True
                    break
            
            # Extract prescription filled date
            date_fields = ['RX_FILLED_DT', 'rx_filled_dt', 'prescription_filled_date', 'filled_date']
            for field in date_fields:
                if field in data and data[field]:
                    current_record["prescription_filled_date"] = str(data[field])
                    result["extraction_summary"]["dates_extracted"] += 1
                    break
            
            # Extract additional pharmacy fields
            additional_fields = ['strength', 'dosage', 'quantity', 'days_supply', 'refills']
            for field in additional_fields:
                if field in data and data[field]:
                    current_record[field] = data[field]
            
            if ndc_found or label_found:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
            
            # Continue recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._enhanced_recursive_pharmacy_extraction_with_llm(value, result, llm_caller_func, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_recursive_pharmacy_extraction_with_llm(item, result, llm_caller_func, new_path)
    
    def _get_llm_code_meaning(self, code: str, code_type: str, llm_caller_func) -> str:
        """Get LLM-powered meaning/description for medical/pharmacy codes"""
        try:
            if code_type == "medical_service":
                prompt = f"Explain what medical service code '{code}' means. Provide a brief, professional explanation (1-2 sentences)."
            elif code_type == "diagnosis":
                prompt = f"Explain what diagnosis code '{code}' means. If it's an ICD-10 code, provide the medical condition it represents (1-2 sentences)."
            elif code_type == "ndc":
                prompt = f"Explain what NDC code '{code}' represents. Provide information about the medication/product (1-2 sentences)."
            elif code_type == "medication_label":
                prompt = f"Provide a brief medical description of medication '{code}'. Include what it's used for and key information (1-2 sentences)."
            else:
                prompt = f"Provide a brief explanation of medical code '{code}' (1-2 sentences)."
            
            response = llm_caller_func(prompt)
            
            # Clean up the response
            if response and not response.startswith("Error"):
                # Take first 2 sentences to keep it concise
                sentences = response.split('.')[:2]
                return '.'.join(sentences).strip() + '.' if sentences else response[:200]
            else:
                return f"Unable to get meaning for {code_type} code: {code}"
                
        except Exception as e:
            logger.warning(f"Error getting LLM meaning for {code}: {str(e)}")
            return f"Error getting meaning for {code_type} code: {code}"
        """Enhanced medical field extraction with better comma-separated diagnosis handling"""
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
            logger.info("🔍 Starting enhanced medical field extraction...")
            
            medical_data = deidentified_medical.get("medical_data", {})
            if not medical_data:
                logger.warning("No medical data found in deidentified medical data")
                return extraction_result
            
            # Enhanced recursive extraction from the entire JSON structure
            self._enhanced_recursive_medical_extraction(medical_data, extraction_result)
            
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_service_codes"] = list(
                extraction_result["extraction_summary"]["unique_service_codes"]
            )
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = list(
                extraction_result["extraction_summary"]["unique_diagnosis_codes"]
            )
            
            logger.info(f"📋 Enhanced medical extraction completed: "
                       f"{extraction_result['extraction_summary']['total_hlth_srvc_records']} health service records, "
                       f"{extraction_result['extraction_summary']['total_diagnosis_codes']} diagnosis codes")
            
        except Exception as e:
            logger.error(f"Error in enhanced medical field extraction: {e}")
            extraction_result["error"] = f"Enhanced medical extraction failed: {str(e)}"
        
        return extraction_result
    
    def _enhanced_recursive_medical_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive search for medical fields in complex nested data structures"""
        if isinstance(data, dict):
            current_record = {}
            
            # Extract health service code with multiple possible field names
            service_code_fields = ['hlth_srvc_cd', 'health_service_code', 'service_code', 'procedure_code']
            for field in service_code_fields:
                if field in data and data[field]:
                    current_record["hlth_srvc_cd"] = data[field]
                    result["extraction_summary"]["unique_service_codes"].add(str(data[field]))
                    break
            
            diagnosis_codes = []
            
            # Enhanced diagnosis code extraction
            # 1. Handle comma-separated diagnosis codes in diag_1_50_cd field
            diag_combo_fields = ['diag_1_50_cd', 'diagnosis_codes', 'icd_codes']
            for field in diag_combo_fields:
                if field in data and data[field]:
                    diag_value = str(data[field]).strip()
                    if diag_value and diag_value.lower() not in ['null', 'none', '']:
                        # Enhanced comma/semicolon/pipe separation
                        separators = [',', ';', '|', '\n']
                        individual_codes = [diag_value]  # Start with the whole value
                        
                        for sep in separators:
                            temp_codes = []
                            for code in individual_codes:
                                temp_codes.extend([c.strip() for c in code.split(sep) if c.strip()])
                            individual_codes = temp_codes
                        
                        for i, code in enumerate(individual_codes, 1):
                            if code and code.lower() not in ['null', 'none', '']:
                                diagnosis_codes.append({
                                    "code": code,
                                    "position": i,
                                    "source": f"{field} (separated)",
                                    "path": path
                                })
                                result["extraction_summary"]["unique_diagnosis_codes"].add(code)
            
            # 2. Handle individual diagnosis fields (diag_1_cd, diag_2_cd, etc.)
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        diagnosis_codes.append({
                            "code": diag_code,
                            "position": i,
                            "source": f"individual field ({diag_key})",
                            "path": path
                        })
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)
            
            # 3. Look for other diagnosis-related fields
            other_diag_fields = ['primary_diagnosis', 'secondary_diagnosis', 'icd10_code', 'icd_10']
            for field in other_diag_fields:
                if field in data and data[field]:
                    diag_code = str(data[field]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        diagnosis_codes.append({
                            "code": diag_code,
                            "position": len(diagnosis_codes) + 1,
                            "source": f"other field ({field})",
                            "path": path
                        })
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)
            
            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
            
            if current_record:
                current_record["data_path"] = path
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1
            
            # Continue recursive search in all nested objects
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._enhanced_recursive_medical_extraction(value, result, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_recursive_medical_extraction(item, result, new_path)
    
    def extract_pharmacy_fields(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pharmacy field extraction with multiple NDC and label name field support"""
        extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set()
            }
        }
        
        try:
            logger.info("🔍 Starting enhanced pharmacy field extraction...")
            
            pharmacy_data = deidentified_pharmacy.get("pharmacy_data", {})
            if not pharmacy_data:
                logger.warning("No pharmacy data found in deidentified pharmacy data")
                return extraction_result
            
            self._enhanced_recursive_pharmacy_extraction(pharmacy_data, extraction_result)
            
            # Convert sets to lists for JSON serialization
            extraction_result["extraction_summary"]["unique_ndc_codes"] = list(
                extraction_result["extraction_summary"]["unique_ndc_codes"]
            )
            extraction_result["extraction_summary"]["unique_label_names"] = list(
                extraction_result["extraction_summary"]["unique_label_names"]
            )
            
            logger.info(f"💊 Enhanced pharmacy extraction completed: "
                       f"{extraction_result['extraction_summary']['total_ndc_records']} NDC records")
            
        except Exception as e:
            logger.error(f"Error in enhanced pharmacy field extraction: {e}")
            extraction_result["error"] = f"Enhanced pharmacy extraction failed: {str(e)}"
        
        return extraction_result
    
    def _enhanced_recursive_pharmacy_extraction(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive search for pharmacy fields in complex nested data structures"""
        if isinstance(data, dict):
            current_record = {}
            
            # Enhanced NDC field detection
            ndc_field_names = [
                'ndc', 'ndc_code', 'ndc_number', 'national_drug_code', 
                'drug_code', 'product_ndc', 'ndc_id'
            ]
            ndc_found = False
            for field in ndc_field_names:
                if field in data and data[field]:
                    current_record["ndc"] = data[field]
                    result["extraction_summary"]["unique_ndc_codes"].add(str(data[field]))
                    ndc_found = True
                    break
            
            # Enhanced label name field detection
            label_field_names = [
                'lbl_nm', 'label_name', 'drug_name', 'medication_name', 
                'product_name', 'brand_name', 'generic_name', 'drug_label',
                'medication', 'product_label'
            ]
            label_found = False
            for field in label_field_names:
                if field in data and data[field]:
                    current_record["lbl_nm"] = data[field]
                    result["extraction_summary"]["unique_label_names"].add(str(data[field]))
                    label_found = True
                    break
            
            # Also extract additional pharmacy-related fields
            additional_fields = ['strength', 'dosage', 'quantity', 'days_supply', 'refills']
            for field in additional_fields:
                if field in data and data[field]:
                    current_record[field] = data[field]
            
            if ndc_found or label_found:
                current_record["data_path"] = path
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
            
            # Continue recursive search in all nested objects
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._enhanced_recursive_pharmacy_extraction(value, result, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_recursive_pharmacy_extraction(item, result, new_path)
    
    def extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any], 
                                        pharmacy_extraction: Dict[str, Any],
                                        medical_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced health entity extraction using comprehensive data analysis"""
        entities = {
            "diabetics": "no",
            "age_group": "unknown", 
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": [],
            "risk_factors": [],
            "chronic_conditions": []
        }
        
        try:
            logger.info("🎯 Starting enhanced health entity extraction...")
            
            # Analyze original pharmacy data
            if pharmacy_data:
                data_str = json.dumps(pharmacy_data).lower()
                self._enhanced_analyze_pharmacy_for_entities(data_str, entities)
            
            # Analyze structured pharmacy extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                self._enhanced_analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
            
            # Analyze medical extraction for conditions
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                self._enhanced_analyze_medical_extraction_for_entities(medical_extraction, entities)
            
            entities["analysis_details"].append(
                f"Enhanced analysis sources: Pharmacy data, "
                f"{len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, "
                f"{len(medical_extraction.get('hlth_srvc_records', []))} medical records"
            )
            
            logger.info(f"✅ Enhanced entity extraction completed: {len(entities['medical_conditions'])} conditions identified")
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in enhanced entity extraction: {str(e)}")
        
        return entities
    
    def _enhanced_analyze_pharmacy_for_entities(self, data_str: str, entities: Dict[str, Any]):
        """Enhanced pharmacy data analysis for health entities"""
        # Expanded diabetes medication detection
        diabetes_keywords = [
            'insulin', 'metformin', 'glipizide', 'diabetes', 'diabetic', 
            'glucophage', 'lantus', 'humalog', 'novolog', 'levemir',
            'glyburide', 'pioglitazone', 'sitagliptin', 'glimepiride',
            'januvia', 'victoza', 'ozempic', 'trulicity'
        ]
        for keyword in diabetes_keywords:
            if keyword in data_str:
                entities["diabetics"] = "yes"
                entities["analysis_details"].append(f"Diabetes indicator found in pharmacy data: {keyword}")
                entities["chronic_conditions"].append("Diabetes")
                break
        
        # Enhanced medication-based age group detection
        senior_medications = [
            'aricept', 'warfarin', 'lisinopril', 'atorvastatin', 'metoprolol',
            'furosemide', 'amlodipine', 'simvastatin', 'donepezil', 'rivaroxaban'
        ]
        adult_medications = [
            'adderall', 'vyvanse', 'accutane', 'birth control', 'isotretinoin'
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
        
        # Enhanced blood pressure medication detection
        bp_medications = [
            'lisinopril', 'amlodipine', 'metoprolol', 'losartan', 'atenolol',
            'carvedilol', 'valsartan', 'enalapril', 'hydrochlorothiazide'
        ]
        for med in bp_medications:
            if med in data_str:
                entities["blood_pressure"] = "managed"
                entities["analysis_details"].append(f"Blood pressure medication found: {med}")
                entities["chronic_conditions"].append("Hypertension")
                break
    
    def _enhanced_analyze_pharmacy_extraction_for_entities(self, pharmacy_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Enhanced analysis of structured pharmacy extraction"""
        ndc_records = pharmacy_extraction.get("ndc_records", [])
        
        for record in ndc_records:
            ndc = record.get("ndc", "")
            lbl_nm = record.get("lbl_nm", "")
            
            if lbl_nm:
                entities["medications_identified"].append({
                    "ndc": ndc,
                    "label_name": lbl_nm,
                    "path": record.get("data_path", ""),
                    "additional_fields": {k: v for k, v in record.items() 
                                       if k not in ['ndc', 'lbl_nm', 'data_path']}
                })
                
                lbl_lower = lbl_nm.lower()
                
                # Enhanced condition detection from medication names
                if any(word in lbl_lower for word in ['insulin', 'metformin', 'glucophage', 'diabetes']):
                    entities["diabetics"] = "yes"
                    entities["analysis_details"].append(f"Diabetes medication found in extraction: {lbl_nm}")
                
                if any(word in lbl_lower for word in ['lisinopril', 'amlodipine', 'metoprolol', 'blood pressure']):
                    entities["blood_pressure"] = "managed"
                    entities["analysis_details"].append(f"Blood pressure medication found in extraction: {lbl_nm}")
                
                # Additional condition detection
                if any(word in lbl_lower for word in ['statin', 'lipitor', 'crestor', 'cholesterol']):
                    entities["chronic_conditions"].append("High Cholesterol")
                    entities["analysis_details"].append(f"Cholesterol medication found: {lbl_nm}")
    
    def _enhanced_analyze_medical_extraction_for_entities(self, medical_extraction: Dict[str, Any], entities: Dict[str, Any]):
        """Enhanced analysis of medical codes for comprehensive health conditions"""
        hlth_srvc_records = medical_extraction.get("hlth_srvc_records", [])
        
        # Enhanced condition mappings with more comprehensive ICD-10 codes
        condition_mappings = {
            "diabetes": ["E10", "E11", "E12", "E13", "E14", "E08", "E09"],
            "hypertension": ["I10", "I11", "I12", "I13", "I15", "I16"],
            "smoking": ["Z72.0", "F17", "Z87.891"],
            "alcohol": ["F10", "Z72.1", "Z87.891"],
            "heart_disease": ["I20", "I21", "I22", "I23", "I24", "I25"],
            "obesity": ["E66", "Z68"],
            "depression": ["F32", "F33", "F34", "F39"],
            "anxiety": ["F40", "F41", "F42", "F43"],
            "copd": ["J44", "J43", "J42"],
            "asthma": ["J45", "J46"]
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
                                entities["chronic_conditions"].append("Diabetes")
                            elif condition == "hypertension":
                                entities["blood_pressure"] = "diagnosed"
                                entities["medical_conditions"].append(f"Hypertension (ICD-10: {diag_code})")
                                entities["chronic_conditions"].append("Hypertension")
                            elif condition == "smoking":
                                entities["smoking"] = "yes"
                                entities["medical_conditions"].append(f"Smoking (ICD-10: {diag_code})")
                                entities["risk_factors"].append("Smoking")
                            elif condition == "heart_disease":
                                entities["risk_factors"].append("Heart Disease History")
                                entities["medical_conditions"].append(f"Heart Disease (ICD-10: {diag_code})")
                            
                            entities["analysis_details"].append(
                                f"Medical condition identified from ICD-10 {diag_code}: {condition}"
                            )
    
    def prepare_complete_deidentified_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare COMPLETE context including MCID, deidentified medical, and raw pharmacy - NO truncation"""
        try:
            context_sections = []
            
            # 1. Patient Overview (small, keep as is)
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")
            
            # 2. COMPLETE DEIDENTIFIED MCID DATA - ENTIRE JSON STRUCTURE
            deidentified_mcid = chat_context.get("deidentified_mcid", {})
            if deidentified_mcid:
                logger.info("🆔 Including COMPLETE deidentified MCID JSON - no truncation")
                
                # Get the ENTIRE mcid_data JSON - this is the complete deidentified structure
                complete_mcid_json = deidentified_mcid.get('mcid_data', {})
                
                # Convert ENTIRE MCID JSON to string with full detail
                complete_mcid_str = json.dumps(complete_mcid_json, indent=2, default=str, ensure_ascii=False)
                
                # Add MCID metadata
                mcid_context = f"""COMPLETE DEIDENTIFIED MCID DATA (ENTIRE JSON):
Deidentification Level: {deidentified_mcid.get('deidentification_level', 'standard')}
Total Fields Processed: {deidentified_mcid.get('total_fields_processed', 0)}

COMPLETE MCID JSON DATA (All nested structures included):
{complete_mcid_str}"""
                
                context_sections.append(mcid_context)
            
            # 3. COMPLETE DEIDENTIFIED MEDICAL DATA - ENTIRE JSON STRUCTURE
            deidentified_medical = chat_context.get("deidentified_medical", {})
            if deidentified_medical:
                logger.info("📋 Including COMPLETE deidentified medical JSON - no truncation")
                
                # Get the ENTIRE medical_data JSON - this is the complete deidentified structure
                complete_medical_json = deidentified_medical.get('medical_data', {})
                
                # Convert ENTIRE medical JSON to string with full detail
                complete_medical_str = json.dumps(complete_medical_json, indent=2, default=str, ensure_ascii=False)
                
                # Add patient metadata
                medical_context = f"""COMPLETE DEIDENTIFIED MEDICAL DATA (ENTIRE JSON):
Patient Name: {deidentified_medical.get('src_mbr_first_nm', 'N/A')} {deidentified_medical.get('src_mbr_last_nm', 'N/A')}
Patient Age: {deidentified_medical.get('src_mbr_age', 'N/A')}
Patient Zip: {deidentified_medical.get('src_mbr_zip_cd', 'N/A')}
Deidentification Level: {deidentified_medical.get('deidentification_level', 'standard')}
Total Fields Processed: {deidentified_medical.get('total_fields_processed', 0)}

COMPLETE MEDICAL JSON DATA (All nested structures included):
{complete_medical_str}"""
                
                context_sections.append(medical_context)
            
            # 4. COMPLETE RAW PHARMACY DATA - ENTIRE JSON STRUCTURE (NOT DEIDENTIFIED)
            raw_pharmacy = chat_context.get("raw_pharmacy", {})
            if raw_pharmacy:
                logger.info("💊 Including COMPLETE raw pharmacy JSON - no truncation, no deidentification")
                
                # Get the ENTIRE pharmacy JSON - this is the raw data
                if 'body' in raw_pharmacy:
                    complete_pharmacy_json = raw_pharmacy['body']
                else:
                    complete_pharmacy_json = raw_pharmacy
                
                # Convert ENTIRE pharmacy JSON to string with full detail
                complete_pharmacy_str = json.dumps(complete_pharmacy_json, indent=2, default=str, ensure_ascii=False)
                
                # Add pharmacy metadata
                pharmacy_context = f"""COMPLETE RAW PHARMACY DATA (ENTIRE JSON - NOT DEIDENTIFIED):
Data Type: Raw (No deidentification applied)
Status: Complete pharmacy records available

COMPLETE PHARMACY JSON DATA (All nested structures included):
{complete_pharmacy_str}"""
                
                context_sections.append(pharmacy_context)
            
            # 5. COMPLETE Structured Extractions with LLM meanings (also full data)
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                extraction_str = json.dumps(medical_extraction, indent=2, default=str)
                context_sections.append(f"COMPLETE MEDICAL DATA EXTRACTIONS (WITH LLM MEANINGS):\n{extraction_str}")
            
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                extraction_str = json.dumps(pharmacy_extraction, indent=2, default=str)
                context_sections.append(f"COMPLETE PHARMACY DATA EXTRACTIONS (WITH LLM MEANINGS FROM RAW DATA):\n{extraction_str}")
            
            # 6. Complete Entity Extraction
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                entity_str = json.dumps(entity_extraction, indent=2, default=str)
                context_sections.append(f"COMPLETE HEALTH ENTITIES:\n{entity_str}")
            
            # 7. Heart Attack Prediction
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                prediction_str = json.dumps(heart_attack_prediction, indent=2, default=str)
                context_sections.append(f"HEART ATTACK PREDICTION:\n{prediction_str}")
            
            # 8. Clinical Analysis (full text)
            health_trajectory = chat_context.get("health_trajectory", "")
            if health_trajectory:
                context_sections.append(f"HEALTH TRAJECTORY ANALYSIS:\n{health_trajectory}")
            
            final_summary = chat_context.get("final_summary", "")
            if final_summary:
                context_sections.append(f"CLINICAL SUMMARY:\n{final_summary}")
            
            # Join all sections - COMPLETE DATA INCLUDING MCID + RAW PHARMACY, NO TRUNCATION
            complete_context = "\n\n" + ("\n" + "="*100 + "\n").join(context_sections)
            
            logger.info(f"📋 Prepared COMPLETE context including MCID + raw pharmacy with {len(context_sections)} sections")
            logger.info(f"📊 Total COMPLETE context length: {len(complete_context)} characters")
            logger.info("🔍 LLM will have access to ENTIRE deidentified JSON structures + raw pharmacy data")
            
            return complete_context
            
        except Exception as e:
            logger.error(f"Error preparing COMPLETE context with MCID + raw pharmacy: {e}")
            return "Error: Could not prepare complete context including MCID and raw pharmacy data."

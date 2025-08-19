import json
import re
from datetime import datetime, date
from typing import Dict, Any, List
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class HealthDataProcessor:
    """Enhanced data processor with comprehensive debug logging for meaning generation"""
 
    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        logger.info("🔧 HealthDataProcessor initialized")
        
        # Debug API integrator
        if self.api_integrator:
            logger.info("✅ API integrator provided to data processor")
            # Check if isolated method exists
            if hasattr(self.api_integrator, 'call_llm_isolated'):
                logger.info("✅ call_llm_isolated method found in API integrator")
                # Test it immediately
                try:
                    test_result = self.api_integrator.call_llm_isolated("Test message for initialization")
                    if test_result != "Explanation unavailable":
                        logger.info("✅ call_llm_isolated test successful during initialization")
                    else:
                        logger.warning("⚠️ call_llm_isolated test returned 'Explanation unavailable'")
                except Exception as e:
                    logger.error(f"❌ call_llm_isolated test failed during initialization: {e}")
            else:
                logger.error("❌ call_llm_isolated method NOT found in API integrator")
        else:
            logger.warning("⚠️ No API integrator provided to data processor")
 
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
                "src_mbr_zip_cd": patient_data.get('zip_code', '12345'),  # Preserve real ZIP code
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
        """Extract medical fields WITH COMPREHENSIVE DEBUG LOGGING AND MEANING GENERATION"""
        logger.info("🔍 ===== STARTING MEDICAL FIELD EXTRACTION WITH MEANING GENERATION =====")
        
        extraction_result = {
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
            "llm_call_status": "not_attempted",
            "debug_info": {
                "api_integrator_available": self.api_integrator is not None,
                "call_llm_isolated_available": hasattr(self.api_integrator, 'call_llm_isolated') if self.api_integrator else False,
                "codes_found": False,
                "meaning_generation_attempts": 0,
                "meaning_generation_successes": 0,
                "meaning_generation_failures": 0,
                "service_codes_processed": [],
                "diagnosis_codes_processed": []
            }
        }

        try:
            # Debug API integrator status
            logger.info(f"🔍 DEBUG: API integrator available: {extraction_result['debug_info']['api_integrator_available']}")
            logger.info(f"🔍 DEBUG: call_llm_isolated available: {extraction_result['debug_info']['call_llm_isolated_available']}")

            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("⚠️ No medical claims data found in deidentified medical data")
                return extraction_result

            # Step 1: Extract codes first
            logger.info("📋 Step 1: Extracting medical codes...")
            self._recursive_medical_extraction(medical_data, extraction_result)

            # Convert sets to lists for JSON serialization
            unique_service_codes = list(extraction_result["extraction_summary"]["unique_service_codes"])
            unique_diagnosis_codes = list(extraction_result["extraction_summary"]["unique_diagnosis_codes"])
            
            extraction_result["extraction_summary"]["unique_service_codes"] = unique_service_codes
            extraction_result["extraction_summary"]["unique_diagnosis_codes"] = unique_diagnosis_codes

            # Debug: Log codes found
            logger.info(f"📊 FOUND {len(unique_service_codes)} unique service codes: {unique_service_codes[:5]}...")
            logger.info(f"📊 FOUND {len(unique_diagnosis_codes)} unique diagnosis codes: {unique_diagnosis_codes[:5]}...")
            
            extraction_result["debug_info"]["codes_found"] = len(unique_service_codes) > 0 or len(unique_diagnosis_codes) > 0

            # Step 2: Generate meanings if API integrator is available
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated'):
                if unique_service_codes or unique_diagnosis_codes:
                    logger.info("🤖 Step 2: Starting meaning generation...")
                    extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        # Generate service code meanings
                        logger.info(f"🔍 Generating meanings for {len(unique_service_codes)} service codes...")
                        for i, service_code in enumerate(unique_service_codes[:15], 1):
                            if service_code and str(service_code).strip():
                                logger.info(f"🔍 Processing service code {i}/15: {service_code}")
                                extraction_result["debug_info"]["meaning_generation_attempts"] += 1
                                extraction_result["debug_info"]["service_codes_processed"].append(str(service_code))
                                
                                meaning = self._get_service_code_meaning_debug(str(service_code))
                                
                                if meaning and meaning != "Explanation unavailable":
                                    extraction_result["code_meanings"]["service_code_meanings"][str(service_code)] = meaning
                                    extraction_result["debug_info"]["meaning_generation_successes"] += 1
                                    logger.info(f"✅ Service code {service_code}: {meaning[:50]}...")
                                else:
                                    extraction_result["debug_info"]["meaning_generation_failures"] += 1
                                    logger.warning(f"❌ Failed to get meaning for service code: {service_code}")
                        
                        # Generate diagnosis code meanings
                        logger.info(f"🔍 Generating meanings for {len(unique_diagnosis_codes)} diagnosis codes...")
                        for i, diag_code in enumerate(unique_diagnosis_codes[:20], 1):
                            if diag_code and str(diag_code).strip():
                                logger.info(f"🔍 Processing diagnosis code {i}/20: {diag_code}")
                                extraction_result["debug_info"]["meaning_generation_attempts"] += 1
                                extraction_result["debug_info"]["diagnosis_codes_processed"].append(str(diag_code))
                                
                                meaning = self._get_diagnosis_code_meaning_debug(str(diag_code))
                                
                                if meaning and meaning != "Explanation unavailable":
                                    extraction_result["code_meanings"]["diagnosis_code_meanings"][str(diag_code)] = meaning
                                    extraction_result["debug_info"]["meaning_generation_successes"] += 1
                                    logger.info(f"✅ Diagnosis code {diag_code}: {meaning[:50]}...")
                                else:
                                    extraction_result["debug_info"]["meaning_generation_failures"] += 1
                                    logger.warning(f"❌ Failed to get meaning for diagnosis code: {diag_code}")
                        
                        # Step 3: Add meanings to individual records
                        logger.info("📋 Step 3: Adding meanings to individual records...")
                        self._add_meanings_to_medical_records(extraction_result)
                        
                        # Final status
                        total_meanings = len(extraction_result["code_meanings"]["service_code_meanings"]) + len(extraction_result["code_meanings"]["diagnosis_code_meanings"])
                        
                        if total_meanings > 0:
                            extraction_result["code_meanings_added"] = True
                            extraction_result["llm_call_status"] = "completed"
                            logger.info(f"✅ Successfully generated {total_meanings} total meanings for medical codes")
                        else:
                            extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Meaning generation completed but no meanings were generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating medical code meanings: {e}")
                        extraction_result["code_meaning_error"] = str(e)
                        extraction_result["llm_call_status"] = "failed"
                else:
                    extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No codes found to generate meanings for")
            else:
                extraction_result["llm_call_status"] = "skipped_no_api"
                if not self.api_integrator:
                    logger.error("❌ No API integrator available for meaning generation")
                elif not hasattr(self.api_integrator, 'call_llm_isolated'):
                    logger.error("❌ API integrator missing call_llm_isolated method")

            # Final summary
            logger.info(f"📋 ===== MEDICAL EXTRACTION COMPLETED =====")
            logger.info(f"  📊 Records: {extraction_result['extraction_summary']['total_hlth_srvc_records']}")
            logger.info(f"  📊 Diagnosis codes: {extraction_result['extraction_summary']['total_diagnosis_codes']}")
            logger.info(f"  🤖 LLM status: {extraction_result['llm_call_status']}")
            logger.info(f"  ✅ Meanings added: {extraction_result['code_meanings_added']}")
            logger.info(f"  📈 Success rate: {extraction_result['debug_info']['meaning_generation_successes']}/{extraction_result['debug_info']['meaning_generation_attempts']}")
            logger.info(f"  🔤 Service meanings: {len(extraction_result['code_meanings']['service_code_meanings'])}")
            logger.info(f"  🔤 Diagnosis meanings: {len(extraction_result['code_meanings']['diagnosis_code_meanings'])}")

        except Exception as e:
            logger.error(f"❌ Error in medical field extraction: {e}")
            extraction_result["error"] = f"Medical extraction failed: {str(e)}"

        return extraction_result

    def _get_service_code_meaning_debug(self, service_code: str) -> str:
        """Get service code meaning with comprehensive debug logging"""
        try:
            logger.info(f"🔍 === GENERATING SERVICE CODE MEANING FOR: {service_code} ===")
            
            if not self.api_integrator:
                logger.error("❌ No API integrator available")
                return "Explanation unavailable"
            
            if not hasattr(self.api_integrator, 'call_llm_isolated'):
                logger.error("❌ call_llm_isolated method not found")
                return "Explanation unavailable"
            
            prompt = f"Explain healthcare service code '{service_code}' in 1-2 lines. What medical service or procedure does this code represent?"
            system_msg = "You are a medical coding expert. Provide brief, accurate explanations for healthcare codes in 1-2 lines."
            
            logger.info(f"📝 Sending prompt: {prompt}")
            logger.info(f"📝 System message: {system_msg}")
            
            response = self.api_integrator.call_llm_isolated(prompt, system_msg)
            
            logger.info(f"📄 Raw response: {response}")
            
            if response and response != "Explanation unavailable":
                # Clean up the response
                lines = response.strip().split('\n')
                clean_response = ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
                logger.info(f"✅ Service code meaning generated successfully: {clean_response}")
                return clean_response
            else:
                logger.warning(f"⚠️ LLM returned unavailable response: {response}")
                return "Explanation unavailable"
                
        except Exception as e:
            logger.error(f"❌ Exception getting service code meaning: {e}")
            return "Explanation unavailable"

    def _get_diagnosis_code_meaning_debug(self, diagnosis_code: str) -> str:
        """Get diagnosis code meaning with comprehensive debug logging"""
        try:
            logger.info(f"🔍 === GENERATING DIAGNOSIS CODE MEANING FOR: {diagnosis_code} ===")
            
            if not self.api_integrator:
                logger.error("❌ No API integrator available")
                return "Explanation unavailable"
            
            if not hasattr(self.api_integrator, 'call_llm_isolated'):
                logger.error("❌ call_llm_isolated method not found")
                return "Explanation unavailable"
            
            prompt = f"Explain ICD-10 diagnosis code '{diagnosis_code}' in 1-2 lines. What medical condition does this code represent?"
            system_msg = "You are a medical coding expert. Provide brief, accurate explanations for ICD-10 diagnosis codes in 1-2 lines."
            
            logger.info(f"📝 Sending prompt: {prompt}")
            logger.info(f"📝 System message: {system_msg}")
            
            response = self.api_integrator.call_llm_isolated(prompt, system_msg)
            
            logger.info(f"📄 Raw response: {response}")
            
            if response and response != "Explanation unavailable":
                # Clean up the response
                lines = response.strip().split('\n')
                clean_response = ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
                logger.info(f"✅ Diagnosis code meaning generated successfully: {clean_response}")
                return clean_response
            else:
                logger.warning(f"⚠️ LLM returned unavailable response: {response}")
                return "Explanation unavailable"
                
        except Exception as e:
            logger.error(f"❌ Exception getting diagnosis code meaning: {e}")
            return "Explanation unavailable"

    def _add_meanings_to_medical_records(self, extraction_result: Dict[str, Any]):
        """Add meanings to individual medical records with debug logging"""
        try:
            logger.info("📋 === ADDING MEANINGS TO MEDICAL RECORDS ===")
            
            service_meanings = extraction_result["code_meanings"]["service_code_meanings"]
            diagnosis_meanings = extraction_result["code_meanings"]["diagnosis_code_meanings"]
            
            meanings_added = 0
            
            for i, record in enumerate(extraction_result["hlth_srvc_records"]):
                logger.info(f"📋 Processing record {i+1}: {record.get('hlth_srvc_cd', 'N/A')}")
                
                # Add service code meaning
                if "hlth_srvc_cd" in record:
                    service_code = str(record["hlth_srvc_cd"])
                    if service_code in service_meanings:
                        record["hlth_srvc_meaning"] = service_meanings[service_code]
                        meanings_added += 1
                        logger.info(f"✅ Added service meaning for {service_code}")
                    else:
                        record["hlth_srvc_meaning"] = "Meaning not available"
                        logger.warning(f"⚠️ No meaning available for service code {service_code}")
                
                # Add diagnosis code meanings
                if "diagnosis_codes" in record:
                    for j, diag in enumerate(record["diagnosis_codes"]):
                        if isinstance(diag, dict) and "code" in diag:
                            diag_code = str(diag["code"])
                            if diag_code in diagnosis_meanings:
                                diag["meaning"] = diagnosis_meanings[diag_code]
                                meanings_added += 1
                                logger.info(f"✅ Added diagnosis meaning for {diag_code}")
                            else:
                                diag["meaning"] = "Meaning not available"
                                logger.warning(f"⚠️ No meaning available for diagnosis code {diag_code}")
                            
            logger.info(f"✅ Added {meanings_added} total meanings to medical records")
            
        except Exception as e:
            logger.error(f"❌ Error adding meanings to medical records: {e}")

    def extract_pharmacy_fields(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pharmacy fields WITH COMPREHENSIVE DEBUG LOGGING AND MEANING GENERATION"""
        logger.info("🔍 ===== STARTING PHARMACY FIELD EXTRACTION WITH MEANING GENERATION =====")
        
        extraction_result = {
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
            "llm_call_status": "not_attempted",
            "debug_info": {
                "api_integrator_available": self.api_integrator is not None,
                "call_llm_isolated_available": hasattr(self.api_integrator, 'call_llm_isolated') if self.api_integrator else False,
                "codes_found": False,
                "meaning_generation_attempts": 0,
                "meaning_generation_successes": 0,
                "meaning_generation_failures": 0,
                "ndc_codes_processed": [],
                "medications_processed": []
            }
        }

        try:
            # Debug API integrator status
            logger.info(f"🔍 DEBUG: API integrator available: {extraction_result['debug_info']['api_integrator_available']}")
            logger.info(f"🔍 DEBUG: call_llm_isolated available: {extraction_result['debug_info']['call_llm_isolated_available']}")

            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("⚠️ No pharmacy claims data found in deidentified pharmacy data")
                return extraction_result

            # Step 1: Extract codes and medications first
            logger.info("💊 Step 1: Extracting pharmacy codes and medications...")
            self._recursive_pharmacy_extraction(pharmacy_data, extraction_result)

            # Convert sets to lists for JSON serialization
            unique_ndc_codes = list(extraction_result["extraction_summary"]["unique_ndc_codes"])
            unique_label_names = list(extraction_result["extraction_summary"]["unique_label_names"])
            
            extraction_result["extraction_summary"]["unique_ndc_codes"] = unique_ndc_codes
            extraction_result["extraction_summary"]["unique_label_names"] = unique_label_names

            # Debug: Log codes found
            logger.info(f"📊 FOUND {len(unique_ndc_codes)} unique NDC codes: {unique_ndc_codes[:5]}...")
            logger.info(f"📊 FOUND {len(unique_label_names)} unique medications: {unique_label_names[:5]}...")
            
            extraction_result["debug_info"]["codes_found"] = len(unique_ndc_codes) > 0 or len(unique_label_names) > 0

            # Step 2: Generate meanings if API integrator is available
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated'):
                if unique_ndc_codes or unique_label_names:
                    logger.info("🤖 Step 2: Starting meaning generation...")
                    extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        # Generate NDC code meanings
                        logger.info(f"🔍 Generating meanings for {len(unique_ndc_codes)} NDC codes...")
                        for i, ndc_code in enumerate(unique_ndc_codes[:10], 1):
                            if ndc_code and str(ndc_code).strip():
                                logger.info(f"🔍 Processing NDC code {i}/10: {ndc_code}")
                                extraction_result["debug_info"]["meaning_generation_attempts"] += 1
                                extraction_result["debug_info"]["ndc_codes_processed"].append(str(ndc_code))
                                
                                meaning = self._get_ndc_code_meaning_debug(str(ndc_code))
                                
                                if meaning and meaning != "Explanation unavailable":
                                    extraction_result["code_meanings"]["ndc_code_meanings"][str(ndc_code)] = meaning
                                    extraction_result["debug_info"]["meaning_generation_successes"] += 1
                                    logger.info(f"✅ NDC code {ndc_code}: {meaning[:50]}...")
                                else:
                                    extraction_result["debug_info"]["meaning_generation_failures"] += 1
                                    logger.warning(f"❌ Failed to get meaning for NDC code: {ndc_code}")
                        
                        # Generate medication meanings
                        logger.info(f"🔍 Generating meanings for {len(unique_label_names)} medications...")
                        for i, medication in enumerate(unique_label_names[:15], 1):
                            if medication and str(medication).strip():
                                logger.info(f"🔍 Processing medication {i}/15: {medication}")
                                extraction_result["debug_info"]["meaning_generation_attempts"] += 1
                                extraction_result["debug_info"]["medications_processed"].append(str(medication))
                                
                                meaning = self._get_medication_meaning_debug(str(medication))
                                
                                if meaning and meaning != "Explanation unavailable":
                                    extraction_result["code_meanings"]["medication_meanings"][str(medication)] = meaning
                                    extraction_result["debug_info"]["meaning_generation_successes"] += 1
                                    logger.info(f"✅ Medication {medication}: {meaning[:50]}...")
                                else:
                                    extraction_result["debug_info"]["meaning_generation_failures"] += 1
                                    logger.warning(f"❌ Failed to get meaning for medication: {medication}")
                        
                        # Step 3: Add meanings to individual records
                        logger.info("💊 Step 3: Adding meanings to individual records...")
                        self._add_meanings_to_pharmacy_records(extraction_result)
                        
                        # Final status
                        total_meanings = len(extraction_result["code_meanings"]["ndc_code_meanings"]) + len(extraction_result["code_meanings"]["medication_meanings"])
                        
                        if total_meanings > 0:
                            extraction_result["code_meanings_added"] = True
                            extraction_result["llm_call_status"] = "completed"
                            logger.info(f"✅ Successfully generated {total_meanings} total meanings for pharmacy codes and medications")
                        else:
                            extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Meaning generation completed but no meanings were generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating pharmacy code meanings: {e}")
                        extraction_result["code_meaning_error"] = str(e)
                        extraction_result["llm_call_status"] = "failed"
                else:
                    extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No codes found to generate meanings for")
            else:
                extraction_result["llm_call_status"] = "skipped_no_api"
                if not self.api_integrator:
                    logger.error("❌ No API integrator available for meaning generation")
                elif not hasattr(self.api_integrator, 'call_llm_isolated'):
                    logger.error("❌ API integrator missing call_llm_isolated method")

            # Final summary
            logger.info(f"💊 ===== PHARMACY EXTRACTION COMPLETED =====")
            logger.info(f"  📊 NDC records: {extraction_result['extraction_summary']['total_ndc_records']}")
            logger.info(f"  🤖 LLM status: {extraction_result['llm_call_status']}")
            logger.info(f"  ✅ Meanings added: {extraction_result['code_meanings_added']}")
            logger.info(f"  📈 Success rate: {extraction_result['debug_info']['meaning_generation_successes']}/{extraction_result['debug_info']['meaning_generation_attempts']}")
            logger.info(f"  🔤 NDC meanings: {len(extraction_result['code_meanings']['ndc_code_meanings'])}")
            logger.info(f"  🔤 Medication meanings: {len(extraction_result['code_meanings']['medication_meanings'])}")

        except Exception as e:
            logger.error(f"❌ Error in pharmacy field extraction: {e}")
            extraction_result["error"] = f"Pharmacy extraction failed: {str(e)}"

        return extraction_result

    def _get_ndc_code_meaning_debug(self, ndc_code: str) -> str:
        """Get NDC code meaning with comprehensive debug logging"""
        try:
            logger.info(f"🔍 === GENERATING NDC CODE MEANING FOR: {ndc_code} ===")
            
            prompt = f"Explain NDC code '{ndc_code}' in 1-2 lines. What medication or drug product does this NDC number represent?"
            system_msg = "You are a pharmacy expert. Provide brief, accurate explanations for NDC (National Drug Code) numbers in 1-2 lines."
            
            logger.info(f"📝 Sending prompt: {prompt}")
            
            response = self.api_integrator.call_llm_isolated(prompt, system_msg)
            
            logger.info(f"📄 Raw response: {response}")
            
            if response and response != "Explanation unavailable":
                lines = response.strip().split('\n')
                clean_response = ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
                logger.info(f"✅ NDC code meaning generated successfully: {clean_response}")
                return clean_response
            else:
                logger.warning(f"⚠️ LLM returned unavailable response: {response}")
                return "Explanation unavailable"
                
        except Exception as e:
            logger.error(f"❌ Exception getting NDC code meaning: {e}")
            return "Explanation unavailable"

    def _get_medication_meaning_debug(self, medication_name: str) -> str:
        """Get medication meaning with comprehensive debug logging"""
        try:
            logger.info(f"🔍 === GENERATING MEDICATION MEANING FOR: {medication_name} ===")
            
            prompt = f"Explain the medication '{medication_name}' in 1-2 lines. What is this drug used for and how does it work?"
            system_msg = "You are a pharmacist. Provide brief, accurate explanations for medications in 1-2 lines focusing on primary use and mechanism."
            
            logger.info(f"📝 Sending prompt: {prompt}")
            
            response = self.api_integrator.call_llm_isolated(prompt, system_msg)
            
            logger.info(f"📄 Raw response: {response}")
            
            if response and response != "Explanation unavailable":
                lines = response.strip().split('\n')
                clean_response = ' '.join(lines[:2]) if len(lines) > 1 else lines[0]
                logger.info(f"✅ Medication meaning generated successfully: {clean_response}")
                return clean_response
            else:
                logger.warning(f"⚠️ LLM returned unavailable response: {response}")
                return "Explanation unavailable"
                
        except Exception as e:
            logger.error(f"❌ Exception getting medication meaning: {e}")
            return "Explanation unavailable"

    def _add_meanings_to_pharmacy_records(self, extraction_result: Dict[str, Any]):
        """Add meanings to individual pharmacy records with debug logging"""
        try:
            logger.info("💊 === ADDING MEANINGS TO PHARMACY RECORDS ===")
            
            ndc_meanings = extraction_result["code_meanings"]["ndc_code_meanings"]
            medication_meanings = extraction_result["code_meanings"]["medication_meanings"]
            
            meanings_added = 0
            
            for i, record in enumerate(extraction_result["ndc_records"]):
                logger.info(f"💊 Processing record {i+1}: NDC={record.get('ndc', 'N/A')}, Medication={record.get('lbl_nm', 'N/A')}")
                
                # Add NDC code meaning
                if "ndc" in record:
                    ndc_code = str(record["ndc"])
                    if ndc_code in ndc_meanings:
                        record["ndc_meaning"] = ndc_meanings[ndc_code]
                        meanings_added += 1
                        logger.info(f"✅ Added NDC meaning for {ndc_code}")
                    else:
                        record["ndc_meaning"] = "Meaning not available"
                        logger.warning(f"⚠️ No meaning available for NDC code {ndc_code}")
                
                # Add medication meaning
                if "lbl_nm" in record:
                    medication = str(record["lbl_nm"])
                    if medication in medication_meanings:
                        record["medication_meaning"] = medication_meanings[medication]
                        meanings_added += 1
                        logger.info(f"✅ Added medication meaning for {medication}")
                    else:
                        record["medication_meaning"] = "Meaning not available"
                        logger.warning(f"⚠️ No meaning available for medication {medication}")
                            
            logger.info(f"✅ Added {meanings_added} total meanings to pharmacy records")
            
        except Exception as e:
            logger.error(f"❌ Error adding meanings to pharmacy records: {e}")
 
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
        """Enhanced health entity extraction using code meanings generated during extraction"""
        logger.info("🎯 ===== STARTING ENHANCED ENTITY EXTRACTION =====")
        
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
            "enhanced_with_code_meanings": False
        }
 
        try:
            # 1. Calculate age from date of birth
            if patient_data and patient_data.get('date_of_birth'):
                age, age_group = self.calculate_age_from_dob(patient_data['date_of_birth'])
                if age is not None:
                    entities["age"] = age
                    entities["age_group"] = age_group
                    entities["analysis_details"].append(f"Age calculated from DOB: {age} years ({age_group})")
 
            # 2. Check if code meanings are available from extractions
            medical_meanings_available = (medical_extraction and 
                                        medical_extraction.get("code_meanings_added", False) and
                                        medical_extraction.get("code_meanings", {}))
            
            pharmacy_meanings_available = (pharmacy_extraction and 
                                         pharmacy_extraction.get("code_meanings_added", False) and
                                         pharmacy_extraction.get("code_meanings", {}))
            
            logger.info(f"🔍 Medical meanings available: {medical_meanings_available}")
            logger.info(f"🔍 Pharmacy meanings available: {pharmacy_meanings_available}")
            
            if medical_meanings_available or pharmacy_meanings_available:
                # Use existing meanings from extractions
                logger.info("✅ Using code meanings from extraction phase")
                entities = self._analyze_entities_with_extracted_meanings(
                    entities, medical_extraction, pharmacy_extraction
                )
                entities["enhanced_with_code_meanings"] = True
                entities["llm_analysis"] = "used_extracted_meanings"
                entities["analysis_details"].append("Used code meanings from extraction phase")
                
            elif api_integrator:
                # Generate meanings on-demand if not available
                logger.info("🔄 Generating meanings on-demand for entity extraction")
                llm_entities = self._extract_entities_with_llm(
                    pharmacy_data, pharmacy_extraction, medical_extraction,
                    patient_data, api_integrator
                )
 
                if llm_entities:
                    entities.update(llm_entities)
                    entities["llm_analysis"] = "completed"
                    entities["enhanced_with_code_meanings"] = True
                    entities["analysis_details"].append("LLM entity extraction completed successfully")
                else:
                    entities["analysis_details"].append("LLM entity extraction failed, using fallback method")
                    self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
            else:
                entities["analysis_details"].append("No LLM available, using direct analysis")
                self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
 
            # 3. Always extract medications for reference
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        medication_info = {
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "path": record.get("data_path", "")
                        }
                        # Add meaning if available
                        if record.get("medication_meaning"):
                            medication_info["meaning"] = record.get("medication_meaning")
                        
                        entities["medications_identified"].append(medication_info)
 
            entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
 
            # Final summary
            logger.info(f"🎯 ===== ENTITY EXTRACTION COMPLETED =====")
            logger.info(f"  ✅ Enhanced with meanings: {entities['enhanced_with_code_meanings']}")
            logger.info(f"  🤖 LLM analysis: {entities['llm_analysis']}")
            logger.info(f"  🩺 Diabetes: {entities['diabetics']}")
            logger.info(f"  💓 Blood pressure: {entities['blood_pressure']}")
            logger.info(f"  🚬 Smoking: {entities['smoking']}")
            logger.info(f"  🍷 Alcohol: {entities['alcohol']}")
            logger.info(f"  💊 Medications identified: {len(entities['medications_identified'])}")

        except Exception as e:
            logger.error(f"❌ Error in enhanced entity extraction: {e}")
            entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
 
        return entities

    def _analyze_entities_with_extracted_meanings(self, entities: Dict[str, Any], 
                                                medical_extraction: Dict[str, Any], 
                                                pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entities using code meanings already generated during extraction"""
        try:
            logger.info("🔍 Analyzing entities using extracted code meanings...")
            
            # Get the code meanings from extractions
            medical_meanings = medical_extraction.get("code_meanings", {}) if medical_extraction else {}
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {}) if pharmacy_extraction else {}
            
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            service_meanings = medical_meanings.get("service_code_meanings", {})
            ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
            medication_meanings = pharmacy_meanings.get("medication_meanings", {})
            
            logger.info(f"🔍 Available meanings - Diagnosis: {len(diagnosis_meanings)}, Service: {len(service_meanings)}, NDC: {len(ndc_meanings)}, Medications: {len(medication_meanings)}")
            
            # Analyze diagnosis meanings for conditions
            medical_conditions = []
            for code, meaning in diagnosis_meanings.items():
                meaning_lower = meaning.lower()
                
                # Check for diabetes
                if any(term in meaning_lower for term in ['diabetes', 'diabetic', 'insulin', 'glucose']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes (from ICD-10 {code}: {meaning})")
                    logger.info(f"✅ Found diabetes from diagnosis code {code}")
                
                # Check for hypertension
                if any(term in meaning_lower for term in ['hypertension', 'high blood pressure', 'elevated blood pressure']):
                    entities["blood_pressure"] = "diagnosed"
                    medical_conditions.append(f"Hypertension (from ICD-10 {code}: {meaning})")
                    logger.info(f"✅ Found hypertension from diagnosis code {code}")
                
                # Check for smoking/tobacco
                if any(term in meaning_lower for term in ['tobacco', 'smoking', 'nicotine']):
                    entities["smoking"] = "yes"
                    medical_conditions.append(f"Tobacco use (from ICD-10 {code}: {meaning})")
                    logger.info(f"✅ Found smoking from diagnosis code {code}")
                
                # Check for alcohol
                if any(term in meaning_lower for term in ['alcohol', 'alcoholism', 'alcohol abuse']):
                    entities["alcohol"] = "yes"
                    medical_conditions.append(f"Alcohol-related condition (from ICD-10 {code}: {meaning})")
                    logger.info(f"✅ Found alcohol condition from diagnosis code {code}")
            
            # Analyze medication meanings for treatments
            for medication, meaning in medication_meanings.items():
                meaning_lower = meaning.lower()
                
                # Check for diabetes medications
                if any(term in meaning_lower for term in ['diabetes', 'blood sugar', 'insulin', 'metformin', 'glucose']):
                    entities["diabetics"] = "yes"
                    medical_conditions.append(f"Diabetes treatment (medication {medication}: {meaning})")
                    logger.info(f"✅ Found diabetes medication: {medication}")
                
                # Check for blood pressure medications
                if any(term in meaning_lower for term in ['blood pressure', 'hypertension', 'antihypertensive', 'ace inhibitor', 'beta blocker']):
                    if entities["blood_pressure"] == "unknown":
                        entities["blood_pressure"] = "managed"
                    medical_conditions.append(f"Blood pressure management (medication {medication}: {meaning})")
                    logger.info(f"✅ Found blood pressure medication: {medication}")
                
                # Check for smoking cessation
                if any(term in meaning_lower for term in ['smoking cessation', 'nicotine replacement', 'quit smoking']):
                    entities["smoking"] = "yes"
                    medical_conditions.append(f"Smoking cessation (medication {medication}: {meaning})")
                    logger.info(f"✅ Found smoking cessation medication: {medication}")
                
                # Check for alcohol treatment
                if any(term in meaning_lower for term in ['alcohol dependence', 'alcoholism', 'naltrexone']):
                    entities["alcohol"] = "yes"
                    medical_conditions.append(f"Alcohol treatment (medication {medication}: {meaning})")
                    logger.info(f"✅ Found alcohol treatment medication: {medication}")
            
            # Analyze NDC meanings for additional medication insights
            for ndc, meaning in ndc_meanings.items():
                meaning_lower = meaning.lower()
                
                if any(term in meaning_lower for term in ['diabetes', 'insulin', 'metformin']):
                    entities["diabetics"] = "yes"
                    logger.info(f"✅ Found diabetes from NDC {ndc}")
                
                if any(term in meaning_lower for term in ['blood pressure', 'hypertension']):
                    if entities["blood_pressure"] == "unknown":
                        entities["blood_pressure"] = "managed"
                    logger.info(f"✅ Found blood pressure medication from NDC {ndc}")
            
            entities["medical_conditions"] = medical_conditions
            entities["analysis_details"].append(f"Analyzed {len(diagnosis_meanings)} diagnosis meanings, {len(medication_meanings)} medication meanings, {len(ndc_meanings)} NDC meanings")
            
            logger.info(f"✅ Entity analysis using extracted meanings completed: diabetes={entities['diabetics']}, bp={entities['blood_pressure']}, smoking={entities['smoking']}, alcohol={entities['alcohol']}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error analyzing entities with extracted meanings: {e}")
            entities["analysis_details"].append(f"Error in meaning-based analysis: {str(e)}")
            return entities

    def _extract_entities_with_llm(self, pharmacy_data: Dict[str, Any],
                                   pharmacy_extraction: Dict[str, Any],
                                   medical_extraction: Dict[str, Any],
                                   patient_data: Dict[str, Any],
                                   api_integrator) -> Dict[str, Any]:
        """Use LLM to extract health entities from claims data (fallback method)"""
        try:
            logger.info("🤖 Starting fallback LLM entity extraction...")
            
            # Basic analysis without meanings
            context = self._prepare_basic_context(
                pharmacy_data, pharmacy_extraction, medical_extraction, patient_data
            )

            entity_prompt = f"""
You are a medical AI expert analyzing patient claims data.

PATIENT CLAIMS DATA:
{context}

ANALYSIS TASK:
Based on the provided data, determine:

1. **diabetics**: "yes" if any medication treats diabetes OR any diagnosis indicates diabetes
2. **smoking**: "yes" if any medication is for smoking cessation OR any diagnosis indicates tobacco use
3. **alcohol**: "yes" if any medication treats alcohol disorders OR any diagnosis indicates alcohol problems  
4. **blood_pressure**: 
   - "managed" if any medication treats hypertension/blood pressure
   - "diagnosed" if any diagnosis indicates hypertension
   - "unknown" if neither

RESPONSE FORMAT (JSON ONLY):
{{
    "diabetics": "yes/no",
    "smoking": "yes/no", 
    "alcohol": "yes/no",
    "blood_pressure": "unknown/managed/diagnosed",
    "medical_conditions": ["condition1", "condition2"],
    "llm_reasoning": "Based on analysis of claims data"
}}
"""

            response = api_integrator.call_llm(entity_prompt)

            if response and not response.startswith("Error"):
                result = self._parse_llm_response_robust(response)
                return result
            else:
                logger.error(f"❌ Fallback LLM call failed: {response}")
                return None

        except Exception as e:
            logger.error(f"❌ Error in fallback LLM entity extraction: {e}")
            return None

    def _prepare_basic_context(self, pharmacy_data: Dict[str, Any],
                              pharmacy_extraction: Dict[str, Any],
                              medical_extraction: Dict[str, Any],
                              patient_data: Dict[str, Any]) -> str:
        """Prepare basic context without meanings"""
        try:
            context_parts = []
            
            # Patient info
            if patient_data:
                age = patient_data.get("calculated_age", patient_data.get("age", "unknown"))
                context_parts.append(f"PATIENT: Age {age}, Gender {patient_data.get('gender', 'unknown')}")
            
            # Medications
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                medications = []
                for record in pharmacy_extraction["ndc_records"][:10]:
                    if record.get("lbl_nm"):
                        medications.append(record.get("lbl_nm"))
                context_parts.append(f"MEDICATIONS: {medications}")
            
            # Diagnosis codes
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                diagnosis_codes = []
                for record in medical_extraction["hlth_srvc_records"][:10]:
                    if record.get("diagnosis_codes"):
                        for diag in record["diagnosis_codes"][:3]:
                            if isinstance(diag, dict) and diag.get("code"):
                                diagnosis_codes.append(diag.get("code"))
                context_parts.append(f"DIAGNOSIS_CODES: {diagnosis_codes}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing basic context: {e}")
            return "Patient claims data available for analysis."

    def _parse_llm_response_robust(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response with multiple fallback strategies"""
        try:
            # Try to extract JSON
            json_str = llm_response.strip()
            
            # Remove markdown wrappers
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            elif json_str.startswith('```'):
                json_str = json_str[3:]
                
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            
            # Find JSON boundaries
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
            
            # Parse JSON
            llm_entities = json.loads(json_str)
            return self._validate_and_clean_entities(llm_entities)
            
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
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
                "llm_reasoning": llm_entities.get("llm_reasoning", "LLM analysis completed")
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
                
            return cleaned_entities
            
        except Exception as e:
            logger.error(f"Error validating entities: {e}")
            return None
 
    def _analyze_entities_direct(self, pharmacy_data: Dict[str, Any],
                                pharmacy_extraction: Dict[str, Any],
                                medical_extraction: Dict[str, Any],
                                entities: Dict[str, Any]):
        """Fallback direct entity analysis without meanings"""
        try:
            logger.info("🔄 Using direct entity analysis (fallback method)")
            
            # Basic pattern matching
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    medication_name = record.get("lbl_nm", "").lower()
                    
                    # Check for diabetes medications
                    if any(term in medication_name for term in ['metformin', 'insulin', 'glipizide']):
                        entities["diabetics"] = "yes"
                        
                    # Check for blood pressure medications
                    if any(term in medication_name for term in ['amlodipine', 'lisinopril', 'atenolol']):
                        entities["blood_pressure"] = "managed"

            entities["analysis_details"].append("Direct entity analysis completed as fallback")

        except Exception as e:
            logger.error(f"Error in direct entity analysis: {e}")
            entities["analysis_details"].append(f"Error in direct entity analysis: {str(e)}")

    def prepare_chunked_context(self, chat_context: Dict[str, Any]) -> str:
        """Prepare comprehensive context for chatbot with all meanings included"""
        try:
            context_sections = []

            # Patient Overview
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"PATIENT OVERVIEW:\n{json.dumps(patient_overview, indent=2)}")

            # Medical Extractions with CODE MEANINGS
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_sections.append(f"MEDICAL EXTRACTIONS WITH CODE MEANINGS:\n{json.dumps(medical_extraction, indent=2)}")

            # Pharmacy Extractions with CODE MEANINGS
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_sections.append(f"PHARMACY EXTRACTIONS WITH CODE MEANINGS:\n{json.dumps(pharmacy_extraction, indent=2)}")

            # Entity Extraction with Enhanced Meanings
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"HEALTH ENTITIES WITH MEANINGS:\n{json.dumps(entity_extraction, indent=2)}")

            # Heart Attack Prediction
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"HEART ATTACK PREDICTION:\n{json.dumps(heart_attack_prediction, indent=2)}")

            return "\n\n" + "\n\n".join(context_sections)

        except Exception as e:
            logger.error(f"Error preparing chunked context: {e}")
            return "Patient claims data with code meanings available for analysis."

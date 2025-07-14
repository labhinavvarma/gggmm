
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
        """Use LLM to extract health entities from claims data"""
        try:
            # Prepare comprehensive context for LLM
            context_data = {
                "pharmacy_claims": pharmacy_data,
                "pharmacy_extraction": pharmacy_extraction,
                "medical_extraction": medical_extraction,
                "patient_info": {
                    "gender": patient_data.get("gender", "unknown"),
                    "age": patient_data.get("calculated_age", "unknown")
                }
            }
            
            # Create detailed prompt for entity extraction
            entity_prompt = f"""
You are a medical AI assistant analyzing patient claims data to extract specific health entities. Based on the comprehensive medical and pharmacy claims data below, extract the following 5 health entities with high accuracy:

COMPREHENSIVE CLAIMS DATA:
{json.dumps(context_data, indent=2)}

ENTITY EXTRACTION REQUIREMENTS:
Extract exactly these 5 entities and return them in JSON format:

1. **diabetics**: "yes" or "no" - Based on diabetes medications (insulin, metformin, etc.) or ICD-10 codes (E10, E11, etc.)
2. **smoking**: "yes" or "no" - Based on smoking-related ICD-10 codes (Z72.0, F17) or smoking cessation medications
3. **alcohol**: "yes" or "no" - Based on alcohol-related ICD-10 codes (F10, Z72.1) or alcohol treatment medications
4. **blood_pressure**: "unknown", "managed", or "diagnosed" - Based on BP medications (lisinopril, amlodipine, etc.) or ICD-10 codes (I10-I15)
5. **medical_conditions**: List of identified conditions from ICD-10 codes and medication patterns

ANALYSIS INSTRUCTIONS:
- Analyze ALL medical codes (ICD-10 diagnosis codes)
- Analyze ALL pharmacy medications (NDC codes and label names)
- Look for patterns in medication usage and medical service codes
- Consider medication combinations that indicate specific conditions
- Be conservative - only mark "yes" if there is clear evidence
- For blood_pressure: "managed" = on BP medication, "diagnosed" = BP-related codes, "unknown" = no evidence

RESPONSE FORMAT (JSON only):
{{
    "diabetics": "yes/no",
    "smoking": "yes/no", 
    "alcohol": "yes/no",
    "blood_pressure": "unknown/managed/diagnosed",
    "medical_conditions": ["condition1", "condition2", ...],
    "llm_reasoning": "Brief explanation of key findings"
}}

Provide only the JSON response with accurate entity extraction based on the claims data.
"""

            logger.info("🤖 Calling LLM for entity extraction...")
            
            # Call LLM with entity extraction prompt
            llm_response = api_integrator.call_llm(entity_prompt)
            
            if llm_response and not llm_response.startswith("Error"):
                # Try to parse JSON response
                try:
                    # Clean the response to extract JSON
                    json_start = llm_response.find('{')
                    json_end = llm_response.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = llm_response[json_start:json_end]
                        llm_entities = json.loads(json_str)
                        
                        # Validate and clean the entities
                        cleaned_entities = {
                            "diabetics": str(llm_entities.get("diabetics", "no")).lower(),
                            "smoking": str(llm_entities.get("smoking", "no")).lower(),
                            "alcohol": str(llm_entities.get("alcohol", "no")).lower(),
                            "blood_pressure": str(llm_entities.get("blood_pressure", "unknown")).lower(),
                            "medical_conditions": llm_entities.get("medical_conditions", []),
                            "llm_reasoning": llm_entities.get("llm_reasoning", "LLM analysis completed")
                        }
                        
                        logger.info(f"✅ LLM entity extraction successful: {cleaned_entities}")
                        return cleaned_entities
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM JSON response: {e}")
                    logger.error(f"LLM response: {llm_response[:500]}...")
                    return None
            else:
                logger.error(f"LLM call failed: {llm_response}")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
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
                self._analyze_pharmacy_for_entities(data_str, entities)
            
            # Analyze structured pharmacy extraction
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                self._analyze_pharmacy_extraction_for_entities(pharmacy_extraction, entities)
            
            # Analyze medical extraction for conditions
            if medical_extraction and medical_extraction.get("hlth_srvc_records"):
                self._analyze_medical_extraction_for_entities(medical_extraction, entities)
            
            entities["analysis_details"].append("Direct entity analysis completed as fallback")
            
        except Exception as e:
            logger.error(f"Error in direct entity analysis: {e}")
            entities["analysis_details"].append(f"Error in direct entity analysis: {str(e)}")

    # ... [Keep all the existing methods: deidentify_medical_data, deidentify_pharmacy_data, 
    # deidentify_mcid_data, _deep_deidentify_json, _deidentify_string, extract_medical_fields, 
    # extract_pharmacy_fields, _analyze_pharmacy_for_entities, _analyze_pharmacy_extraction_for_entities, 
    # _analyze_medical_extraction_for_entities, prepare_chunked_context, etc. - these remain unchanged]

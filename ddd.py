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
Analyze the medications and ICD-10 codes to determine:

1. diabetics: "yes" if diabetes medications (metformin, insulin, etc.) or diabetes ICD codes (E10.x, E11.x)
2. smoking: "yes" if smoking cessation meds or tobacco ICD codes (F17.x, Z72.0)  
3. alcohol: "yes" if alcohol treatment meds or alcohol ICD codes (F10.x)
4. blood_pressure: "managed" if BP meds, "diagnosed" if hypertension ICD codes, "unknown" if neither
5. medical_conditions: list all identified conditions

RESPONSE FORMAT (JSON ONLY, NO MARKDOWN):
{{
    "diabetics": "yes",
    "smoking": "no", 
    "alcohol": "no",
    "blood_pressure": "managed",
    "medical_conditions": ["hypertension", "asthma"],
    "llm_reasoning": "Found amlodipine (BP med) and I10 (hypertension code)"
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
            "age": patient_data.get("calculated_age", "unknown"),
            "gender": patient_data.get("gender", "unknown")
        }
        context_parts.append(f"PATIENT: {patient_info}")
        
        # 2. Medications Summary (top 10 most relevant)
        medications = []
        if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
            for record in pharmacy_extraction["ndc_records"][:10]:  # Limit to 10
                if record.get("lbl_nm"):
                    medications.append({
                        "name": record.get("lbl_nm", ""),
                        "ndc": record.get("ndc", "")
                    })
        context_parts.append(f"MEDICATIONS: {medications}")
        
        # 3. Diagnosis Codes Summary (top 15 most relevant)
        diagnosis_codes = []
        if medical_extraction and medical_extraction.get("hlth_srvc_records"):
            for record in medical_extraction["hlth_srvc_records"][:15]:  # Limit to 15
                if record.get("diagnosis_codes"):
                    for diag in record["diagnosis_codes"][:3]:  # Max 3 per record
                        diagnosis_codes.append(diag.get("code", ""))
        
        # Remove duplicates and keep top 20
        unique_codes = list(dict.fromkeys(diagnosis_codes))[:20]
        context_parts.append(f"ICD_CODES: {unique_codes}")
        
        # 4. Service Codes Summary (top 10)
        service_codes = []
        if medical_extraction and medical_extraction.get("hlth_srvc_records"):
            for record in medical_extraction["hlth_srvc_records"][:10]:
                if record.get("hlth_srvc_cd"):
                    service_codes.append(record.get("hlth_srvc_cd"))
        
        unique_services = list(dict.fromkeys(service_codes))[:10]
        context_parts.append(f"SERVICE_CODES: {unique_services}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error preparing summarized context: {e}")
        return "Patient claims data available for analysis."

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

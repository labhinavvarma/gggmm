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
You are a medical AI expert analyzing patient claims data. Use your medical knowledge to understand what each medication treats and what each ICD-10 code means.

COMPREHENSIVE CLAIMS DATA:
{json.dumps(context_data, indent=2)}

ANALYSIS METHODOLOGY:

1. MEDICATION ANALYSIS:
   - For each medication found, determine what medical condition it treats
   - Use your medical knowledge of therapeutic indications
   - Consider both generic and brand names
   - Example: "METFORMIN HCL 500 MG" → treats Type 2 diabetes → diabetics = "yes"

2. ICD-10 CODE ANALYSIS:
   - For each ICD-10 code found, determine what medical condition it represents
   - Use your medical knowledge of ICD-10 code meanings
   - Example: "E11.9" → Type 2 diabetes mellitus without complications → diabetics = "yes"
   - Example: "I10" → Essential hypertension → blood_pressure = "diagnosed"

3. ENTITY EXTRACTION:

diabetics: "yes" or "no"
- YES if any medication treats diabetes (any type)
- YES if any ICD-10 code represents diabetes (any type)
- Consider: insulin, metformin, sulfonylureas, SGLT2 inhibitors, GLP-1 agonists, etc.
- Consider: E10.x (Type 1), E11.x (Type 2), E12.x (Malnutrition-related), E13.x (Other specified), E14.x (Unspecified)

smoking: "yes" or "no"
- YES if any medication is for smoking cessation
- YES if any ICD-10 code represents tobacco use/dependence
- Consider: nicotine replacement, varenicline, bupropion for smoking cessation
- Consider: Z72.0 (Tobacco use), F17.x (Tobacco dependence)

alcohol: "yes" or "no"
- YES if any medication treats alcohol use disorders
- YES if any ICD-10 code represents alcohol use/dependence
- Consider: naltrexone, disulfiram, acamprosate
- Consider: F10.x (Alcohol use disorders), Z72.1 (Alcohol use)

blood_pressure: "unknown", "managed", or "diagnosed"
- "managed" if taking antihypertensive medications
- "diagnosed" if ICD-10 codes represent hypertension
- "unknown" if no evidence
- Consider: ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, diuretics
- Consider: I10 (Essential hypertension), I11.x (Hypertensive heart disease), I12.x (Hypertensive chronic kidney disease), etc.

medical_conditions: Array of all conditions identified

CRITICAL INSTRUCTIONS:
- Use your medical knowledge to understand what each medication TREATS
- Use your medical knowledge to understand what each ICD-10 code MEANS
- Don't just pattern match - understand the medical meaning
- Cross-reference findings between medications and diagnosis codes
- Be comprehensive in your medical analysis

RESPONSE FORMAT - RETURN ONLY VALID JSON, NO MARKDOWN:
{{
    "diabetics": "yes/no",
    "smoking": "yes/no", 
    "alcohol": "yes/no",
    "blood_pressure": "unknown/managed/diagnosed",
    "medical_conditions": ["condition1", "condition2"],
    "llm_reasoning": "Medical analysis summary",
    "diabetes_evidence": ["medication/code → medical meaning"],
    "bp_evidence": ["medication/code → medical meaning"],
    "smoking_evidence": ["medication/code → medical meaning"],
    "alcohol_evidence": ["medication/code → medical meaning"],
    "medication_analysis": ["medication_name → treats_condition"],
    "icd10_analysis": ["ICD_code → medical_condition_meaning"]
}}
"""

        logger.info("🤖 Calling LLM for entity extraction...")

        # Call LLM with entity extraction prompt
        llm_response = api_integrator.call_llm(entity_prompt)

        if llm_response and not llm_response.startswith("Error"):
            # Log the full response for debugging
            logger.info(f"📄 Full LLM response length: {len(llm_response)} characters")
            logger.info(f"📄 LLM response preview: {llm_response[:200]}...")
            
            # Try to parse JSON response with improved extraction
            try:
                # Method 1: Handle markdown code blocks
                json_str = llm_response.strip()
                
                # Remove markdown code block wrappers if present
                if json_str.startswith('```json'):
                    json_str = json_str[7:]  # Remove ```json
                elif json_str.startswith('```'):
                    json_str = json_str[3:]   # Remove ```
                    
                if json_str.endswith('```'):
                    json_str = json_str[:-3]  # Remove closing ```
                
                json_str = json_str.strip()
                
                # Method 2: Find JSON boundaries more carefully
                json_start = json_str.find('{')
                
                # Find the matching closing brace by counting braces
                if json_start != -1:
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
                    
                    if json_end != -1:
                        json_str = json_str[json_start:json_end]
                        logger.info(f"🔍 Extracted JSON string: {json_str[:100]}...")
                        
                        # Parse the JSON
                        llm_entities = json.loads(json_str)
                        
                        # Validate and clean the entities
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

                        logger.info(f"✅ LLM entity extraction successful: {cleaned_entities}")
                        return cleaned_entities
                    else:
                        logger.error("❌ Could not find matching closing brace for JSON")
                        return None
                else:
                    logger.error("❌ Could not find JSON start marker '{'")
                    return None

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse LLM JSON response: {e}")
                logger.error(f"📄 Problematic JSON string: {json_str[:500]}...")
                
                # Try to fix common JSON issues
                try:
                    # Attempt to fix truncated JSON by adding missing closing braces
                    fixed_json = json_str
                    
                    # Count opening vs closing braces
                    open_braces = fixed_json.count('{')
                    close_braces = fixed_json.count('}')
                    
                    if open_braces > close_braces:
                        missing_braces = open_braces - close_braces
                        fixed_json += '}' * missing_braces
                        logger.info(f"🔧 Attempting to fix JSON by adding {missing_braces} closing braces")
                        
                        llm_entities = json.loads(fixed_json)
                        
                        cleaned_entities = {
                            "diabetics": str(llm_entities.get("diabetics", "no")).lower(),
                            "smoking": str(llm_entities.get("smoking", "no")).lower(),
                            "alcohol": str(llm_entities.get("alcohol", "no")).lower(),
                            "blood_pressure": str(llm_entities.get("blood_pressure", "unknown")).lower(),
                            "medical_conditions": llm_entities.get("medical_conditions", []),
                            "llm_reasoning": "LLM analysis completed (JSON repaired)"
                        }
                        
                        logger.info(f"✅ LLM entity extraction successful after JSON repair: {cleaned_entities}")
                        return cleaned_entities
                        
                except Exception as repair_error:
                    logger.error(f"❌ JSON repair attempt failed: {repair_error}")
                    return None
                    
        else:
            logger.error(f"❌ LLM call failed: {llm_response}")
            return None

    except Exception as e:
        logger.error(f"❌ Error in LLM entity extraction: {e}")
        return None

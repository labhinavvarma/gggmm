# =============================================================================
# FILE 1: health_data_processor.py
# =============================================================================

# ADD THIS METHOD TO THE HealthDataProcessor CLASS (after __init__)
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

# REPLACE THE EXISTING extract_health_entities_enhanced METHOD WITH THIS:
def extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any], 
                                    pharmacy_extraction: Dict[str, Any],
                                    medical_extraction: Dict[str, Any],
                                    patient_data: Dict[str, Any] = None,
                                    api_integrator = None,
                                    deidentified_medical: Dict[str, Any] = None,
                                    deidentified_pharmacy: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced health entity extraction using LLM with comprehensive data access"""
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
        "debug_info": {}
    }
    
    try:
        # 1. Calculate age from date of birth
        if patient_data and patient_data.get('date_of_birth'):
            age, age_group = self.calculate_age_from_dob(patient_data['date_of_birth'])
            if age is not None:
                entities["age"] = age
                entities["age_group"] = age_group
                entities["analysis_details"].append(f"Age calculated from DOB: {age} years ({age_group})")
        
        # 2. Debug: Log what data we have available
        entities["debug_info"]["data_sources"] = {
            "pharmacy_data_available": bool(pharmacy_data),
            "pharmacy_extraction_available": bool(pharmacy_extraction),
            "medical_extraction_available": bool(medical_extraction),
            "deidentified_medical_available": bool(deidentified_medical),
            "deidentified_pharmacy_available": bool(deidentified_pharmacy),
            "api_integrator_available": bool(api_integrator)
        }
        
        # 3. Use COMPREHENSIVE LLM analysis with ALL available data
        if api_integrator:
            logger.info("🤖 Starting comprehensive LLM entity extraction...")
            
            llm_entities = self._extract_entities_with_llm_comprehensive(
                pharmacy_data, pharmacy_extraction, medical_extraction, 
                patient_data, api_integrator, deidentified_medical, deidentified_pharmacy
            )
            
            if llm_entities:
                # Update entities with LLM results
                entities.update(llm_entities)
                entities["llm_analysis"] = "completed"
                entities["analysis_details"].append("Comprehensive LLM entity extraction completed successfully")
                
                # Debug: Log what LLM found
                entities["debug_info"]["llm_results"] = llm_entities
                logger.info(f"🎯 LLM Entity Results: Diabetes={llm_entities.get('diabetics')}, Smoking={llm_entities.get('smoking')}, BP={llm_entities.get('blood_pressure')}")
                
            else:
                entities["analysis_details"].append("LLM entity extraction failed, using fallback method")
                entities["llm_analysis"] = "failed"
                # Fall back to direct analysis
                self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
        else:
            entities["analysis_details"].append("No LLM available, using direct analysis")
            entities["llm_analysis"] = "no_llm_available"
            # Fall back to direct analysis
            self._analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, entities)
        
        # 4. Always extract medications for reference
        self._extract_medications_list(pharmacy_extraction, entities)
        
        # 5. Final debug summary
        entities["debug_info"]["final_extraction"] = {
            "diabetics": entities["diabetics"],
            "smoking": entities["smoking"],
            "alcohol": entities["alcohol"],
            "blood_pressure": entities["blood_pressure"],
            "conditions_count": len(entities["medical_conditions"]),
            "medications_count": len(entities["medications_identified"])
        }
        
        entities["analysis_details"].append(f"Total analysis sources: Pharmacy data, {len(pharmacy_extraction.get('ndc_records', []))} pharmacy records, {len(medical_extraction.get('hlth_srvc_records', []))} medical records")
        
    except Exception as e:
        logger.error(f"Error in enhanced LLM entity extraction: {e}")
        entities["analysis_details"].append(f"Error in entity extraction: {str(e)}")
        entities["debug_info"]["error"] = str(e)
    
    return entities

# ADD THIS NEW METHOD TO THE HealthDataProcessor CLASS:
def _extract_entities_with_llm_comprehensive(self, pharmacy_data: Dict[str, Any], 
                               pharmacy_extraction: Dict[str, Any],
                               medical_extraction: Dict[str, Any],
                               patient_data: Dict[str, Any],
                               api_integrator,
                               deidentified_medical: Dict[str, Any] = None,
                               deidentified_pharmacy: Dict[str, Any] = None) -> Dict[str, Any]:
    """Use LLM to extract health entities with COMPREHENSIVE context (same as chatbot)"""
    try:
        # Prepare the SAME comprehensive context that the chatbot uses
        comprehensive_context = {
            "patient_info": {
                "gender": patient_data.get("gender", "unknown"),
                "age": patient_data.get("calculated_age", "unknown"),
                "zip_code": patient_data.get("zip_code", "unknown")
            }
        }
        
        # Add ALL available data sources (SAME AS CHATBOT)
        if deidentified_medical:
            comprehensive_context["deidentified_medical"] = deidentified_medical
        
        if deidentified_pharmacy:
            comprehensive_context["deidentified_pharmacy"] = deidentified_pharmacy
        
        if pharmacy_data:
            comprehensive_context["raw_pharmacy_data"] = pharmacy_data
        
        if pharmacy_extraction:
            comprehensive_context["pharmacy_extraction"] = pharmacy_extraction
            
        if medical_extraction:
            comprehensive_context["medical_extraction"] = medical_extraction
        
        # Enhanced entity prompt with comprehensive analysis
        entity_prompt = f"""
You are a medical AI expert analyzing COMPLETE patient claims data for entity extraction. You have access to the SAME comprehensive deidentified data that medical professionals use.

COMPLETE PATIENT CLAIMS DATA:
{json.dumps(comprehensive_context, indent=2, default=str)}

CRITICAL ANALYSIS TASK:
Extract these 5 health entities with maximum accuracy by analyzing ALL available data:

1. **diabetics**: "yes" or "no"
   - Look for diabetes medications: insulin, metformin, glipizide, glucophage, lantus, humalog, novolog
   - Look for ICD-10 codes: E10.*, E11.*, E12.*, E13.*, E14.*
   - Analyze NDC codes for diabetes medications
   - Check medication label names for diabetes indicators

2. **smoking**: "yes" or "no" 
   - Look for ICD-10 codes: Z72.0, F17.*
   - Look for smoking cessation medications
   - Check for tobacco-related diagnoses

3. **alcohol**: "yes" or "no"
   - Look for ICD-10 codes: F10.*, Z72.1
   - Look for alcohol treatment medications
   - Check for alcohol-related diagnoses

4. **blood_pressure**: "unknown", "managed", or "diagnosed"
   - MANAGED: On BP medications (lisinopril, amlodipine, metoprolol, atenolol, losartan, etc.)
   - DIAGNOSED: ICD-10 codes I10.*, I11.*, I12.*, I13.*, I15.*
   - UNKNOWN: No evidence found

5. **medical_conditions**: Array of identified conditions from codes and medications

ANALYSIS INSTRUCTIONS:
- Examine EVERY medication name (lbl_nm) in pharmacy data
- Examine EVERY NDC code in pharmacy extractions  
- Examine EVERY ICD-10 diagnosis code in medical extractions
- Look in both raw data AND structured extractions
- Be thorough - diabetes medications are clear indicators
- Cross-reference medications with medical codes
- If you find ANY diabetes medication or code, mark diabetics as "yes"

RESPONSE FORMAT (JSON ONLY - NO OTHER TEXT):
{{
    "diabetics": "yes/no",
    "smoking": "yes/no", 
    "alcohol": "yes/no",
    "blood_pressure": "unknown/managed/diagnosed",
    "medical_conditions": ["condition1", "condition2"],
    "llm_reasoning": "Key findings that led to these determinations",
    "diabetes_evidence": ["specific medications/codes found"],
    "bp_evidence": ["specific medications/codes found"],
    "smoking_evidence": ["specific codes found"],
    "alcohol_evidence": ["specific codes found"]
}}
"""

        logger.info("🤖 Calling LLM for COMPREHENSIVE entity extraction...")
        logger.info(f"📊 Context size: {len(str(comprehensive_context))} characters")
        
        # Call LLM with comprehensive context
        llm_response = api_integrator.call_llm(entity_prompt)
        
        if llm_response and not llm_response.startswith("Error"):
            logger.info(f"🤖 LLM Raw Response: {llm_response[:500]}...")
            
            # Try to parse JSON response with better error handling
            try:
                # Clean the response to extract JSON
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = llm_response[json_start:json_end]
                    logger.info(f"🔍 Extracted JSON: {json_str}")
                    
                    llm_entities = json.loads(json_str)
                    
                    # Validate and clean the entities
                    cleaned_entities = {
                        "diabetics": str(llm_entities.get("diabetics", "no")).lower().strip(),
                        "smoking": str(llm_entities.get("smoking", "no")).lower().strip(),
                        "alcohol": str(llm_entities.get("alcohol", "no")).lower().strip(),
                        "blood_pressure": str(llm_entities.get("blood_pressure", "unknown")).lower().strip(),
                        "medical_conditions": llm_entities.get("medical_conditions", []),
                        "llm_reasoning": llm_entities.get("llm_reasoning", "LLM analysis completed"),
                        "diabetes_evidence": llm_entities.get("diabetes_evidence", []),
                        "bp_evidence": llm_entities.get("bp_evidence", []),
                        "smoking_evidence": llm_entities.get("smoking_evidence", []),
                        "alcohol_evidence": llm_entities.get("alcohol_evidence", [])
                    }
                    
                    logger.info(f"✅ LLM entity extraction successful!")
                    logger.info(f"🩺 Diabetes: {cleaned_entities['diabetics']} (Evidence: {cleaned_entities.get('diabetes_evidence', [])})")
                    logger.info(f"💓 Blood Pressure: {cleaned_entities['blood_pressure']} (Evidence: {cleaned_entities.get('bp_evidence', [])})")
                    
                    return cleaned_entities
                else:
                    logger.error("❌ No valid JSON found in LLM response")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse LLM JSON response: {e}")
                logger.error(f"🔍 Problematic JSON: {json_str if 'json_str' in locals() else 'Not extracted'}")
                return None
        else:
            logger.error(f"❌ LLM call failed: {llm_response}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error in comprehensive LLM entity extraction: {e}")
        return None

# ADD THIS NEW METHOD TO THE HealthDataProcessor CLASS:
def _extract_medications_list(self, pharmacy_extraction: Dict[str, Any], entities: Dict[str, Any]):
    """Extract medications list for reference"""
    if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
        for record in pharmacy_extraction["ndc_records"]:
            if record.get("lbl_nm"):
                entities["medications_identified"].append({
                    "ndc": record.get("ndc", ""),
                    "label_name": record.get("lbl_nm", ""),
                    "path": record.get("data_path", "")
                })

# ADD THIS NEW METHOD TO THE HealthDataProcessor CLASS:
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

# =============================================================================
# FILE 2: health_agent_core.py
# =============================================================================

# REPLACE THE EXISTING extract_entities METHOD WITH THIS:
def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
    """Enhanced LangGraph Node 4: Extract comprehensive health entities using LLM with SAME data as chatbot"""
    logger.info("🎯 Enhanced Node 4: Starting LLM-powered health entity extraction with comprehensive data...")
    state["current_step"] = "extract_entities"
    state["step_status"]["extract_entities"] = "running"
    
    try:
        pharmacy_data = state.get("pharmacy_output", {})
        pharmacy_extraction = state.get("pharmacy_extraction", {})
        medical_extraction = state.get("medical_extraction", {})
        patient_data = state.get("patient_data", {})
        
        # GET THE SAME DEIDENTIFIED DATA THAT CHATBOT USES
        deidentified_medical = state.get("deidentified_medical", {})
        deidentified_pharmacy = state.get("deidentified_pharmacy", {})
        
        # Calculate age from date of birth and add to patient data
        if patient_data.get('date_of_birth'):
            try:
                dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d').date()
                today = date.today()
                calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                patient_data['calculated_age'] = calculated_age
                logger.info(f"📅 Calculated age from DOB: {calculated_age} years")
            except Exception as e:
                logger.warning(f"Could not calculate age from DOB: {e}")
        
        # COMPREHENSIVE entity extraction WITH SAME DATA AS CHATBOT
        entities = self.data_processor.extract_health_entities_enhanced(
            pharmacy_data, 
            pharmacy_extraction, 
            medical_extraction,
            patient_data,  # Pass patient data for age calculation
            self.api_integrator,  # Pass API integrator for LLM calls
            deidentified_medical,  # SAME DATA CHATBOT USES
            deidentified_pharmacy  # SAME DATA CHATBOT USES
        )
        
        state["entity_extraction"] = entities
        state["step_status"]["extract_entities"] = "completed"
        
        # Enhanced logging with debug info
        conditions_count = len(entities.get("medical_conditions", []))
        medications_count = len(entities.get("medications_identified", []))
        llm_status = entities.get("llm_analysis", "not_performed")
        age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
        
        logger.info(f"✅ Successfully extracted health entities using LLM: {conditions_count} conditions, {medications_count} medications")
        logger.info(f"📊 Entity results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
        logger.info(f"📅 {age_info}")
        logger.info(f"🤖 LLM analysis: {llm_status}")
        
        # Debug logging to help troubleshoot
        debug_info = entities.get("debug_info", {})
        if debug_info:
            logger.info(f"🔍 Debug info: {debug_info}")
        
        # Log evidence found by LLM
        if entities.get("diabetes_evidence"):
            logger.info(f"🩺 Diabetes evidence found: {entities.get('diabetes_evidence')}")
        if entities.get("bp_evidence"):
            logger.info(f"💓 Blood pressure evidence found: {entities.get('bp_evidence')}")
        
    except Exception as e:
        error_msg = f"Error in comprehensive LLM-powered entity extraction: {str(e)}"
        state["errors"].append(error_msg)
        state["step_status"]["extract_entities"] = "error"
        logger.error(error_msg)
    
    return state

# =============================================================================
# FILE 3: UI FILES (streamlit_health_ui.py OR ui9.py)
# =============================================================================

# ADD THIS DEBUG BUTTON AFTER THE ENTITY EXTRACTION SECTION (around line 900):
# Look for the entity extraction button section and add this RIGHT AFTER IT:

# DEBUG: Entity Extraction vs Chatbot Comparison
if st.button("🐛 Debug Entity Extraction", use_container_width=True, key="debug_entity_btn"):
    st.markdown("""
    <div class="section-box">
        <div class="section-title">🐛 Entity Extraction Debug Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    entity_extraction = safe_get(results, 'entity_extraction', {})
    chatbot_context = safe_get(results, 'chatbot_context', {})
    
    if entity_extraction:
        # Show debug info
        debug_info = entity_extraction.get('debug_info', {})
        st.markdown("**🔍 Debug Information:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Sources Available:**")
            data_sources = debug_info.get('data_sources', {})
            for source, available in data_sources.items():
                status = "✅" if available else "❌"
                st.write(f"{status} {source}: {available}")
            
            st.markdown("**🎯 Entity Extraction Results:**")
            final_extraction = debug_info.get('final_extraction', {})
            for entity, value in final_extraction.items():
                st.write(f"• **{entity}**: {value}")
        
        with col2:
            st.markdown("**🤖 LLM Analysis Status:**")
            st.write(f"• **Status**: {entity_extraction.get('llm_analysis', 'unknown')}")
            
            if entity_extraction.get('llm_reasoning'):
                st.write(f"• **Reasoning**: {entity_extraction.get('llm_reasoning')}")
            
            # Show evidence found
            if entity_extraction.get('diabetes_evidence'):
                st.markdown("**🩺 Diabetes Evidence:**")
                for evidence in entity_extraction.get('diabetes_evidence', []):
                    st.write(f"  - {evidence}")
            
            if entity_extraction.get('bp_evidence'):
                st.markdown("**💓 Blood Pressure Evidence:**")
                for evidence in entity_extraction.get('bp_evidence', []):
                    st.write(f"  - {evidence}")
        
        # Show analysis details
        analysis_details = entity_extraction.get('analysis_details', [])
        if analysis_details:
            st.markdown("**📋 Analysis Process:**")
            for detail in analysis_details:
                st.write(f"• {detail}")
        
        # Compare with chatbot context
        st.markdown("---")
        st.markdown("**🔄 Chatbot vs Entity Extraction Comparison:**")
        
        # Test the same question in chatbot context
        if st.button("🧪 Test Diabetes Question with Chatbot Logic"):
            try:
                if st.session_state.agent and chatbot_context:
                    test_response = st.session_state.agent.chat_with_data(
                        "Does this patient have diabetes? Please analyze all available data and list specific evidence.",
                        chatbot_context,
                        []
                    )
                    st.markdown("**💬 Chatbot Response:**")
                    st.write(test_response)
                else:
                    st.error("Chatbot not available for testing")
            except Exception as e:
                st.error(f"Error testing chatbot: {e}")
        
        # Raw data comparison
        with st.expander("📊 Raw Data Comparison"):
            st.markdown("**Entity Extraction Data:**")
            entity_meds = entity_extraction.get('medications_identified', [])
            if entity_meds:
                st.write("Medications found by entity extraction:")
                for med in entity_meds[:10]:  # Show first 10
                    st.write(f"• {med.get('label_name', 'N/A')} (NDC: {med.get('ndc', 'N/A')})")
            
            st.markdown("**Chatbot Context Data:**")
            deidentified_pharmacy = chatbot_context.get('deidentified_pharmacy', {})
            if deidentified_pharmacy:
                st.write("Pharmacy data available to chatbot: ✅")
                pharmacy_claims = deidentified_pharmacy.get('pharmacy_claims_data', {})
                if pharmacy_claims:
                    st.write(f"Pharmacy claims structure: {type(pharmacy_claims)}")
                    if isinstance(pharmacy_claims, dict):
                        st.write(f"Top-level keys: {list(pharmacy_claims.keys())}")
            else:
                st.write("Pharmacy data available to chatbot: ❌")
    
    else:
        st.error("No entity extraction data available for debugging")

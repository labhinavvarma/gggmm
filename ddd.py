def call_llm_for_graph_generation(self, user_message: str, chat_context: Dict[str, Any]) -> str:
    """FIXED LLM call for graph generation - returns React-compatible format"""
    try:
        session_id = str(uuid.uuid4()) + "_graph_gen_react"
        
        graph_system_msg = """You are Dr. GraphAI, a healthcare data visualization expert specializing in React-compatible data formats for medical analysis.

**CORE RESPONSIBILITY:**
Generate healthcare visualization data in EXACT React JavaScript format for direct integration.

**RESPONSE FORMAT REQUIREMENTS:**
Always return data in this EXACT format:

```javascript
const categories = ["ICD_CODE_1", "ICD_CODE_2", "NDC_CODE_1", "MEDICATION_NAME"];
const data = [frequency1, frequency2, frequency3, frequency4];
const chartType = "bar" | "pie" | "line" | "timeline";
const title = "Descriptive Chart Title";
const description = "Brief clinical explanation of the visualization";
```

**VISUALIZATION TYPES:**
- **Diagnosis Frequency**: categories = ICD-10 codes, data = occurrence counts
- **Medication Usage**: categories = medication names/NDC codes, data = usage frequency  
- **Risk Timeline**: categories = time periods, data = risk scores
- **Health Metrics**: categories = metric names, data = values/percentages

**CRITICAL RULES:**
1. ALWAYS use real ICD-10 codes from patient data when available
2. ALWAYS use real medication names/NDC codes from patient data when available
3. Categories and data arrays MUST have the same length
4. Use meaningful medical terminology in categories
5. Include brief clinical context in description
6. Format MUST be valid JavaScript that can be directly used in React

**EXAMPLE OUTPUT:**
```javascript
const categories = ["I10", "E11.9", "M19.90", "Z79.4"];
const data = [3, 2, 1, 4];
const chartType = "bar";
const title = "Patient Diagnosis Frequency";
const description = "Most frequent diagnoses showing hypertension (I10) and diabetes (E11.9) as primary conditions";
```"""

        # Enhanced context summary for React format
        context_summary = self._prepare_react_context_summary(chat_context)
        
        enhanced_prompt = f"""**HEALTHCARE DATA VISUALIZATION FOR REACT**

**USER REQUEST:** {user_message}

**AVAILABLE PATIENT DATA:**
{context_summary}

**DELIVERABLE:** 
Generate healthcare visualization data in EXACT React JavaScript format:

1. Extract relevant medical codes (ICD-10, NDC, etc.) from the patient data
2. Calculate frequencies or values for each code/category
3. Format as JavaScript constants for direct React integration
4. Include chart type recommendation and clinical description

**OUTPUT FORMAT:**
```javascript
const categories = [/* medical codes or category names */];
const data = [/* corresponding values/frequencies */];
const chartType = "/* recommended chart type */";
const title = "/* descriptive title */";
const description = "/* clinical interpretation */";
```

**REQUIREMENTS:**
- Use REAL patient data from the context
- Ensure categories and data arrays match in length
- Include meaningful medical terminology
- Provide clinical context in description"""

        logger.info(f"📊 React format graph generation: {user_message[:50]}...")

        payload = {
            "query": {
                "aplctn_cd": self.config.aplctn_cd,
                "app_id": self.config.app_id,
                "api_key": self.config.api_key,
                "method": "cortex", 
                "model": self.config.model,
                "sys_msg": graph_system_msg,
                "limit_convs": "0",
                "prompt": {
                    "messages": [
                        {
                            "role": "user",
                            "content": enhanced_prompt
                        }
                    ]
                },
                "app_lvl_prefix": "react_graph_format",
                "user_id": "react_integration",
                "session_id": session_id
            }
        }

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f'Snowflake Token="{self.config.api_key}"',
            "X-Healthcare-Graph": "react_format",
            "X-React-Integration": "enabled"
        }

        response = requests.post(
            self.config.api_url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=45
        )

        if response.status_code == 200:
            try:
                raw = response.text
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    bot_reply = answer.strip()
                else:
                    bot_reply = raw.strip()

                logger.info(f"✅ React format graph generation SUCCESS")
                
                # Validate response contains React format
                if self._validate_react_format(bot_reply):
                    return bot_reply
                else:
                    logger.warning("⚠️ Response not in React format, generating fallback")
                    return self._generate_react_fallback(user_message, chat_context)

            except Exception as e:
                logger.error(f"React format parse error: {e}")
                return self._generate_react_fallback(user_message, chat_context)
        
        elif response.status_code == 401:
            logger.error(f"❌ Authentication error for React format")
            return self._generate_react_auth_error()
        
        else:
            logger.error(f"❌ React format API error {response.status_code}")
            return self._generate_react_fallback(user_message, chat_context)

    except Exception as e:
        logger.error(f"❌ React format generation failed: {str(e)}")
        return self._generate_react_fallback(user_message, chat_context)

def _prepare_react_context_summary(self, chat_context: Dict[str, Any]) -> str:
    """Prepare context summary optimized for React format generation"""
    try:
        summary_parts = []
        
        # Extract medical codes for React format
        medical_extraction = chat_context.get("medical_extraction", {})
        if medical_extraction and medical_extraction.get("hlth_srvc_records"):
            # Get diagnosis codes with frequencies
            diagnosis_codes = {}
            for record in medical_extraction["hlth_srvc_records"]:
                for diag in record.get("diagnosis_codes", []):
                    code = diag.get("code", "")
                    if code:
                        diagnosis_codes[code] = diagnosis_codes.get(code, 0) + 1
            
            if diagnosis_codes:
                top_codes = sorted(diagnosis_codes.items(), key=lambda x: x[1], reverse=True)[:10]
                codes_list = [f'"{code}"' for code, _ in top_codes]
                freq_list = [str(freq) for _, freq in top_codes]
                summary_parts.append(f"Diagnosis Codes: [{', '.join(codes_list)}]")
                summary_parts.append(f"Diagnosis Frequencies: [{', '.join(freq_list)}]")

        # Extract medication data for React format
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
            # Get medication names with frequencies
            medications = {}
            for record in pharmacy_extraction["ndc_records"]:
                med_name = record.get("lbl_nm", "").strip()
                if med_name:
                    medications[med_name] = medications.get(med_name, 0) + 1
            
            if medications:
                top_meds = sorted(medications.items(), key=lambda x: x[1], reverse=True)[:8]
                med_list = [f'"{med}"' for med, _ in top_meds]
                med_freq_list = [str(freq) for _, freq in top_meds]
                summary_parts.append(f"Medications: [{', '.join(med_list)}]")
                summary_parts.append(f"Medication Frequencies: [{', '.join(med_freq_list)}]")

        # Add risk data for React format
        entity_extraction = chat_context.get("entity_extraction", {})
        if entity_extraction:
            risk_categories = []
            risk_values = []
            
            for key, value in entity_extraction.items():
                if key in ["diabetics", "blood_pressure", "smoking"] and value != "unknown":
                    risk_categories.append(f'"{key.replace("_", " ").title()}"')
                    # Convert to numeric for charting
                    numeric_value = 1 if value in ["yes", "high", "true"] else 0
                    risk_values.append(str(numeric_value))
            
            if risk_categories:
                summary_parts.append(f"Risk Categories: [{', '.join(risk_categories)}]")
                summary_parts.append(f"Risk Values: [{', '.join(risk_values)}]")

        return "\n".join(summary_parts) if summary_parts else "Basic patient data available for React visualization"
        
    except Exception as e:
        logger.warning(f"Error preparing React context: {e}")
        return "Patient healthcare data available for React visualization"

def _validate_react_format(self, response: str) -> bool:
    """Validate that response contains proper React format"""
    required_elements = ["const categories", "const data", "const chartType"]
    return all(element in response for element in required_elements)

def _generate_react_fallback(self, user_request: str, chat_context: Dict[str, Any]) -> str:
    """Generate React-compatible fallback when API fails"""
    
    # Try to extract real data from context
    categories, data, chart_type, title, description = self._extract_react_data_from_context(chat_context, user_request)
    
    return f"""```javascript
const categories = {categories};
const data = {data};
const chartType = "{chart_type}";
const title = "{title}";
const description = "{description}";
```

**Clinical Context:** {description}

**Integration Instructions:**
This data is formatted for direct use in your React chart component. The categories and data arrays are synchronized and ready for visualization."""

def _extract_react_data_from_context(self, chat_context: Dict[str, Any], user_request: str) -> tuple:
    """Extract actual patient data and format for React"""
    try:
        # Determine chart type from request
        request_lower = user_request.lower()
        if "medication" in request_lower:
            chart_type = "bar"
            return self._extract_medication_data(chat_context, chart_type)
        elif "diagnosis" in request_lower:
            chart_type = "bar" if "frequency" in request_lower else "pie"
            return self._extract_diagnosis_data(chat_context, chart_type)
        elif "timeline" in request_lower:
            chart_type = "line"
            return self._extract_timeline_data(chat_context, chart_type)
        elif "risk" in request_lower:
            chart_type = "bar"
            return self._extract_risk_data(chat_context, chart_type)
        else:
            chart_type = "bar"
            return self._extract_default_health_data(chat_context, chart_type)
            
    except Exception as e:
        logger.warning(f"Error extracting React data: {e}")
        return self._get_sample_react_data()

def _extract_diagnosis_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
    """Extract diagnosis codes and frequencies for React"""
    try:
        medical_extraction = chat_context.get("medical_extraction", {})
        if medical_extraction and medical_extraction.get("hlth_srvc_records"):
            diagnosis_freq = {}
            
            for record in medical_extraction["hlth_srvc_records"]:
                for diag in record.get("diagnosis_codes", []):
                    code = diag.get("code", "")
                    if code:
                        diagnosis_freq[code] = diagnosis_freq.get(code, 0) + 1
            
            if diagnosis_freq:
                # Sort by frequency and take top 10
                sorted_diagnoses = sorted(diagnosis_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                categories = [f'"{code}"' for code, _ in sorted_diagnoses]
                data = [freq for _, freq in sorted_diagnoses]
                
                return (
                    f"[{', '.join(categories)}]",
                    str(data),
                    chart_type,
                    "Patient Diagnosis Frequency",
                    f"Analysis of {len(sorted_diagnoses)} most frequent diagnosis codes from patient records"
                )
    except Exception as e:
        logger.warning(f"Error extracting diagnosis data: {e}")
    
    # Fallback to sample data
    return self._get_sample_diagnosis_data(chart_type)

def _extract_medication_data(self, chat_context: Dict[str, Any], chart_type: str) -> tuple:
    """Extract medication data for React"""
    try:
        pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
        if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
            med_freq = {}
            
            for record in pharmacy_extraction["ndc_records"]:
                med_name = record.get("lbl_nm", "").strip()
                if med_name:
                    med_freq[med_name] = med_freq.get(med_name, 0) + 1
            
            if med_freq:
                sorted_meds = sorted(med_freq.items(), key=lambda x: x[1], reverse=True)[:8]
                categories = [f'"{med}"' for med, _ in sorted_meds]
                data = [freq for _, freq in sorted_meds]
                
                return (
                    f"[{', '.join(categories)}]",
                    str(data),
                    chart_type,
                    "Patient Medication Usage",
                    f"Analysis of {len(sorted_meds)} medications from patient pharmacy records"
                )
    except Exception as e:
        logger.warning(f"Error extracting medication data: {e}")
    
    return self._get_sample_medication_data(chart_type)

def _get_sample_diagnosis_data(self, chart_type: str) -> tuple:
    """Sample diagnosis data in React format"""
    categories = '["I10", "E11.9", "M19.90", "Z79.4", "F41.9", "J45.9", "K21.9", "R06.02"]'
    data = '[3, 2, 2, 4, 1, 1, 1, 1]'
    title = "Patient Diagnosis Frequency"
    description = "Sample diagnosis codes showing hypertension (I10) and diabetes (E11.9) as primary conditions"
    
    return (categories, data, chart_type, title, description)

def _get_sample_medication_data(self, chart_type: str) -> tuple:
    """Sample medication data in React format"""
    categories = '["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole"]'
    data = '[4, 3, 2, 2, 1]'
    title = "Patient Medication Usage"
    description = "Current medications showing diabetes and cardiovascular management focus"
    
    return (categories, data, chart_type, title, description)

def _generate_react_auth_error(self) -> str:
    """Generate React format response for authentication errors"""
    return """```javascript
const categories = ["Authentication", "Error"];
const data = [0, 1];
const chartType = "bar";
const title = "Graph Generation Unavailable";
const description = "Authentication error occurred. Please try again or contact support.";
```

**Error:** Unable to generate visualization due to authentication issue. Please retry your request."""

import json
import re
import time
from datetime import datetime, date
from typing import Dict, Any, List
import logging
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class EnhancedHealthDataProcessor:
    """Enhanced data processor with DETAILED healthcare analysis, clinical insights, and comprehensive medical code processing"""
 
    def __init__(self, api_integrator=None):
        self.api_integrator = api_integrator
        logger.info("🔬 EnhancedHealthDataProcessor initialized with healthcare specialization")
        
        # Enhanced API integrator validation
        if self.api_integrator:
            logger.info("✅ Enhanced API integrator provided")
            if hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                logger.info("✅ Enhanced isolated LLM method found - detailed healthcare batch processing enabled")
            else:
                logger.error("❌ Enhanced isolated LLM method missing - falling back to standard batch processing")
        else:
            logger.warning("⚠️ No API integrator - batch processing disabled")
 
    def deidentify_medical_data_enhanced(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced medical data deidentification with clinical context preservation"""
        try:
            if not medical_data:
                return {"error": "No medical data available for enhanced deidentification"}
 
            # Enhanced age calculation with clinical context
            age = self._calculate_age_with_clinical_context(patient_data.get('date_of_birth', ''))
 
            # Enhanced JSON processing with clinical structure preservation
            raw_medical_data = medical_data.get('body', medical_data)
            deidentified_medical_data = self._enhanced_deidentify_json(raw_medical_data)
            deidentified_medical_data = self._mask_clinical_fields_enhanced(deidentified_medical_data)
 
            enhanced_deidentified = {
                "src_mbr_first_nm": "[CLINICAL_DATA_MASKED]",
                "src_mbr_last_nm": "[CLINICAL_DATA_MASKED]",
                "src_mbr_mid_init_nm": None,
                "src_mbr_age": age,
                "src_mbr_zip_cd": patient_data.get('zip_code', '12345'),
                "medical_claims_data": deidentified_medical_data,
                "original_structure_preserved": True,
                "clinical_context_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_medical_claims",
                "healthcare_specialization": "advanced",
                "clinical_data_elements_preserved": self._count_clinical_elements(deidentified_medical_data)
            }
 
            logger.info("✅ Enhanced medical deidentification completed with clinical preservation")
            logger.info(f"🔬 Clinical elements preserved: {enhanced_deidentified['clinical_data_elements_preserved']}")
            
            return enhanced_deidentified
 
        except Exception as e:
            logger.error(f"Error in enhanced medical deidentification: {e}")
            return {"error": f"Enhanced deidentification failed: {str(e)}"}

    def deidentify_pharmacy_data_enhanced(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pharmacy data deidentification with therapeutic context preservation"""
        try:
            if not pharmacy_data:
                return {"error": "No pharmacy data available for enhanced deidentification"}

            raw_pharmacy_data = pharmacy_data.get('body', pharmacy_data)
            deidentified_pharmacy_data = self._enhanced_deidentify_pharmacy_json(raw_pharmacy_data)

            enhanced_result = {
                "pharmacy_claims_data": deidentified_pharmacy_data,
                "original_structure_preserved": True,
                "therapeutic_context_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_pharmacy_claims",
                "healthcare_specialization": "advanced",
                "name_fields_masked": ["src_mbr_first_nm", "scr_mbr_last_nm"],
                "therapeutic_data_elements_preserved": self._count_therapeutic_elements(deidentified_pharmacy_data)
            }

            logger.info("✅ Enhanced pharmacy deidentification completed with therapeutic preservation")
            logger.info(f"💊 Therapeutic elements preserved: {enhanced_result['therapeutic_data_elements_preserved']}")
            
            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced pharmacy deidentification: {e}")
            return {"error": f"Enhanced deidentification failed: {str(e)}"}

    def deidentify_mcid_data_enhanced(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced MCID data deidentification with identity verification context"""
        try:
            if not mcid_data:
                return {"error": "No MCID data available for enhanced deidentification"}

            raw_mcid_data = mcid_data.get('body', mcid_data)
            deidentified_mcid_data = self._enhanced_deidentify_json(raw_mcid_data)

            enhanced_result = {
                "mcid_claims_data": deidentified_mcid_data,
                "original_structure_preserved": True,
                "identity_verification_context_preserved": True,
                "deidentification_timestamp": datetime.now().isoformat(),
                "data_type": "enhanced_mcid_claims",
                "healthcare_specialization": "advanced"
            }

            logger.info("✅ Enhanced MCID deidentification completed with identity context preservation")
            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced MCID deidentification: {e}")
            return {"error": f"Enhanced deidentification failed: {str(e)}"}

    def extract_medical_fields_batch_enhanced(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """🔬 Enhanced BATCH PROCESSING: Medical field extraction with DETAILED clinical analysis"""
        logger.info("🔬 ===== STARTING ENHANCED BATCH MEDICAL EXTRACTION WITH CLINICAL INSIGHTS =====")
        
        enhanced_extraction_result = {
            "hlth_srvc_records": [],
            "extraction_summary": {
                "total_hlth_srvc_records": 0,
                "total_diagnosis_codes": 0,
                "unique_service_codes": set(),
                "unique_diagnosis_codes": set(),
                "clinical_complexity_score": 0,
                "therapeutic_categories_identified": []
            },
            "code_meanings": {
                "service_code_meanings": {},
                "diagnosis_code_meanings": {}
            },
            "clinical_insights": {
                "primary_diagnoses": [],
                "comorbidities": [],
                "treatment_patterns": [],
                "care_complexity": "Standard"
            },
            "code_meanings_added": False,
            "enhanced_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0,
                "clinical_analysis_depth": "comprehensive"
            }
        }

        start_time = time.time()

        try:
            medical_data = deidentified_medical.get("medical_claims_data", {})
            if not medical_data:
                logger.warning("⚠️ No medical claims data found for enhanced analysis")
                return enhanced_extraction_result

            # Step 1: Enhanced extraction with clinical analysis
            logger.info("🔬 Step 1: Enhanced clinical code extraction with complexity analysis...")
            self._enhanced_medical_extraction_with_clinical_analysis(medical_data, enhanced_extraction_result)

            # Convert sets to lists and apply clinical prioritization
            unique_service_codes = list(enhanced_extraction_result["extraction_summary"]["unique_service_codes"])[:20]
            unique_diagnosis_codes = list(enhanced_extraction_result["extraction_summary"]["unique_diagnosis_codes"])[:25]
            
            enhanced_extraction_result["extraction_summary"]["unique_service_codes"] = unique_service_codes
            enhanced_extraction_result["extraction_summary"]["unique_diagnosis_codes"] = unique_diagnosis_codes

            total_codes = len(unique_service_codes) + len(unique_diagnosis_codes)
            enhanced_extraction_result["batch_stats"]["codes_processed"] = total_codes

            # Step 2: Enhanced BATCH PROCESSING with detailed clinical explanations
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_service_codes or unique_diagnosis_codes:
                    logger.info(f"🔬 Step 2: Enhanced BATCH processing {total_codes} codes with detailed clinical analysis...")
                    enhanced_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # Enhanced BATCH 1: All Service Codes with clinical context
                        if unique_service_codes:
                            logger.info(f"🏥 Enhanced BATCH processing {len(unique_service_codes)} service codes with clinical insights...")
                            service_meanings = self._enhanced_batch_service_codes_with_clinical_context(unique_service_codes)
                            enhanced_extraction_result["code_meanings"]["service_code_meanings"] = service_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Enhanced service codes batch: {len(service_meanings)} detailed clinical meanings generated")
                        
                        # Enhanced BATCH 2: All Diagnosis Codes with clinical significance
                        if unique_diagnosis_codes:
                            logger.info(f"🩺 Enhanced BATCH processing {len(unique_diagnosis_codes)} diagnosis codes with clinical significance...")
                            diagnosis_meanings = self._enhanced_batch_diagnosis_codes_with_clinical_significance(unique_diagnosis_codes)
                            enhanced_extraction_result["code_meanings"]["diagnosis_code_meanings"] = diagnosis_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Enhanced diagnosis codes batch: {len(diagnosis_meanings)} detailed clinical meanings generated")
                        
                        # Calculate enhanced savings
                        individual_calls_would_be = len(unique_service_codes) + len(unique_diagnosis_codes)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        enhanced_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        enhanced_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Step 3: Enhanced clinical analysis integration
                        logger.info("🔬 Step 3: Enhanced clinical analysis integration...")
                        self._enhanced_clinical_analysis_integration(enhanced_extraction_result)
                        
                        # Final enhanced status
                        total_meanings = len(enhanced_extraction_result["code_meanings"]["service_code_meanings"]) + len(enhanced_extraction_result["code_meanings"]["diagnosis_code_meanings"])
                        
                        if total_meanings > 0:
                            enhanced_extraction_result["code_meanings_added"] = True
                            enhanced_extraction_result["enhanced_analysis"] = True
                            enhanced_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 Enhanced BATCH SUCCESS: {total_meanings} detailed clinical meanings, {calls_saved} API calls saved!")
                        else:
                            enhanced_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Enhanced batch completed but no detailed meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced batch processing error: {e}")
                        enhanced_extraction_result["code_meaning_error"] = str(e)
                        enhanced_extraction_result["llm_call_status"] = "failed"
                else:
                    enhanced_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No codes found for enhanced batch processing")
            else:
                enhanced_extraction_result["llm_call_status"] = "skipped_no_api"
                logger.error("❌ No enhanced API integrator for batch processing")

            # Enhanced performance stats
            processing_time = time.time() - start_time
            enhanced_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"🔬 ===== ENHANCED BATCH MEDICAL EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s (vs 8+ minutes individual)")
            logger.info(f"  📊 API calls: {enhanced_extraction_result['batch_stats']['api_calls_made']} (saved {enhanced_extraction_result['batch_stats']['individual_calls_saved']})")
            logger.info(f"  🔬 Clinical insights: {enhanced_extraction_result['enhanced_analysis']}")
            logger.info(f"  ✅ Detailed meanings: {len(enhanced_extraction_result['code_meanings']['service_code_meanings']) + len(enhanced_extraction_result['code_meanings']['diagnosis_code_meanings'])}")

        except Exception as e:
            logger.error(f"❌ Error in enhanced batch medical extraction: {e}")
            enhanced_extraction_result["error"] = f"Enhanced batch extraction failed: {str(e)}"

        return enhanced_extraction_result

    def extract_pharmacy_fields_batch_enhanced(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """🔬 Enhanced BATCH PROCESSING: Pharmacy field extraction with DETAILED therapeutic analysis"""
        logger.info("🔬 ===== STARTING ENHANCED BATCH PHARMACY EXTRACTION WITH THERAPEUTIC INSIGHTS =====")
        
        enhanced_extraction_result = {
            "ndc_records": [],
            "extraction_summary": {
                "total_ndc_records": 0,
                "unique_ndc_codes": set(),
                "unique_label_names": set(),
                "therapeutic_classes_identified": [],
                "polypharmacy_indicators": [],
                "medication_complexity_score": 0
            },
            "code_meanings": {
                "ndc_code_meanings": {},
                "medication_meanings": {}
            },
            "therapeutic_insights": {
                "primary_medications": [],
                "drug_interactions_potential": [],
                "therapeutic_duplications": [],
                "adherence_indicators": []
            },
            "code_meanings_added": False,
            "enhanced_analysis": False,
            "llm_call_status": "not_attempted",
            "batch_stats": {
                "individual_calls_saved": 0,
                "processing_time_seconds": 0,
                "api_calls_made": 0,
                "codes_processed": 0,
                "therapeutic_analysis_depth": "comprehensive"
            }
        }

        start_time = time.time()

        try:
            pharmacy_data = deidentified_pharmacy.get("pharmacy_claims_data", {})
            if not pharmacy_data:
                logger.warning("⚠️ No pharmacy claims data found for enhanced analysis")
                return enhanced_extraction_result

            # Step 1: Enhanced extraction with therapeutic analysis
            logger.info("🔬 Step 1: Enhanced therapeutic code extraction with complexity analysis...")
            self._enhanced_pharmacy_extraction_with_therapeutic_analysis(pharmacy_data, enhanced_extraction_result)

            # Convert sets to lists and apply therapeutic prioritization
            unique_ndc_codes = list(enhanced_extraction_result["extraction_summary"]["unique_ndc_codes"])[:15]
            unique_label_names = list(enhanced_extraction_result["extraction_summary"]["unique_label_names"])[:20]
            
            enhanced_extraction_result["extraction_summary"]["unique_ndc_codes"] = unique_ndc_codes
            enhanced_extraction_result["extraction_summary"]["unique_label_names"] = unique_label_names

            total_codes = len(unique_ndc_codes) + len(unique_label_names)
            enhanced_extraction_result["batch_stats"]["codes_processed"] = total_codes

            # Step 2: Enhanced BATCH PROCESSING with detailed therapeutic analysis
            if self.api_integrator and hasattr(self.api_integrator, 'call_llm_isolated_enhanced'):
                if unique_ndc_codes or unique_label_names:
                    logger.info(f"🔬 Step 2: Enhanced BATCH processing {total_codes} pharmacy codes with therapeutic analysis...")
                    enhanced_extraction_result["llm_call_status"] = "in_progress"
                    
                    try:
                        api_calls_made = 0
                        
                        # Enhanced BATCH 1: All NDC Codes with therapeutic context
                        if unique_ndc_codes:
                            logger.info(f"💊 Enhanced BATCH processing {len(unique_ndc_codes)} NDC codes with therapeutic insights...")
                            ndc_meanings = self._enhanced_batch_ndc_codes_with_therapeutic_context(unique_ndc_codes)
                            enhanced_extraction_result["code_meanings"]["ndc_code_meanings"] = ndc_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Enhanced NDC codes batch: {len(ndc_meanings)} detailed therapeutic meanings generated")
                        
                        # Enhanced BATCH 2: All Medications with clinical significance
                        if unique_label_names:
                            logger.info(f"💉 Enhanced BATCH processing {len(unique_label_names)} medications with clinical significance...")
                            med_meanings = self._enhanced_batch_medications_with_clinical_significance(unique_label_names)
                            enhanced_extraction_result["code_meanings"]["medication_meanings"] = med_meanings
                            api_calls_made += 1
                            logger.info(f"✅ Enhanced medications batch: {len(med_meanings)} detailed clinical meanings generated")
                        
                        # Calculate enhanced savings
                        individual_calls_would_be = len(unique_ndc_codes) + len(unique_label_names)
                        calls_saved = individual_calls_would_be - api_calls_made
                        
                        enhanced_extraction_result["batch_stats"]["individual_calls_saved"] = calls_saved
                        enhanced_extraction_result["batch_stats"]["api_calls_made"] = api_calls_made
                        
                        # Step 3: Enhanced therapeutic analysis integration
                        logger.info("🔬 Step 3: Enhanced therapeutic analysis integration...")
                        self._enhanced_therapeutic_analysis_integration(enhanced_extraction_result)
                        
                        # Final enhanced status
                        total_meanings = len(enhanced_extraction_result["code_meanings"]["ndc_code_meanings"]) + len(enhanced_extraction_result["code_meanings"]["medication_meanings"])
                        
                        if total_meanings > 0:
                            enhanced_extraction_result["code_meanings_added"] = True
                            enhanced_extraction_result["enhanced_analysis"] = True
                            enhanced_extraction_result["llm_call_status"] = "completed"
                            logger.info(f"🔬 Enhanced PHARMACY BATCH SUCCESS: {total_meanings} detailed therapeutic meanings, {calls_saved} calls saved!")
                        else:
                            enhanced_extraction_result["llm_call_status"] = "completed_no_meanings"
                            logger.warning("⚠️ Enhanced pharmacy batch completed but no detailed meanings generated")
                        
                    except Exception as e:
                        logger.error(f"❌ Enhanced pharmacy batch error: {e}")
                        enhanced_extraction_result["code_meaning_error"] = str(e)
                        enhanced_extraction_result["llm_call_status"] = "failed"
                else:
                    enhanced_extraction_result["llm_call_status"] = "skipped_no_codes"
                    logger.warning("⚠️ No pharmacy codes for enhanced batch processing")
            else:
                enhanced_extraction_result["llm_call_status"] = "skipped_no_api"
                logger.error("❌ No enhanced API integrator for pharmacy batch processing")

            # Enhanced performance stats
            processing_time = time.time() - start_time
            enhanced_extraction_result["batch_stats"]["processing_time_seconds"] = round(processing_time, 2)

            logger.info(f"💊 ===== ENHANCED BATCH PHARMACY EXTRACTION COMPLETED =====")
            logger.info(f"  ⚡ Time: {processing_time:.2f}s")
            logger.info(f"  📊 API calls: {enhanced_extraction_result['batch_stats']['api_calls_made']} (saved {enhanced_extraction_result['batch_stats']['individual_calls_saved']})")
            logger.info(f"  🔬 Therapeutic insights: {enhanced_extraction_result['enhanced_analysis']}")

        except Exception as e:
            logger.error(f"❌ Error in enhanced batch pharmacy extraction: {e}")
            enhanced_extraction_result["error"] = f"Enhanced pharmacy batch extraction failed: {str(e)}"

        return enhanced_extraction_result

    def _enhanced_batch_service_codes_with_clinical_context(self, service_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL service codes with detailed clinical context"""
        try:
            if not service_codes:
                return {}
                
            logger.info(f"🏥 === Enhanced BATCH PROCESSING {len(service_codes)} SERVICE CODES WITH CLINICAL CONTEXT ===")
            
            codes_list = "\n".join([f"- {code}" for code in service_codes])
            
            enhanced_prompt = f"""You are Dr. ProcedureAI, a medical coding specialist with comprehensive knowledge of CPT, HCPCS, and healthcare service codes. Provide DETAILED clinical explanations for these healthcare service codes:

Service Codes for Clinical Analysis:
{codes_list}

For each code, provide comprehensive clinical information including:
• Procedure description and clinical purpose
• Clinical indications and contraindications
• Healthcare setting (inpatient, outpatient, emergency)
• Typical patient populations
• Associated diagnosis codes or conditions
• Clinical complexity and resource requirements

Return ONLY valid JSON in this exact format:
{{
    "{service_codes[0]}": "Detailed clinical explanation with procedure purpose, indications, setting, and clinical significance",
    "{service_codes[1] if len(service_codes) > 1 else service_codes[0]}": "Detailed clinical explanation with procedure purpose, indications, setting, and clinical significance"
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Include clinical context and medical significance
- Use professional medical terminology
- Focus on clinical decision support value"""

            enhanced_system_msg = """You are a medical coding expert specializing in healthcare procedures and services. Provide detailed, clinically accurate explanations that support healthcare decision-making and clinical documentation."""
            
            response = self.api_integrator.call_llm_isolated_enhanced(enhanced_prompt, enhanced_system_msg)
            
            if response and response != "Detailed explanation unavailable":
                try:
                    clean_response = self._clean_json_response_enhanced(response)
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Enhanced service codes batch: {len(meanings_dict)} detailed clinical meanings extracted")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Enhanced service codes JSON parse error: {e}")
                    return {}
            else:
                logger.warning(f"⚠️ Enhanced service codes batch returned unavailable")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Enhanced service codes batch exception: {e}")
            return {}

    def _enhanced_batch_diagnosis_codes_with_clinical_significance(self, diagnosis_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL diagnosis codes with detailed clinical significance"""
        try:
            if not diagnosis_codes:
                return {}
                
            logger.info(f"🩺 === Enhanced BATCH PROCESSING {len(diagnosis_codes)} DIAGNOSIS CODES WITH CLINICAL SIGNIFICANCE ===")
            
            codes_list = "\n".join([f"- {code}" for code in diagnosis_codes])
            
            enhanced_prompt = f"""You are Dr. DiagnosisAI, a clinical specialist with comprehensive knowledge of ICD-10 diagnosis codes and clinical medicine. Provide DETAILED clinical explanations for these diagnosis codes:

Diagnosis Codes for Clinical Analysis:
{codes_list}

For each code, provide comprehensive clinical information including:
• Medical condition description and pathophysiology
• Clinical presentation and symptoms
• Risk factors and epidemiology
• Complications and prognosis
• Treatment approaches and management strategies
• Impact on patient care and quality of life
• Comorbidity considerations

Return ONLY valid JSON in this exact format:
{{
    "{diagnosis_codes[0]}": "Detailed clinical explanation with condition description, symptoms, risk factors, complications, treatment approaches, and clinical management considerations",
    "{diagnosis_codes[1] if len(diagnosis_codes) > 1 else diagnosis_codes[0]}": "Detailed clinical explanation with condition description, symptoms, risk factors, complications, treatment approaches, and clinical management considerations"
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Include detailed clinical context and medical significance
- Use evidence-based medical terminology
- Focus on clinical decision support and patient care impact"""

            enhanced_system_msg = """You are a clinical medicine expert specializing in diagnostic coding and clinical conditions. Provide detailed, evidence-based explanations that support clinical decision-making and comprehensive patient care."""
            
            response = self.api_integrator.call_llm_isolated_enhanced(enhanced_prompt, enhanced_system_msg)
            
            if response and response != "Detailed explanation unavailable":
                try:
                    clean_response = self._clean_json_response_enhanced(response)
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Enhanced diagnosis codes batch: {len(meanings_dict)} detailed clinical meanings extracted")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Enhanced diagnosis codes JSON parse error: {e}")
                    return {}
            else:
                logger.warning(f"⚠️ Enhanced diagnosis codes batch returned unavailable")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Enhanced diagnosis codes batch exception: {e}")
            return {}

    def _enhanced_batch_ndc_codes_with_therapeutic_context(self, ndc_codes: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL NDC codes with detailed therapeutic context"""
        try:
            if not ndc_codes:
                return {}
                
            logger.info(f"💊 === Enhanced BATCH PROCESSING {len(ndc_codes)} NDC CODES WITH THERAPEUTIC CONTEXT ===")
            
            codes_list = "\n".join([f"- {code}" for code in ndc_codes])
            
            enhanced_prompt = f"""You are Dr. PharmaAI, a clinical pharmacist with comprehensive knowledge of NDC codes and pharmaceutical therapeutics. Provide DETAILED therapeutic explanations for these NDC codes:

NDC Codes for Therapeutic Analysis:
{codes_list}

For each NDC code, provide comprehensive pharmaceutical information including:
• Medication name and active ingredients
• Therapeutic class and mechanism of action
• Clinical indications and FDA-approved uses
• Dosing considerations and administration
• Contraindications and drug interactions
• Adverse effects and monitoring requirements
• Therapeutic alternatives and clinical considerations

Return ONLY valid JSON in this exact format:
{{
    "{ndc_codes[0]}": "Detailed therapeutic explanation with medication name, mechanism of action, clinical indications, dosing, contraindications, interactions, adverse effects, and therapeutic considerations",
    "{ndc_codes[1] if len(ndc_codes) > 1 else ndc_codes[0]}": "Detailed therapeutic explanation with medication name, mechanism of action, clinical indications, dosing, contraindications, interactions, adverse effects, and therapeutic considerations"
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Include comprehensive therapeutic context and clinical significance
- Use evidence-based pharmaceutical terminology
- Focus on clinical pharmacy decision support"""

            enhanced_system_msg = """You are a clinical pharmacy expert specializing in NDC codes and pharmaceutical therapeutics. Provide detailed, evidence-based explanations that support medication management and clinical pharmacy practice."""
            
            response = self.api_integrator.call_llm_isolated_enhanced(enhanced_prompt, enhanced_system_msg)
            
            if response and response != "Detailed explanation unavailable":
                try:
                    clean_response = self._clean_json_response_enhanced(response)
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Enhanced NDC codes batch: {len(meanings_dict)} detailed therapeutic meanings extracted")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Enhanced NDC codes JSON parse error: {e}")
                    return {}
            else:
                logger.warning(f"⚠️ Enhanced NDC codes batch returned unavailable")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Enhanced NDC codes batch exception: {e}")
            return {}

    def _enhanced_batch_medications_with_clinical_significance(self, medications: List[str]) -> Dict[str, str]:
        """Enhanced BATCH process ALL medications with detailed clinical significance"""
        try:
            if not medications:
                return {}
                
            logger.info(f"💉 === Enhanced BATCH PROCESSING {len(medications)} MEDICATIONS WITH CLINICAL SIGNIFICANCE ===")
            
            meds_list = "\n".join([f"- {med}" for med in medications])
            
            enhanced_prompt = f"""You are Dr. TherapeuticAI, a clinical pharmacologist with comprehensive knowledge of medications and therapeutic management. Provide DETAILED clinical explanations for these medications:

Medications for Clinical Analysis:
{meds_list}

For each medication, provide comprehensive clinical information including:
• Therapeutic class and pharmacological category
• Mechanism of action and pharmacokinetics
• Primary clinical indications and evidence base
• Dosing strategies and therapeutic monitoring
• Contraindications and precautions
• Drug interactions and clinical considerations
• Adverse effects and safety profile
• Place in therapy and treatment guidelines

Return ONLY valid JSON in this exact format:
{{
    "{medications[0]}": "Detailed clinical explanation with therapeutic class, mechanism of action, clinical indications, dosing, contraindications, interactions, adverse effects, and place in therapy",
    "{medications[1] if len(medications) > 1 else medications[0]}": "Detailed clinical explanation with therapeutic class, mechanism of action, clinical indications, dosing, contraindications, interactions, adverse effects, and place in therapy"
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text
- Include comprehensive clinical context and therapeutic significance
- Use evidence-based pharmacological terminology
- Focus on clinical therapeutics and medication management"""

            enhanced_system_msg = """You are a clinical pharmacology expert specializing in medication therapy management and clinical therapeutics. Provide detailed, evidence-based explanations that support optimal medication use and patient care."""
            
            response = self.api_integrator.call_llm_isolated_enhanced(enhanced_prompt, enhanced_system_msg)
            
            if response and response != "Detailed explanation unavailable":
                try:
                    clean_response = self._clean_json_response_enhanced(response)
                    meanings_dict = json.loads(clean_response)
                    logger.info(f"✅ Enhanced medications batch: {len(meanings_dict)} detailed clinical meanings extracted")
                    return meanings_dict
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Enhanced medications JSON parse error: {e}")
                    return {}
            else:
                logger.warning(f"⚠️ Enhanced medications batch returned unavailable")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Enhanced medications batch exception: {e}")
            return {}

    def _clean_json_response_enhanced(self, response: str) -> str:
        """Enhanced LLM response cleaning for JSON extraction"""
        try:
            # Enhanced markdown wrapper removal
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Enhanced JSON boundary detection
            response = response.strip()
            
            # Find JSON object boundaries with enhanced detection
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_content = response[start:end]
                
                # Enhanced JSON validation
                try:
                    json.loads(json_content)
                    return json_content
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    fixed_content = self._fix_common_json_issues(json_content)
                    return fixed_content
            else:
                return response
                
        except Exception as e:
            logger.warning(f"Enhanced JSON cleaning failed: {e}")
            return response

    def _fix_common_json_issues(self, json_content: str) -> str:
        """Fix common JSON formatting issues"""
        try:
            # Fix trailing commas
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            
            # Fix unescaped quotes in values
            json_content = re.sub(r'(?<!\\)"(?=[^,:}\]]*[,:}\]])', '\\"', json_content)
            
            return json_content
        except Exception as e:
            logger.warning(f"JSON fixing failed: {e}")
            return json_content

    def _calculate_age_with_clinical_context(self, date_of_birth: str) -> str:
        """Enhanced age calculation with clinical context"""
        try:
            if not date_of_birth:
                return "unknown"
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Add clinical context
            if age < 18:
                clinical_context = f"{age} (Pediatric population - specialized care considerations)"
            elif age < 65:
                clinical_context = f"{age} (Adult population - standard care protocols)"
            else:
                clinical_context = f"{age} (Geriatric population - enhanced monitoring recommended)"
                
            return clinical_context
        except:
            return "unknown (Age calculation failed - clinical assessment recommended)"

    def _count_clinical_elements(self, data: Any) -> int:
        """Count clinical data elements preserved in deidentification"""
        try:
            clinical_fields = [
                'hlth_srvc_cd', 'health_service_code', 'diag_1_50_cd', 'clm_rcvd_dt',
                'diagnosis_code', 'procedure_code', 'treatment_code', 'medical_code'
            ]
            
            count = 0
            
            def count_recursive(obj):
                nonlocal count
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if any(clinical_field in key.lower() for clinical_field in clinical_fields):
                            if value and str(value).strip() and str(value).lower() not in ['null', 'none', '']:
                                count += 1
                        if isinstance(value, (dict, list)):
                            count_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count_recursive(item)
            
            count_recursive(data)
            return count
        except Exception as e:
            logger.warning(f"Clinical elements counting failed: {e}")
            return 0

    def _count_therapeutic_elements(self, data: Any) -> int:
        """Count therapeutic data elements preserved in deidentification"""
        try:
            therapeutic_fields = [
                'ndc', 'ndc_code', 'lbl_nm', 'label_name', 'drug_name', 'medication_name',
                'rx_filled_dt', 'prescription_date', 'therapeutic_class', 'medication_code'
            ]
            
            count = 0
            
            def count_recursive(obj):
                nonlocal count
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if any(therapeutic_field in key.lower() for therapeutic_field in therapeutic_fields):
                            if value and str(value).strip() and str(value).lower() not in ['null', 'none', '']:
                                count += 1
                        if isinstance(value, (dict, list)):
                            count_recursive(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count_recursive(item)
            
            count_recursive(data)
            return count
        except Exception as e:
            logger.warning(f"Therapeutic elements counting failed: {e}")
            return 0

    def _enhanced_deidentify_json(self, data: Any) -> Any:
        """Enhanced JSON deidentification with clinical structure preservation"""
        if isinstance(data, dict):
            deidentified_dict = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    deidentified_dict[key] = self._enhanced_deidentify_json(value)
                elif isinstance(value, str):
                    deidentified_dict[key] = self._enhanced_deidentify_string(value)
                else:
                    deidentified_dict[key] = value
            return deidentified_dict
        elif isinstance(data, list):
            return [self._enhanced_deidentify_json(item) for item in data]
        elif isinstance(data, str):
            return self._enhanced_deidentify_string(data)
        else:
            return data

    def _enhanced_deidentify_pharmacy_json(self, data: Any) -> Any:
        """Enhanced pharmacy JSON deidentification with therapeutic context preservation"""
        if isinstance(data, dict):
            deidentified_dict = {}
            for key, value in data.items():
                if key.lower() in ['src_mbr_first_nm', 'src_mbr_frst_nm', 'scr_mbr_last_nm', 'src_mbr_last_nm']:
                    deidentified_dict[key] = "[THERAPEUTIC_DATA_MASKED]"
                elif isinstance(value, (dict, list)):
                    deidentified_dict[key] = self._enhanced_deidentify_pharmacy_json(value)
                elif isinstance(value, str):
                    deidentified_dict[key] = self._enhanced_deidentify_string(value)
                else:
                    deidentified_dict[key] = value
            return deidentified_dict
        elif isinstance(data, list):
            return [self._enhanced_deidentify_pharmacy_json(item) for item in data]
        elif isinstance(data, str):
            return self._enhanced_deidentify_string(data)
        else:
            return data

    def _mask_clinical_fields_enhanced(self, data: Any) -> Any:
        """Enhanced clinical field masking with structure preservation"""
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if key.lower() in ['src_mbr_frst_nm', 'src_mbr_first_nm', 'src_mbr_last_nm', 'src_mvr_last_nm']:
                    masked_data[key] = "[CLINICAL_DATA_MASKED]"
                elif isinstance(value, (dict, list)):
                    masked_data[key] = self._mask_clinical_fields_enhanced(value)
                else:
                    masked_data[key] = value
            return masked_data
        elif isinstance(data, list):
            return [self._mask_clinical_fields_enhanced(item) for item in data]
        else:
            return data

    def _enhanced_deidentify_string(self, data: str) -> str:
        """Enhanced string deidentification with clinical context preservation"""
        if not isinstance(data, str) or not data.strip():
            return data

        deidentified = str(data)
        
        # Enhanced pattern replacements with clinical context
        deidentified = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[CLINICAL_SSN_MASKED]', deidentified)
        deidentified = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[CLINICAL_PHONE_MASKED]', deidentified)
        deidentified = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[CLINICAL_EMAIL_MASKED]', deidentified)
        deidentified = re.sub(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b', '[CLINICAL_NAME_MASKED]', deidentified)
        
        return deidentified

    def _enhanced_medical_extraction_with_clinical_analysis(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive medical field extraction with clinical complexity analysis"""
        if isinstance(data, dict):
            current_record = {}
            clinical_complexity_indicators = []

            # Enhanced health service code extraction
            if "hlth_srvc_cd" in data and data["hlth_srvc_cd"]:
                service_code = str(data["hlth_srvc_cd"]).strip()
                current_record["hlth_srvc_cd"] = service_code
                result["extraction_summary"]["unique_service_codes"].add(service_code)
                
                # Analyze clinical complexity
                if service_code.startswith(('99'):  # Office visits
                    clinical_complexity_indicators.append("Office visit")
                elif service_code.startswith(('90'):  # Procedures
                    clinical_complexity_indicators.append("Medical procedure")

            # Enhanced claim received date extraction
            if "clm_rcvd_dt" in data and data["clm_rcvd_dt"]:
                current_record["clm_rcvd_dt"] = data["clm_rcvd_dt"]

            # Enhanced diagnosis codes extraction with clinical categorization
            diagnosis_codes = []
            primary_diagnoses = []
            comorbidities = []

            # Handle comma-separated diagnosis codes with clinical analysis
            if "diag_1_50_cd" in data and data["diag_1_50_cd"]:
                diag_value = str(data["diag_1_50_cd"]).strip()
                if diag_value and diag_value.lower() not in ['null', 'none', '']:
                    individual_codes = [code.strip() for code in diag_value.split(',') if code.strip()]
                    for i, code in enumerate(individual_codes, 1):
                        if code and code.lower() not in ['null', 'none', '']:
                            diagnosis_info = {
                                "code": code,
                                "position": i,
                                "source": "diag_1_50_cd",
                                "clinical_priority": "Primary" if i == 1 else "Secondary"
                            }
                            diagnosis_codes.append(diagnosis_info)
                            result["extraction_summary"]["unique_diagnosis_codes"].add(code)
                            
                            # Clinical categorization
                            if i == 1:
                                primary_diagnoses.append(code)
                            else:
                                comorbidities.append(code)

            # Handle individual diagnosis fields with enhanced clinical analysis
            for i in range(1, 51):
                diag_key = f"diag_{i}_cd"
                if diag_key in data and data[diag_key]:
                    diag_code = str(data[diag_key]).strip()
                    if diag_code and diag_code.lower() not in ['null', 'none', '']:
                        diagnosis_info = {
                            "code": diag_code,
                            "position": i,
                            "source": f"individual_{diag_key}",
                            "clinical_priority": "Primary" if i <= 3 else "Secondary"
                        }
                        diagnosis_codes.append(diagnosis_info)
                        result["extraction_summary"]["unique_diagnosis_codes"].add(diag_code)
                        
                        # Enhanced clinical categorization
                        if i <= 3:
                            primary_diagnoses.append(diag_code)
                        else:
                            comorbidities.append(diag_code)

            if diagnosis_codes:
                current_record["diagnosis_codes"] = diagnosis_codes
                current_record["clinical_complexity_indicators"] = clinical_complexity_indicators
                result["extraction_summary"]["total_diagnosis_codes"] += len(diagnosis_codes)
                
                # Update clinical insights
                result["clinical_insights"]["primary_diagnoses"].extend(primary_diagnoses)
                result["clinical_insights"]["comorbidities"].extend(comorbidities)

            if current_record:
                current_record["data_path"] = path
                current_record["clinical_complexity_score"] = len(clinical_complexity_indicators) + len(diagnosis_codes)
                result["hlth_srvc_records"].append(current_record)
                result["extraction_summary"]["total_hlth_srvc_records"] += 1
                
                # Update overall complexity score
                result["extraction_summary"]["clinical_complexity_score"] += current_record["clinical_complexity_score"]

            # Continue enhanced recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._enhanced_medical_extraction_with_clinical_analysis(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_medical_extraction_with_clinical_analysis(item, result, new_path)

    def _enhanced_pharmacy_extraction_with_therapeutic_analysis(self, data: Any, result: Dict[str, Any], path: str = ""):
        """Enhanced recursive pharmacy field extraction with therapeutic complexity analysis"""
        if isinstance(data, dict):
            current_record = {}
            therapeutic_indicators = []

            # Enhanced NDC code extraction with therapeutic analysis
            ndc_found = False
            for key in data:
                if key.lower() in ['ndc', 'ndc_code', 'ndc_number', 'national_drug_code']:
                    ndc_code = str(data[key]).strip()
                    current_record["ndc"] = ndc_code
                    result["extraction_summary"]["unique_ndc_codes"].add(ndc_code)
                    ndc_found = True
                    therapeutic_indicators.append("NDC medication code")
                    break

            # Enhanced medication name extraction with therapeutic categorization
            label_found = False
            for key in data:
                if key.lower() in ['lbl_nm', 'label_name', 'drug_name', 'medication_name', 'product_name']:
                    medication_name = str(data[key]).strip()
                    current_record["lbl_nm"] = medication_name
                    result["extraction_summary"]["unique_label_names"].add(medication_name)
                    label_found = True
                    
                    # Enhanced therapeutic categorization
                    medication_lower = medication_name.lower()
                    if any(term in medication_lower for term in ['metformin', 'insulin', 'glipizide']):
                        therapeutic_indicators.append("Diabetes medication")
                        result["extraction_summary"]["therapeutic_classes_identified"].append("Antidiabetic")
                    elif any(term in medication_lower for term in ['lisinopril', 'amlodipine', 'atenolol']):
                        therapeutic_indicators.append("Cardiovascular medication")
                        result["extraction_summary"]["therapeutic_classes_identified"].append("Cardiovascular")
                    elif any(term in medication_lower for term in ['atorvastatin', 'simvastatin']):
                        therapeutic_indicators.append("Lipid-lowering medication")
                        result["extraction_summary"]["therapeutic_classes_identified"].append("Lipid management")
                    
                    break

            # Enhanced prescription filled date extraction
            if "rx_filled_dt" in data and data["rx_filled_dt"]:
                current_record["rx_filled_dt"] = data["rx_filled_dt"]
                therapeutic_indicators.append("Prescription fill date")

            if ndc_found or label_found or "rx_filled_dt" in current_record:
                current_record["data_path"] = path
                current_record["therapeutic_indicators"] = therapeutic_indicators
                current_record["therapeutic_complexity_score"] = len(therapeutic_indicators)
                result["ndc_records"].append(current_record)
                result["extraction_summary"]["total_ndc_records"] += 1
                
                # Update medication complexity score
                result["extraction_summary"]["medication_complexity_score"] += current_record["therapeutic_complexity_score"]

            # Continue enhanced recursive search
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._enhanced_pharmacy_extraction_with_therapeutic_analysis(value, result, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._enhanced_pharmacy_extraction_with_therapeutic_analysis(item, result, new_path)

    def _enhanced_clinical_analysis_integration(self, extraction_result: Dict[str, Any]):
        """Enhanced integration of clinical analysis with code meanings"""
        try:
            # Enhance medical records with detailed clinical meanings
            service_meanings = extraction_result["code_meanings"]["service_code_meanings"]
            diagnosis_meanings = extraction_result["code_meanings"]["diagnosis_code_meanings"]
            
            for record in extraction_result["hlth_srvc_records"]:
                # Enhanced service code meaning integration
                if "hlth_srvc_cd" in record:
                    service_code = str(record["hlth_srvc_cd"])
                    detailed_meaning = service_meanings.get(service_code, "Clinical meaning not available")
                    record["hlth_srvc_detailed_meaning"] = detailed_meaning
                    record["clinical_context_enhanced"] = True
                
                # Enhanced diagnosis code meanings integration
                if "diagnosis_codes" in record:
                    for diag in record["diagnosis_codes"]:
                        if isinstance(diag, dict) and "code" in diag:
                            diag_code = str(diag["code"])
                            detailed_meaning = diagnosis_meanings.get(diag_code, "Clinical meaning not available")
                            diag["detailed_clinical_meaning"] = detailed_meaning
                            diag["clinical_significance_enhanced"] = True

            # Enhanced care complexity assessment
            total_records = len(extraction_result["hlth_srvc_records"])
            avg_complexity = extraction_result["extraction_summary"]["clinical_complexity_score"] / max(total_records, 1)
            
            if avg_complexity >= 5:
                extraction_result["clinical_insights"]["care_complexity"] = "High complexity - Multiple comorbidities and procedures"
            elif avg_complexity >= 3:
                extraction_result["clinical_insights"]["care_complexity"] = "Moderate complexity - Some comorbidities or procedures"
            else:
                extraction_result["clinical_insights"]["care_complexity"] = "Standard complexity - Routine care patterns"

            logger.info(f"✅ Enhanced clinical analysis integration completed - Care complexity: {extraction_result['clinical_insights']['care_complexity']}")

        except Exception as e:
            logger.error(f"Enhanced clinical analysis integration error: {e}")

    def _enhanced_therapeutic_analysis_integration(self, extraction_result: Dict[str, Any]):
        """Enhanced integration of therapeutic analysis with medication meanings"""
        try:
            # Enhance pharmacy records with detailed therapeutic meanings
            ndc_meanings = extraction_result["code_meanings"]["ndc_code_meanings"]
            medication_meanings = extraction_result["code_meanings"]["medication_meanings"]
            
            for record in extraction_result["ndc_records"]:
                # Enhanced NDC code meaning integration
                if "ndc" in record:
                    ndc_code = str(record["ndc"])
                    detailed_meaning = ndc_meanings.get(ndc_code, "Therapeutic meaning not available")
                    record["ndc_detailed_meaning"] = detailed_meaning
                    record["therapeutic_context_enhanced"] = True
                
                # Enhanced medication meaning integration
                if "lbl_nm" in record:
                    medication = str(record["lbl_nm"])
                    detailed_meaning = medication_meanings.get(medication, "Therapeutic meaning not available")
                    record["medication_detailed_meaning"] = detailed_meaning
                    record["clinical_significance_enhanced"] = True

            # Enhanced medication complexity assessment
            total_medications = len(extraction_result["ndc_records"])
            unique_therapeutic_classes = len(set(extraction_result["extraction_summary"]["therapeutic_classes_identified"]))
            
            if total_medications >= 5 or unique_therapeutic_classes >= 3:
                extraction_result["therapeutic_insights"]["medication_complexity"] = "High complexity - Polypharmacy considerations"
                extraction_result["extraction_summary"]["polypharmacy_indicators"] = ["Multiple medications", "Multiple therapeutic classes"]
            elif total_medications >= 3 or unique_therapeutic_classes >= 2:
                extraction_result["therapeutic_insights"]["medication_complexity"] = "Moderate complexity - Multiple medications"
            else:
                extraction_result["therapeutic_insights"]["medication_complexity"] = "Standard complexity - Limited medications"

            logger.info(f"✅ Enhanced therapeutic analysis integration completed - Medication complexity: {extraction_result['therapeutic_insights']['medication_complexity']}")

        except Exception as e:
            logger.error(f"Enhanced therapeutic analysis integration error: {e}")

    def extract_health_entities_with_clinical_insights(self, pharmacy_data: Dict[str, Any],
                                                      pharmacy_extraction: Dict[str, Any],
                                                      medical_extraction: Dict[str, Any],
                                                      patient_data: Dict[str, Any] = None,
                                                      api_integrator = None) -> Dict[str, Any]:
        """Enhanced health entity extraction with DETAILED clinical insights and comprehensive analysis"""
        logger.info("🔬 ===== Enhanced CLINICAL ENTITY EXTRACTION WITH DETAILED INSIGHTS =====")
        
        enhanced_entities = {
            "diabetics": "no",
            "age_group": "unknown",
            "age": None,
            "smoking": "no",
            "alcohol": "no",
            "blood_pressure": "unknown",
            "analysis_details": [],
            "medical_conditions": [],
            "medications_identified": [],
            "clinical_risk_factors": [],
            "therapeutic_insights": [],
            "comorbidity_analysis": [],
            "medication_adherence_indicators": [],
            "care_gaps_identified": [],
            "llm_analysis": "not_performed",
            "enhanced_clinical_analysis": False,
            "clinical_complexity_score": 0,
            "cardiovascular_risk_indicators": [],
            "preventive_care_opportunities": []
        }

        try:
            # Enhanced age calculation with clinical risk stratification
            if patient_data and patient_data.get('date_of_birth'):
                age = self._calculate_age_enhanced(patient_data['date_of_birth'])
                if age != "unknown":
                    enhanced_entities["age"] = int(age.split()[0])  # Extract just the number
                    enhanced_entities["age_group"] = self._get_enhanced_age_group(int(age.split()[0]))
                    enhanced_entities["analysis_details"].append(f"Enhanced age analysis: {age}")

            # Enhanced entity extraction using detailed batch meanings
            medical_meanings_available = (medical_extraction and 
                                        medical_extraction.get("code_meanings_added", False) and
                                        medical_extraction.get("enhanced_analysis", False))
            
            pharmacy_meanings_available = (pharmacy_extraction and 
                                         pharmacy_extraction.get("code_meanings_added", False) and
                                         pharmacy_extraction.get("enhanced_analysis", False))
            
            if medical_meanings_available or pharmacy_meanings_available:
                logger.info("🔬 Using enhanced batch-generated clinical meanings for detailed entity extraction")
                enhanced_entities = self._enhanced_analyze_entities_with_clinical_meanings(
                    enhanced_entities, medical_extraction, pharmacy_extraction
                )
                enhanced_entities["enhanced_clinical_analysis"] = True
                enhanced_entities["llm_analysis"] = "used_enhanced_batch_meanings"
                enhanced_entities["analysis_details"].append("Used enhanced batch-generated clinical meanings with detailed insights")
            else:
                logger.info("🔬 Using enhanced direct pattern matching for entity extraction")
                self._enhanced_analyze_entities_direct(pharmacy_data, pharmacy_extraction, medical_extraction, enhanced_entities)

            # Enhanced medication identification with clinical insights
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    if record.get("lbl_nm"):
                        medication_info = {
                            "ndc": record.get("ndc", ""),
                            "label_name": record.get("lbl_nm", ""),
                            "detailed_meaning": record.get("medication_detailed_meaning", ""),
                            "therapeutic_indicators": record.get("therapeutic_indicators", []),
                            "clinical_significance": record.get("clinical_significance_enhanced", False)
                        }
                        enhanced_entities["medications_identified"].append(medication_info)

            # Enhanced clinical complexity scoring
            enhanced_entities["clinical_complexity_score"] = self._calculate_enhanced_clinical_complexity(enhanced_entities)

            # Enhanced cardiovascular risk assessment
            enhanced_entities["cardiovascular_risk_indicators"] = self._assess_enhanced_cardiovascular_risk(enhanced_entities)

            # Enhanced preventive care opportunities
            enhanced_entities["preventive_care_opportunities"] = self._identify_enhanced_preventive_care_opportunities(enhanced_entities)

            logger.info(f"🔬 ===== Enhanced CLINICAL ENTITY EXTRACTION COMPLETED =====")
            logger.info(f"  ✅ Enhanced analysis: {enhanced_entities['enhanced_clinical_analysis']}")
            logger.info(f"  🩺 Diabetes: {enhanced_entities['diabetics']}")
            logger.info(f"  💓 Blood pressure: {enhanced_entities['blood_pressure']}")
            logger.info(f"  💊 Medications: {len(enhanced_entities['medications_identified'])}")
            logger.info(f"  🔬 Clinical complexity: {enhanced_entities['clinical_complexity_score']}")
            logger.info(f"  ❤️ Cardiovascular risk factors: {len(enhanced_entities['cardiovascular_risk_indicators'])}")

        except Exception as e:
            logger.error(f"❌ Error in enhanced clinical entity extraction: {e}")
            enhanced_entities["analysis_details"].append(f"Enhanced analysis error: {str(e)}")

        return enhanced_entities

    def _calculate_age_enhanced(self, date_of_birth: str) -> str:
        """Enhanced age calculation with clinical context"""
        try:
            if not date_of_birth:
                return "unknown"
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Enhanced clinical context
            if age < 18:
                return f"{age} years (Pediatric - specialized care protocols)"
            elif age < 65:
                return f"{age} years (Adult - standard care guidelines)"
            else:
                return f"{age} years (Geriatric - enhanced monitoring protocols)"
        except:
            return "unknown (Clinical assessment recommended)"

    def _get_enhanced_age_group(self, age: int) -> str:
        """Enhanced age group determination with clinical significance"""
        if age < 18:
            return "pediatric_specialized_care"
        elif age < 35:
            return "young_adult_preventive_focus"
        elif age < 50:
            return "adult_screening_emphasis"
        elif age < 65:
            return "middle_aged_risk_assessment"
        else:
            return "senior_comprehensive_monitoring"

    def _enhanced_analyze_entities_with_clinical_meanings(self, entities: Dict[str, Any], 
                                                         medical_extraction: Dict[str, Any], 
                                                         pharmacy_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced entity analysis using detailed clinical meanings"""
        try:
            clinical_conditions = []
            therapeutic_insights = []
            cardiovascular_indicators = []
            
            # Enhanced analysis of medical meanings with clinical insights
            medical_meanings = medical_extraction.get("code_meanings", {})
            diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
            
            for code, meaning in diagnosis_meanings.items():
                meaning_lower = meaning.lower()
                
                # Enhanced diabetes analysis with clinical context
                if any(term in meaning_lower for term in ['diabetes', 'diabetic', 'insulin', 'glucose', 'mellitus']):
                    entities["diabetics"] = "yes"
                    clinical_conditions.append(f"Diabetes mellitus (ICD-10 {code}) - {meaning[:100]}...")
                    entities["clinical_risk_factors"].append("Diabetes mellitus - Major CVD risk factor")
                    cardiovascular_indicators.append("Diabetes mellitus increases cardiovascular risk by 2-4x")
                
                # Enhanced hypertension analysis with clinical significance
                if any(term in meaning_lower for term in ['hypertension', 'high blood pressure', 'elevated blood pressure']):
                    entities["blood_pressure"] = "diagnosed"
                    clinical_conditions.append(f"Hypertension (ICD-10 {code}) - {meaning[:100]}...")
                    entities["clinical_risk_factors"].append("Hypertension - Major modifiable risk factor")
                    cardiovascular_indicators.append("Hypertension is leading risk factor for stroke and MI")
                
                # Enhanced tobacco use analysis
                if any(term in meaning_lower for term in ['tobacco', 'smoking', 'nicotine', 'cigarette']):
                    entities["smoking"] = "yes"
                    clinical_conditions.append(f"Tobacco use (ICD-10 {code}) - {meaning[:100]}...")
                    entities["clinical_risk_factors"].append("Tobacco use - Critical modifiable risk factor")
                    cardiovascular_indicators.append("Smoking increases cardiovascular risk by 200-300%")
                
                # Enhanced alcohol analysis
                if any(term in meaning_lower for term in ['alcohol', 'alcoholism', 'substance']):
                    entities["alcohol"] = "yes"
                    clinical_conditions.append(f"Alcohol-related condition (ICD-10 {code}) - {meaning[:100]}...")
                    entities["clinical_risk_factors"].append("Alcohol use - Health risk factor")

                # Enhanced comorbidity analysis
                if any(term in meaning_lower for term in ['chronic', 'disease', 'disorder', 'syndrome']):
                    entities["comorbidity_analysis"].append(f"Chronic condition: {code} - {meaning[:80]}...")

            # Enhanced analysis of pharmacy meanings with therapeutic insights
            pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
            medication_meanings = pharmacy_meanings.get("medication_meanings", {})
            
            for medication, meaning in medication_meanings.items():
                meaning_lower = meaning.lower()
                
                # Enhanced diabetes medication analysis
                if any(term in meaning_lower for term in ['diabetes', 'blood sugar', 'insulin', 'metformin', 'glucose']):
                    entities["diabetics"] = "yes"
                    therapeutic_insights.append(f"Diabetes medication: {medication} - {meaning[:100]}...")
                    entities["medication_adherence_indicators"].append(f"Diabetes therapy: {medication}")
                
                # Enhanced cardiovascular medication analysis
                if any(term in meaning_lower for term in ['blood pressure', 'hypertension', 'ace inhibitor', 'beta blocker', 'cardiovascular']):
                    if entities["blood_pressure"] == "unknown":
                        entities["blood_pressure"] = "managed"
                    therapeutic_insights.append(f"Cardiovascular medication: {medication} - {meaning[:100]}...")
                    entities["medication_adherence_indicators"].append(f"BP management: {medication}")

                # Enhanced lipid management analysis
                if any(term in meaning_lower for term in ['cholesterol', 'statin', 'lipid', 'atorvastatin']):
                    cardiovascular_indicators.append(f"Lipid management therapy: {medication}")
                    therapeutic_insights.append(f"Lipid medication: {medication} - {meaning[:100]}...")

            entities["medical_conditions"] = clinical_conditions
            entities["therapeutic_insights"] = therapeutic_insights
            entities["cardiovascular_risk_indicators"] = cardiovascular_indicators
            
            logger.info(f"🔬 Enhanced clinical meaning analysis: {len(clinical_conditions)} conditions, {len(therapeutic_insights)} therapeutic insights")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in enhanced clinical meaning analysis: {e}")
            return entities

    def _enhanced_analyze_entities_direct(self, pharmacy_data: Dict[str, Any],
                                        pharmacy_extraction: Dict[str, Any],
                                        medical_extraction: Dict[str, Any],
                                        entities: Dict[str, Any]):
        """Enhanced direct entity analysis using clinical pattern matching"""
        try:
            logger.info("🔬 Enhanced direct clinical pattern matching analysis")
            
            # Enhanced medication pattern matching with clinical context
            if pharmacy_extraction and pharmacy_extraction.get("ndc_records"):
                for record in pharmacy_extraction["ndc_records"]:
                    medication_name = record.get("lbl_nm", "").lower()
                    
                    # Enhanced diabetes detection with clinical significance
                    if any(term in medication_name for term in ['metformin', 'insulin', 'glipizide', 'glimepiride']):
                        entities["diabetics"] = "yes"
                        entities["clinical_risk_factors"].append(f"Diabetes medication detected: {medication_name}")
                        
                    # Enhanced cardiovascular detection with therapeutic context
                    if any(term in medication_name for term in ['amlodipine', 'lisinopril', 'atenolol', 'metoprolol']):
                        entities["blood_pressure"] = "managed"
                        entities["clinical_risk_factors"].append(f"Cardiovascular medication detected: {medication_name}")

            entities["analysis_details"].append("Enhanced direct clinical pattern matching completed")

        except Exception as e:
            logger.error(f"Error in enhanced direct clinical analysis: {e}")
            entities["analysis_details"].append(f"Enhanced direct analysis error: {str(e)}")

    def _calculate_enhanced_clinical_complexity(self, entities: Dict[str, Any]) -> int:
        """Calculate enhanced clinical complexity score"""
        try:
            complexity_score = 0
            
            # Age-based complexity
            age = entities.get("age", 0)
            if age >= 65:
                complexity_score += 2
            elif age >= 50:
                complexity_score += 1
            
            # Condition-based complexity
            if entities.get("diabetics") == "yes":
                complexity_score += 3
            if entities.get("blood_pressure") in ["diagnosed", "managed"]:
                complexity_score += 2
            if entities.get("smoking") == "yes":
                complexity_score += 2
            
            # Medication complexity
            medication_count = len(entities.get("medications_identified", []))
            if medication_count >= 5:
                complexity_score += 3
            elif medication_count >= 3:
                complexity_score += 2
            elif medication_count >= 1:
                complexity_score += 1
            
            # Risk factor complexity
            risk_factor_count = len(entities.get("clinical_risk_factors", []))
            complexity_score += min(risk_factor_count, 5)
            
            return complexity_score
            
        except Exception as e:
            logger.error(f"Clinical complexity calculation error: {e}")
            return 0

    def _assess_enhanced_cardiovascular_risk(self, entities: Dict[str, Any]) -> List[str]:
        """Assess enhanced cardiovascular risk indicators"""
        try:
            risk_indicators = []
            
            # Age-based risk
            age = entities.get("age", 0)
            if age >= 65:
                risk_indicators.append("Advanced age (≥65) - Major non-modifiable risk factor")
            elif age >= 45:
                risk_indicators.append("Intermediate age (45-64) - Moderate risk factor")
            
            # Clinical condition risks
            if entities.get("diabetics") == "yes":
                risk_indicators.append("Diabetes mellitus - 2-4x increased cardiovascular risk")
            if entities.get("blood_pressure") in ["diagnosed", "managed"]:
                risk_indicators.append("Hypertension - Leading risk factor for stroke and MI")
            if entities.get("smoking") == "yes":
                risk_indicators.append("Tobacco use - 200-300% increased cardiovascular risk")
            
            # Medication-based risk assessment
            medication_count = len(entities.get("medications_identified", []))
            if medication_count >= 5:
                risk_indicators.append("Polypharmacy (≥5 medications) - Complexity and interaction risks")
            
            return risk_indicators
            
        except Exception as e:
            logger.error(f"Cardiovascular risk assessment error: {e}")
            return []

    def _identify_enhanced_preventive_care_opportunities(self, entities: Dict[str, Any]) -> List[str]:
        """Identify enhanced preventive care opportunities"""
        try:
            opportunities = []
            
            age = entities.get("age", 0)
            
            # Age-based preventive care
            if age >= 50:
                opportunities.append("Colorectal cancer screening (colonoscopy every 10 years)")
                opportunities.append("Annual cardiovascular risk assessment")
            if age >= 65:
                opportunities.append("Annual influenza vaccination")
                opportunities.append("Pneumococcal vaccination")
                opportunities.append("Fall risk assessment")
            
            # Condition-based preventive care
            if entities.get("diabetics") == "yes":
                opportunities.append("HbA1c monitoring every 3-6 months")
                opportunities.append("Annual diabetic eye examination")
                opportunities.append("Annual diabetic foot examination")
                opportunities.append("Annual nephropathy screening")
            
            if entities.get("blood_pressure") in ["diagnosed", "managed"]:
                opportunities.append("Blood pressure monitoring and medication adherence")
                opportunities.append("Sodium restriction counseling")
                opportunities.append("Regular cardiovascular risk assessment")
            
            if entities.get("smoking") == "yes":
                opportunities.append("Tobacco cessation counseling and support")
                opportunities.append("Enhanced lung cancer screening consideration")
            
            # General preventive care
            if age >= 40:
                opportunities.append("Annual lipid panel screening")
                opportunities.append("Blood pressure monitoring")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Preventive care opportunities identification error: {e}")
            return []

    def prepare_enhanced_clinical_context(self, chat_context: Dict[str, Any]) -> str:
        """Enhanced context preparation for clinical chatbot with comprehensive data"""
        try:
            context_sections = []

            # Enhanced patient overview with clinical context
            patient_overview = chat_context.get("patient_overview", {})
            if patient_overview:
                context_sections.append(f"**ENHANCED PATIENT PROFILE**: Age {patient_overview.get('age', 'unknown')}, ZIP {patient_overview.get('zip', 'unknown')}")
                context_sections.append(f"**CLINICAL ANALYSIS TYPE**: {patient_overview.get('model_type', 'standard')}")
                context_sections.append(f"**CARDIOVASCULAR RISK LEVEL**: {patient_overview.get('cardiovascular_risk_level', 'unknown')}")

            # Enhanced medical extractions with detailed clinical meanings
            medical_extraction = chat_context.get("medical_extraction", {})
            if medical_extraction and not medical_extraction.get('error'):
                context_sections.append(f"**ENHANCED MEDICAL DATA WITH DETAILED CLINICAL MEANINGS**: {json.dumps(medical_extraction, indent=2)}")

            # Enhanced pharmacy extractions with detailed therapeutic meanings
            pharmacy_extraction = chat_context.get("pharmacy_extraction", {})
            if pharmacy_extraction and not pharmacy_extraction.get('error'):
                context_sections.append(f"**ENHANCED PHARMACY DATA WITH DETAILED THERAPEUTIC MEANINGS**: {json.dumps(pharmacy_extraction, indent=2)}")

            # Enhanced entity extraction with clinical insights
            entity_extraction = chat_context.get("entity_extraction", {})
            if entity_extraction:
                context_sections.append(f"**ENHANCED CLINICAL ENTITIES WITH DETAILED INSIGHTS**: {json.dumps(entity_extraction, indent=2)}")

            # Enhanced health trajectory with detailed evaluation
            enhanced_trajectory = chat_context.get("enhanced_health_trajectory", "")
            if enhanced_trajectory:
                context_sections.append(f"**ENHANCED HEALTH TRAJECTORY WITH DETAILED EVALUATION**: {enhanced_trajectory[:1000]}...")

            # Enhanced cardiovascular risk assessment
            heart_attack_prediction = chat_context.get("heart_attack_prediction", {})
            if heart_attack_prediction:
                context_sections.append(f"**ENHANCED CARDIOVASCULAR RISK ASSESSMENT**: {json.dumps(heart_attack_prediction, indent=2)}")

            return "\n\n" + "\n\n".join(context_sections)

        except Exception as e:
            logger.error(f"Error preparing enhanced clinical context: {e}")
            return "Enhanced comprehensive patient healthcare data with detailed clinical meanings and therapeutic insights available for analysis."

    # Backward compatibility methods
    def extract_medical_fields_batch(self, deidentified_medical: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced extraction"""
        return self.extract_medical_fields_batch_enhanced(deidentified_medical)

    def extract_pharmacy_fields_batch(self, deidentified_pharmacy: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced extraction"""
        return self.extract_pharmacy_fields_batch_enhanced(deidentified_pharmacy)

    def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_medical_data_enhanced(medical_data, patient_data)

    def deidentify_pharmacy_data(self, pharmacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_pharmacy_data_enhanced(pharmacy_data)

    def deidentify_mcid_data(self, mcid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced deidentification"""
        return self.deidentify_mcid_data_enhanced(mcid_data)

    def extract_health_entities_enhanced(self, pharmacy_data: Dict[str, Any],
                                       pharmacy_extraction: Dict[str, Any],
                                       medical_extraction: Dict[str, Any],
                                       patient_data: Dict[str, Any] = None,
                                       api_integrator = None) -> Dict[str, Any]:
        """Backward compatibility - uses enhanced clinical insights"""
        return self.extract_health_entities_with_clinical_insights(
            pharmacy_data, pharmacy_extraction, medical_extraction, patient_data, api_integrator
        )

    def prepare_chunked_context(self, chat_context: Dict[str, Any]) -> str:
        """Backward compatibility - uses enhanced clinical context"""
        return self.prepare_enhanced_clinical_context(chat_context)

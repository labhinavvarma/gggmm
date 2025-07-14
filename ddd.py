def extract_entities(self, state: HealthAnalysisState) -> HealthAnalysisState:
        """Enhanced LangGraph Node 4: Extract comprehensive health entities using LLM"""
        logger.info("🎯 Enhanced Node 4: Starting LLM-powered health entity extraction...")
        state["current_step"] = "extract_entities"
        state["step_status"]["extract_entities"] = "running"
        
        try:
            pharmacy_data = state.get("pharmacy_output", {})
            pharmacy_extraction = state.get("pharmacy_extraction", {})
            medical_extraction = state.get("medical_extraction", {})
            patient_data = state.get("patient_data", {})
            
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
            
            # Enhanced entity extraction WITH LLM
            entities = self.data_processor.extract_health_entities_enhanced(
                pharmacy_data, 
                pharmacy_extraction, 
                medical_extraction,
                patient_data,  # Pass patient data for age calculation
                self.api_integrator  # Pass API integrator for LLM calls
            )
            
            state["entity_extraction"] = entities
            state["step_status"]["extract_entities"] = "completed"
            
            # Enhanced logging
            conditions_count = len(entities.get("medical_conditions", []))
            medications_count = len(entities.get("medications_identified", []))
            llm_status = entities.get("llm_analysis", "not_performed")
            age_info = f"Age: {entities.get('age', 'unknown')} ({entities.get('age_group', 'unknown')})"
            
            logger.info(f"✅ Successfully extracted health entities using LLM: {conditions_count} conditions, {medications_count} medications")
            logger.info(f"📊 Entity results: Diabetes={entities.get('diabetics')}, Smoking={entities.get('smoking')}, BP={entities.get('blood_pressure')}")
            logger.info(f"📅 {age_info}")
            logger.info(f"🤖 LLM analysis: {llm_status}")
            
        except Exception as e:
            error_msg = f"Error in LLM-powered entity extraction: {str(e)}"
            state["errors"].append(error_msg)
            state["step_status"]["extract_entities"] = "error"
            logger.error(error_msg)
        
        return state

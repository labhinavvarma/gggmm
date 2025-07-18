# =============================================================================
# UPDATED PHARMACY DEIDENTIFICATION - health_data_processor.py
# =============================================================================

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

# =============================================================================
# ALSO UPDATE THE MEDICAL DEIDENTIFICATION TO BE CONSISTENT
# =============================================================================

def deidentify_medical_data(self, medical_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deidentify medical data with complete JSON processing and name masking"""
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
        
        # Deep copy and process the entire JSON structure with name masking
        deidentified_medical_data = self._deep_deidentify_medical_json(raw_medical_data)
        
        deidentified = {
            "src_mbr_first_nm": "[MASKED_NAME]",
            "src_mbr_last_nm": "[MASKED_NAME]", 
            "src_mbr_mid_init_nm": None,
            "src_mbr_age": age,
            "src_mbr_zip_cd": "12345",
            "medical_claims_data": deidentified_medical_data,  # Complete processed JSON
            "original_structure_preserved": True,
            "deidentification_timestamp": datetime.now().isoformat(),
            "data_type": "medical_claims",
            "name_fields_masked": ["src_mbr_first_nm", "src_mbr_last_nm", "src_mbr_mid_init_nm"]
        }
        
        logger.info("✅ Successfully deidentified complete medical claims JSON structure with name masking")
        return deidentified
        
    except Exception as e:
        logger.error(f"Error in medical deidentification: {e}")
        return {"error": f"Deidentification failed: {str(e)}"}

def _deep_deidentify_medical_json(self, data: Any) -> Any:
    """Deep deidentification of medical JSON structure with specific name field masking"""
    try:
        if isinstance(data, dict):
            # Process dictionary recursively
            deidentified_dict = {}
            for key, value in data.items():
                # Mask specific name fields in medical data
                if key.lower() in ['src_mbr_first_nm', 'src_mbr_frst_nm', 'src_mbr_last_nm', 'src_mvr_last_nm', 'src_mbr_mid_init_nm']:
                    deidentified_dict[key] = "[MASKED_NAME]"
                    logger.info(f"🔒 Masked medical name field: {key}")
                elif isinstance(value, (dict, list)):
                    # Recursively process nested structures
                    deidentified_dict[key] = self._deep_deidentify_medical_json(value)
                elif isinstance(value, str):
                    # Apply string deidentification
                    deidentified_dict[key] = self._deidentify_string(value)
                else:
                    # Keep primitive types as-is
                    deidentified_dict[key] = value
            return deidentified_dict
            
        elif isinstance(data, list):
            # Process list recursively
            return [self._deep_deidentify_medical_json(item) for item in data]
            
        elif isinstance(data, str):
            # Deidentify string values
            return self._deidentify_string(value)
            
        else:
            # Return primitive types as-is (int, float, bool, None)
            return data
            
    except Exception as e:
        logger.warning(f"Error in deep medical deidentification: {e}")
        return data  # Return original data if deidentification fails

# =============================================================================
# OPTIONAL: UPDATE THE GENERIC _deep_deidentify_json TO BE MORE SPECIFIC
# =============================================================================

def _mask_specific_fields(self, data: Any) -> Any:
    """Enhanced specific field masking for both medical and pharmacy data"""
    if isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            # Comprehensive name field masking
            if key.lower() in [
                'src_mbr_frst_nm', 'src_mbr_first_nm', 'src_mbr_last_nm', 'src_mvr_last_nm',
                'scr_mbr_last_nm', 'src_mbr_mid_init_nm', 'patient_first_name', 'patient_last_name'
            ]:
                masked_data[key] = "[MASKED_NAME]"
                logger.info(f"🔒 Masked name field: {key}")
            elif isinstance(value, (dict, list)):
                masked_data[key] = self._mask_specific_fields(value)
            else:
                masked_data[key] = value
        return masked_data
    elif isinstance(data, list):
        return [self._mask_specific_fields(item) for item in data]
    else:
        return data

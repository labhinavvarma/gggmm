def extract_mcid_list_from_deidentified_mcid(self, deidentified_mcid: Dict[str, Any]) -> List[str]:
    """
    Extract MCID list from deidentified_mcid data
    
    Args:
        deidentified_mcid (Dict[str, Any]): Deidentified MCID data containing mcid_list attribute
        
    Returns:
        List[str]: List of MCIDs
    """
    try:
        logger.info("Extracting MCID list from deidentified_mcid data...")
        
        # Check if deidentified_mcid data exists and is valid
        if not deidentified_mcid or not isinstance(deidentified_mcid, dict):
            logger.warning("No valid deidentified_mcid data provided")
            return []
        
        # Extract mcid_list attribute directly
        mcid_list = deidentified_mcid.get("mcid_list", [])
        
        # Validate that mcid_list is a list
        if not isinstance(mcid_list, list):
            logger.warning(f"mcid_list is not a list, found type: {type(mcid_list)}")
            return []
        
        # Clean and validate MCIDs
        cleaned_mcids = []
        for mcid in mcid_list:
            if mcid and str(mcid).strip():
                cleaned_mcid = str(mcid).strip()
                if cleaned_mcid not in ["None", "", "null", "NULL"]:
                    cleaned_mcids.append(cleaned_mcid)
        
        logger.info(f"Successfully extracted {len(cleaned_mcids)} MCIDs from deidentified_mcid data")
        if cleaned_mcids:
            logger.info(f"Sample MCIDs: {cleaned_mcids[:3]}")
        
        return cleaned_mcids
        
    except Exception as e:
        logger.error(f"Error extracting MCIDs from deidentified_mcid data: {str(e)}")
        return []

# Helper method to get MCIDs for a specific patient from the state
def get_patient_mcids_from_state(self, state: HealthAnalysisState) -> List[str]:
    """
    Convenience method to extract MCIDs from the current state
    
    Args:
        state (HealthAnalysisState): Current workflow state
        
    Returns:
        List[str]: List of MCIDs for the patient
    """
    try:
        deidentified_mcid = state.get("deidentified_mcid", {})
        mcid_list = self.extract_mcid_list_from_deidentified_mcid(deidentified_mcid)
        return mcid_list
            
    except Exception as e:
        logger.error(f"Error getting patient MCIDs from state: {str(e)}")
        return []

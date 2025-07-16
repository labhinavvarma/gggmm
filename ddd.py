def _prepare_enhanced_fastapi_features(self, features: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Prepare enhanced feature data for FastAPI server call"""
    try:
        # Extract the features from the feature extraction result
        extracted_features = features.get("extracted_features", {})
        
        # Convert to FastAPI format with proper parameter names
        fastapi_features = {
            "age": int(extracted_features.get("Age", 50)),
            "gender": int(extracted_features.get("Gender", 0)),
            "diabetes": int(extracted_features.get("Diabetes", 0)),
            "high_bp": int(extracted_features.get("High_BP", 0)),
            "smoking": int(extracted_features.get("Smoking", 0))
        }
        
        # Validate age range
        if fastapi_features["age"] < 0 or fastapi_features["age"] > 120:
            logger.warning(f"Age {fastapi_features['age']} out of range, using default 50")
            fastapi_features["age"] = 50
        
        # Validate binary features (0 or 1 only)
        binary_features = ["gender", "diabetes", "high_bp", "smoking"]
        for key in binary_features:
            if fastapi_features[key] not in [0, 1]:
                logger.warning(f"{key} value {fastapi_features[key]} invalid, using 0")
                fastapi_features[key] = 0
        
        logger.info(f"✅ FastAPI features prepared: {fastapi_features}")
        return fastapi_features
        
    except Exception as e:
        logger.error(f"Error preparing FastAPI features: {e}")
        return None

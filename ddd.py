diabetes_keywords = [
        # === INSULIN TYPES ===
        # Rapid-acting insulins
        'insulin', 'humalog', 'novolog', 'apidra', 'fiasp', 'lyumjev',
        'insulin lispro', 'insulin aspart', 'insulin glulisine',
        
        # Long-acting insulins  
        'lantus', 'levemir', 'tresiba', 'toujeo', 'basaglar',
        'insulin glargine', 'insulin detemir', 'insulin degludec',
        
        # Intermediate-acting insulins
        'nph insulin', 'humulin n', 'novolin n',
        
        # Mixed insulins
        'humulin 70/30', 'novolin 70/30', 'humalog mix', 'novolog mix',
        
        # === ORAL DIABETES MEDICATIONS ===
        # Metformin (Biguanides)
        'metformin', 'glucophage', 'glucophage xr', 'fortamet', 'glumetza', 'riomet',
        
        # Sulfonylureas
        'glipizide', 'glyburide', 'glimepiride', 'glucotrol', 'diabeta', 'micronase',
        'glynase', 'amaryl', 'glipizide xl', 'glipizide er',
        
        # DPP-4 Inhibitors
        'sitagliptin', 'saxagliptin', 'linagliptin', 'alogliptin',
        'januvia', 'onglyza', 'tradjenta', 'nesina',
        
        # SGLT2 Inhibitors
        'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin',
        'invokana', 'farxiga', 'jardiance', 'steglatro',
        
        # GLP-1 Receptor Agonists
        'semaglutide', 'liraglutide', 'dulaglutide', 'exenatide', 'lixisenatide',
        'ozempic', 'wegovy', 'victoza', 'saxenda', 'trulicity', 'byetta', 'bydureon', 'adlyxin',
        
        # Thiazolidinediones (TZDs)
        'pioglitazone', 'rosiglitazone', 'actos', 'avandia',
        
        # Alpha-glucosidase inhibitors
        'acarbose', 'miglitol', 'precose', 'glyset',
        
        # Meglitinides
        'repaglinide', 'nateglinide', 'prandin', 'starlix',
        
        # === COMBINATION MEDICATIONS ===
        'janumet', 'kombiglyze', 'jentadueto', 'kazano', 'oseni', 'invokamet',
        'xigduo', 'synjardy', 'steglujan', 'qtern', 'trijardy',
        
        # === DIABETES-RELATED TERMS ===
        'diabetes', 'diabetic', 'diabetes mellitus', 'dm', 'type 1 diabetes', 'type 2 diabetes',
        'diabetes type 1', 'diabetes type 2', 't1dm', 't2dm', 'iddm', 'niddm',
        'gestational diabetes', 'diabetic ketoacidosis', 'dka', 'hyperglycemia', 'hypoglycemia',
        
        # === BLOOD GLUCOSE MONITORING ===
        'glucose', 'blood glucose', 'blood sugar', 'glucometer', 'glucose meter',
        'glucose strips', 'test strips', 'lancets', 'glucose monitor',
        'continuous glucose monitor', 'cgm', 'dexcom', 'freestyle', 'omnipod',
        
        # === DIABETIC COMPLICATIONS & MANAGEMENT ===
        'diabetic nephropathy', 'diabetic retinopathy', 'diabetic neuropathy',
        'diabetic foot', 'hba1c', 'hemoglobin a1c', 'a1c', 'fructosamine',
        
        # === ADDITIONAL BRAND NAMES ===
        'humulin', 'novolin', 'admelog', 'insulin pump', 'omnipod', 'medtronic',
        'freestyle libre', 'glucose gel', 'glucose tablets',
        
        # === INSULIN DELIVERY DEVICES ===
        'insulin pen', 'insulin pump', 'insulin syringe', 'pen needles',
        'pump supplies', 'infusion sets', 'reservoirs',
        
        # === NEWER MEDICATIONS ===
        'tirzepatide', 'mounjaro', 'rybelsus', 'oral semaglutide',
        'sotagliflozin', 'zynquista', 'bexagliflozin', 'brenzavvy'
    ]





# =============================================================================
# EXACT CHANGES TO MAKE IN health_data_processor.py
# =============================================================================

# 1. FIND this method around line ~380:
def _analyze_pharmacy_for_entities(self, data_str: str, entities: Dict[str, Any]):
    """Original pharmacy data analysis for entities"""
    
    # REPLACE these lines:
    diabetes_keywords = [
        'insulin', 'metformin', 'glipizide', 'diabetes', 'diabetic', 
        'glucophage', 'lantus', 'humalog', 'novolog', 'levemir'
    ]
    
    # WITH this comprehensive list:
    diabetes_keywords = [
        # === INSULIN TYPES ===
        'insulin', 'humalog', 'novolog', 'apidra', 'fiasp', 'lyumjev',
        'insulin lispro', 'insulin aspart', 'insulin glulisine',
        'lantus', 'levemir', 'tresiba', 'toujeo', 'basaglar',
        'insulin glargine', 'insulin detemir', 'insulin degludec',
        'nph insulin', 'humulin n', 'novolin n', 'humulin', 'novolin',
        'humulin 70/30', 'novolin 70/30', 'humalog mix', 'novolog mix',
        
        # === ORAL DIABETES MEDICATIONS ===
        'metformin', 'glucophage', 'glucophage xr', 'fortamet', 'glumetza', 'riomet',
        'glipizide', 'glyburide', 'glimepiride', 'glucotrol', 'diabeta', 'micronase',
        'glynase', 'amaryl', 'glipizide xl', 'glipizide er',
        'sitagliptin', 'saxagliptin', 'linagliptin', 'alogliptin',
        'januvia', 'onglyza', 'tradjenta', 'nesina',
        'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin',
        'invokana', 'farxiga', 'jardiance', 'steglatro',
        'pioglitazone', 'rosiglitazone', 'actos', 'avandia',
        'acarbose', 'miglitol', 'precose', 'glyset',
        'repaglinide', 'nateglinide', 'prandin', 'starlix',
        
        # === INJECTABLE NON-INSULIN ===
        'semaglutide', 'liraglutide', 'dulaglutide', 'exenatide', 'lixisenatide',
        'ozempic', 'wegovy', 'victoza', 'saxenda', 'trulicity', 'byetta', 'bydureon', 'adlyxin',
        'tirzepatide', 'mounjaro', 'rybelsus',
        
        # === COMBINATION MEDICATIONS ===
        'janumet', 'kombiglyze', 'jentadueto', 'kazano', 'oseni', 'invokamet',
        'xigduo', 'synjardy', 'steglujan', 'qtern', 'trijardy',
        
        # === DIABETES TERMS ===
        'diabetes', 'diabetic', 'diabetes mellitus', 'type 1 diabetes', 'type 2 diabetes',
        'diabetes type 1', 'diabetes type 2', 't1dm', 't2dm', 'iddm', 'niddm',
        'gestational diabetes', 'hyperglycemia', 'hypoglycemia',
        
        # === MONITORING SUPPLIES ===
        'glucose', 'blood glucose', 'glucometer', 'glucose meter',
        'glucose strips', 'test strips', 'lancets', 'glucose monitor',
        'continuous glucose monitor', 'cgm', 'dexcom', 'freestyle', 'omnipod',
        'hba1c', 'hemoglobin a1c', 'a1c', 'glucose tablets', 'glucose gel'
    ]

# 2. FIND this method around line ~420:
def _analyze_medical_extraction_for_entities(self, medical_extraction: Dict[str, Any], entities: Dict[str, Any]):
    """Analyze medical codes for health conditions"""
    
    # REPLACE these lines:
    condition_mappings = {
        "diabetes": ["E10", "E11", "E12", "E13", "E14"],
        "hypertension": ["I10", "I11", "I12", "I13", "I15"],
        "smoking": ["Z72.0", "F17"],
        "alcohol": ["F10", "Z72.1"],
    }
    
    # WITH this comprehensive mapping:
    condition_mappings = {
        "diabetes": [
            # Type 1 Diabetes
            "E10", "E10.1", "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8", "E10.9",
            "E10.10", "E10.11", "E10.21", "E10.22", "E10.29", "E10.31", "E10.32", "E10.33", "E10.34", "E10.35", "E10.36", "E10.37", "E10.39",
            "E10.40", "E10.41", "E10.42", "E10.43", "E10.44", "E10.49", "E10.51", "E10.52", "E10.59",
            "E10.61", "E10.62", "E10.63", "E10.64", "E10.65", "E10.69",
            
            # Type 2 Diabetes (most common)
            "E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9",
            "E11.00", "E11.01", "E11.10", "E11.11", "E11.21", "E11.22", "E11.29", 
            "E11.31", "E11.32", "E11.33", "E11.34", "E11.35", "E11.36", "E11.37", "E11.39",
            "E11.40", "E11.41", "E11.42", "E11.43", "E11.44", "E11.49",
            "E11.51", "E11.52", "E11.59", "E11.61", "E11.62", "E11.63", "E11.64", "E11.65", "E11.69",
            
            # Other diabetes types
            "E13", "E09", "E08", "O24", "R73"
        ],
        "hypertension": ["I10", "I11", "I12", "I13", "I15", "I16"],
        "smoking": ["Z72.0", "F17", "Z87.891"],
        "alcohol": ["F10", "Z72.1", "Z87.891"],
    }

# 3. FIND this method around line ~350:
def _analyze_pharmacy_extraction_for_entities(self, pharmacy_extraction: Dict[str, Any], entities: Dict[str, Any]):
    """Analyze structured pharmacy extraction for health entities"""
    
    # ADD this enhanced detection after the existing medication identification:
    
    # Enhanced diabetes medication detection
    diabetes_indicators = [
        'insulin', 'metformin', 'glucophage', 'diabetes', 'diabetic',
        'lantus', 'levemir', 'humalog', 'novolog', 'glipizide', 'glyburide',
        'sitagliptin', 'januvia', 'ozempic', 'semaglutide', 'trulicity',
        'jardiance', 'farxiga', 'invokana', 'actos', 'amaryl', 'glimepiride',
        'byetta', 'victoza', 'saxenda', 'mounjaro', 'tirzepatide', 'dulaglutide',
        'empagliflozin', 'dapagliflozin', 'canagliflozin', 'pioglitazone',
        'liraglutide', 'exenatide', 'repaglinide', 'nateglinide', 'acarbose'
    ]
    
    # Replace the existing simple detection with this enhanced version:
    if any(indicator in lbl_lower for indicator in diabetes_indicators):
        entities["diabetics"] = "yes"
        entities["analysis_details"].append(f"Enhanced diabetes medication found: {lbl_nm}")

# =============================================================================
# SUMMARY OF CHANGES:
# =============================================================================
# 1. Expanded diabetes keywords from 10 to 80+ terms
# 2. Enhanced ICD-10 codes from 5 to 50+ diabetes codes  
# 3. Improved pharmacy medication detection with 30+ drug names
# 4. Added brand names, generic names, and medical device terms
# =============================================================================

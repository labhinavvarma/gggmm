#!/usr/bin/env python3
"""
PKL File Inspector - Debug Tool for Heart Attack Prediction Model
================================================================

This standalone script helps you inspect your PKL file structure to debug loading issues.

Usage:
    python pkl_inspector.py your_model.pkl

or run interactively and it will prompt for the file path.
"""

import pickle
import os
import sys
from typing import Any, Dict

def inspect_pkl_file(pkl_path: str) -> Dict[str, Any]:
    """
    Comprehensive PKL file inspector
    """
    print(f"🔍 Inspecting PKL file: {pkl_path}")
    print("=" * 60)
    
    # Check file existence and basic info
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return {"error": "File not found"}
    
    file_size = os.path.getsize(pkl_path)
    print(f"📁 File exists: ✅")
    print(f"📦 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    try:
        # Load the PKL file
        print(f"\n🔄 Loading PKL file...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ PKL file loaded successfully!")
        print(f"📋 Data type: {type(data)}")
        print(f"📋 Data class: {data.__class__.__name__}")
        
        # Detailed inspection based on type
        print(f"\n🔍 DETAILED INSPECTION:")
        print("-" * 40)
        
        if isinstance(data, dict):
            print("📊 DICTIONARY FORMAT DETECTED")
            print(f"   Keys: {list(data.keys())}")
            print(f"   Number of keys: {len(data.keys())}")
            
            print(f"\n   Key Details:")
            for key, value in data.items():
                print(f"   - '{key}': {type(value).__name__}")
                if hasattr(value, 'predict'):
                    print(f"     └─ ✅ Has predict method (likely a model)")
                if hasattr(value, 'transform'):
                    print(f"     └─ ✅ Has transform method (likely a scaler)")
                if isinstance(value, (list, tuple)) and len(value) < 20:
                    print(f"     └─ Content: {value}")
            
            # Look for model-like objects
            model_candidates = []
            scaler_candidates = []
            feature_candidates = []
            
            for key, value in data.items():
                if hasattr(value, 'predict') and hasattr(value, 'fit'):
                    model_candidates.append(key)
                if hasattr(value, 'transform') and hasattr(value, 'fit_transform'):
                    scaler_candidates.append(key)
                if isinstance(value, (list, tuple)) and len(value) < 20:
                    feature_candidates.append(key)
            
            print(f"\n   🤖 Potential MODEL keys: {model_candidates}")
            print(f"   🔧 Potential SCALER keys: {scaler_candidates}")
            print(f"   📊 Potential FEATURE keys: {feature_candidates}")
            
            # Show how to access
            if model_candidates:
                print(f"\n   💡 To access model: data['{model_candidates[0]}']")
            if scaler_candidates:
                print(f"   💡 To access scaler: data['{scaler_candidates[0]}']")
            if feature_candidates:
                print(f"   💡 To access features: data['{feature_candidates[0]}']")
        
        elif hasattr(data, '__dict__'):
            print("📊 OBJECT WITH ATTRIBUTES DETECTED")
            attributes = [attr for attr in dir(data) if not attr.startswith('_')]
            print(f"   Attributes: {attributes[:10]}{'...' if len(attributes) > 10 else ''}")
            print(f"   Number of attributes: {len(attributes)}")
            
            print(f"\n   Attribute Details:")
            for attr in attributes[:15]:  # Show first 15 attributes
                try:
                    value = getattr(data, attr)
                    if not callable(value):  # Skip methods
                        print(f"   - {attr}: {type(value).__name__}")
                        if hasattr(value, 'predict'):
                            print(f"     └─ ✅ Has predict method (likely a model)")
                        if hasattr(value, 'transform'):
                            print(f"     └─ ✅ Has transform method (likely a scaler)")
                except:
                    print(f"   - {attr}: <could not access>")
            
            # Look for model-like attributes
            model_attrs = []
            scaler_attrs = []
            
            for attr in attributes:
                try:
                    value = getattr(data, attr)
                    if hasattr(value, 'predict') and hasattr(value, 'fit'):
                        model_attrs.append(attr)
                    if hasattr(value, 'transform') and hasattr(value, 'fit_transform'):
                        scaler_attrs.append(attr)
                except:
                    pass
            
            print(f"\n   🤖 Potential MODEL attributes: {model_attrs}")
            print(f"   🔧 Potential SCALER attributes: {scaler_attrs}")
            
            # Show how to access
            if model_attrs:
                print(f"\n   💡 To access model: data.{model_attrs[0]}")
            if scaler_attrs:
                print(f"   💡 To access scaler: data.{scaler_attrs[0]}")
        
        elif isinstance(data, (list, tuple)):
            print(f"📊 {type(data).__name__.upper()} FORMAT DETECTED")
            print(f"   Length: {len(data)}")
            
            print(f"\n   Item Details:")
            for i, item in enumerate(data):
                print(f"   [{i}]: {type(item).__name__}")
                if hasattr(item, 'predict'):
                    print(f"       └─ ✅ Has predict method (likely a model)")
                if hasattr(item, 'transform'):
                    print(f"       └─ ✅ Has transform method (likely a scaler)")
                if isinstance(item, (list, tuple)) and len(item) < 20:
                    print(f"       └─ Content: {item}")
            
            print(f"\n   💡 To access items: data[0], data[1], data[2], etc.")
        
        else:
            print("📊 DIRECT OBJECT DETECTED")
            print(f"   Type: {type(data).__name__}")
            print(f"   Module: {getattr(data, '__module__', 'Unknown')}")
            
            # Check if it's a model
            has_predict = hasattr(data, 'predict')
            has_fit = hasattr(data, 'fit')
            has_predict_proba = hasattr(data, 'predict_proba')
            
            print(f"\n   Model Capabilities:")
            print(f"   - predict method: {'✅' if has_predict else '❌'}")
            print(f"   - fit method: {'✅' if has_fit else '❌'}")
            print(f"   - predict_proba method: {'✅' if has_predict_proba else '❌'}")
            
            if has_predict and has_fit:
                print(f"\n   ✅ This appears to be a machine learning model!")
                print(f"   💡 Can be used directly as: model = data")
                
                # Try to get more info
                try:
                    if hasattr(data, 'feature_names_in_'):
                        print(f"   📊 Features: {data.feature_names_in_}")
                    elif hasattr(data, 'n_features_'):
                        print(f"   📊 Number of features: {data.n_features_}")
                    
                    # Try a test prediction
                    print(f"\n   🧪 Testing model prediction...")
                    import numpy as np
                    
                    # Try with different feature counts
                    for n_features in [5, 13, 4, 1]:
                        try:
                            test_data = np.array([[0] * n_features])
                            prediction = data.predict(test_data)
                            print(f"   ✅ Test prediction successful with {n_features} features: {prediction}")
                            if has_predict_proba:
                                proba = data.predict_proba(test_data)
                                print(f"   ✅ Test probability successful: {proba}")
                            break
                        except Exception as e:
                            print(f"   ❌ Test with {n_features} features failed: {str(e)[:50]}...")
                            continue
                    
                except Exception as e:
                    print(f"   ⚠️ Could not test model: {e}")
            else:
                print(f"\n   ❌ This doesn't appear to be a standard ML model")
        
        # Summary and recommendations
        print(f"\n🎯 SUMMARY & RECOMMENDATIONS:")
        print("-" * 40)
        
        if isinstance(data, dict):
            model_keys = [k for k, v in data.items() if hasattr(v, 'predict')]
            if model_keys:
                print(f"✅ Dictionary format with model at key: '{model_keys[0]}'")
                print(f"💡 Use: model = data['{model_keys[0]}']")
            else:
                print(f"❌ No model found in dictionary")
                
        elif hasattr(data, '__dict__'):
            model_attrs = [attr for attr in dir(data) if not attr.startswith('_') and hasattr(getattr(data, attr, None), 'predict')]
            if model_attrs:
                print(f"✅ Object format with model at attribute: '{model_attrs[0]}'")
                print(f"💡 Use: model = data.{model_attrs[0]}")
            else:
                print(f"❌ No model found in object attributes")
                
        elif isinstance(data, (list, tuple)):
            model_indices = [i for i, item in enumerate(data) if hasattr(item, 'predict')]
            if model_indices:
                print(f"✅ Sequence format with model at index: {model_indices[0]}")
                print(f"💡 Use: model = data[{model_indices[0]}]")
            else:
                print(f"❌ No model found in sequence")
                
        else:
            if hasattr(data, 'predict'):
                print(f"✅ Direct model format - ready to use!")
                print(f"💡 Use: model = data")
            else:
                print(f"❌ No model found - unknown format")
        
        print(f"\n🔧 FOR HEART ATTACK PREDICTION:")
        print(f"   Expected features: ['Age', 'Gender', 'Diabetes', 'High_BP', 'Smoking']")
        print(f"   Feature count: 5")
        print(f"   Age: numeric (years)")
        print(f"   Gender: 0=Female, 1=Male")
        print(f"   Diabetes: 0=No, 1=Yes")
        print(f"   High_BP: 0=No, 1=Yes")
        print(f"   Smoking: 0=No, 1=Yes")
        
        return {"success": True, "data_type": type(data).__name__, "structure": "analyzed"}
        
    except Exception as e:
        print(f"❌ Error loading PKL file: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Error details: {str(e)}")
        
        # Additional troubleshooting
        print(f"\n🔧 TROUBLESHOOTING:")
        print("-" * 40)
        print(f"1. Check Python version compatibility")
        print(f"2. Ensure all required libraries are installed:")
        print(f"   - scikit-learn")
        print(f"   - numpy")
        print(f"   - pandas")
        print(f"3. Try loading with different pickle protocols")
        print(f"4. Verify the PKL file isn't corrupted")
        
        return {"error": str(e)}

def main():
    """Main function for interactive use"""
    print("🏥 Heart Attack Prediction PKL Inspector")
    print("=" * 60)
    
    # Get file path
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        pkl_path = input("📁 Enter path to your PKL file: ").strip()
        if not pkl_path:
            pkl_path = "heart_attack_combined_model.pkl"  # Default
    
    # Inspect the file
    result = inspect_pkl_file(pkl_path)
    
    print(f"\n" + "=" * 60)
    if result.get("success"):
        print(f"✅ Inspection completed successfully!")
    else:
        print(f"❌ Inspection failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n💡 If you're still having issues:")
    print(f"   1. Share the output above with the development team")
    print(f"   2. Try re-saving your model with: pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)")
    print(f"   3. Verify your model works: model.predict([[50, 1, 0, 0, 0]])")

if __name__ == "__main__":
    main()

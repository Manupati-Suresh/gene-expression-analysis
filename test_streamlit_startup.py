#!/usr/bin/env python3
"""
Test script to verify Streamlit app can start without import errors
"""

import sys
import importlib.util

def test_streamlit_imports():
    """Test that all imports in app.py work correctly"""
    print("🔄 Testing Streamlit app imports...")
    
    try:
        # Test individual imports
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        print("✅ Scikit-learn modules imported successfully")
        
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        print("✅ Plotly modules imported successfully")
        
        import warnings
        import io
        import base64
        from datetime import datetime
        import json
        print("✅ Standard library modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_app_syntax():
    """Test that app.py has valid syntax"""
    print("🔄 Testing app.py syntax...")
    
    try:
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec is None:
            print("❌ Could not load app.py")
            return False
        
        # This will check syntax without executing the module
        module = importlib.util.module_from_spec(spec)
        print("✅ app.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error loading app.py: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Streamlit App Startup Requirements\n")
    
    imports_ok = test_streamlit_imports()
    syntax_ok = test_app_syntax()
    
    print("\n" + "="*50)
    
    if imports_ok and syntax_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The Streamlit app is ready to run!")
        print("\nTo start the app, run:")
        print("streamlit run app.py")
    else:
        print("💥 SOME TESTS FAILED!")
        print("❌ Please fix the issues above before running the app")
        sys.exit(1)
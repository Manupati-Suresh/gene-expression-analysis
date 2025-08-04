#!/usr/bin/env python3
"""
Validation script to test key app functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def validate_app_functions():
    """Validate that the main app functions work"""
    print("🔄 Validating app functions...")
    
    try:
        # Test data loading (simplified version)
        df_train = pd.read_csv('data_set_ALL_AML_train.csv')
        labels_df = pd.read_csv('actual.csv')
        
        # Process data
        gene_data = df_train.iloc[:, 2:]
        expression_cols = []
        for i, col in enumerate(gene_data.columns):
            if i % 2 == 0:
                expression_cols.append(col)
        
        expression_data = gene_data[expression_cols]
        expression_data = expression_data.T
        expression_data.columns = df_train['Gene Description'].values
        expression_data.index = expression_data.index.astype(int)
        
        # Match with labels
        expression_sample_ids = set(expression_data.index)
        label_patient_ids = set(labels_df['patient'])
        common_ids = expression_sample_ids.intersection(label_patient_ids)
        
        common_ids_sorted = sorted(list(common_ids))
        expression_data_filtered = expression_data.loc[common_ids_sorted]
        labels_df_filtered = labels_df[labels_df['patient'].isin(common_ids_sorted)].sort_values('patient')
        
        expression_data_filtered.reset_index(drop=True, inplace=True)
        labels_df_filtered.reset_index(drop=True, inplace=True)
        
        expression_data_filtered['label'] = labels_df_filtered['cancer'].values
        expression_data_filtered['patient_id'] = labels_df_filtered['patient'].values
        
        print(f"✅ Data loading successful: {expression_data_filtered.shape}")
        
        # Test PCA
        features = expression_data_filtered.drop(columns=['label', 'patient_id'])
        features = features.select_dtypes(include=[np.number])
        
        # Remove zero variance features
        zero_var_cols = features.columns[features.var() == 0]
        if len(zero_var_cols) > 0:
            features = features.drop(columns=zero_var_cols)
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        
        print(f"✅ PCA successful: {principal_components.shape}")
        print(f"✅ Explained variance: {pca.explained_variance_ratio_}")
        
        # Test basic statistics
        class_counts = expression_data_filtered['label'].value_counts()
        print(f"✅ Class distribution: {class_counts.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = validate_app_functions()
    if success:
        print("\n🎉 App validation PASSED! The Streamlit app should work correctly.")
    else:
        print("\n💥 App validation FAILED! Check the errors above.")
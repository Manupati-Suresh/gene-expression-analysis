#!/usr/bin/env python3
"""
Simple test script to verify data loading works correctly
"""

import pandas as pd
import numpy as np

def test_data_loading():
    """Test the data loading logic"""
    try:
        print("🔄 Testing data loading...")
        
        # Load the training data
        df_train = pd.read_csv('data_set_ALL_AML_train.csv')
        print(f"✅ Loaded expression data: {df_train.shape}")
        
        # Load the labels
        labels_df = pd.read_csv('actual.csv')
        print(f"✅ Loaded labels: {labels_df.shape}")
        
        # Process the gene expression data
        gene_data = df_train.iloc[:, 2:]
        
        # Extract only the expression values (skip 'call' columns)
        # The pattern is: number, 'call', number, 'call', etc.
        expression_cols = []
        for i, col in enumerate(gene_data.columns):
            if i % 2 == 0:  # Even indices are expression values, odd are 'call'
                expression_cols.append(col)
        
        expression_data = gene_data[expression_cols]
        
        print(f"📊 Expression columns found: {len(expression_cols)}")
        print(f"📊 Sample IDs in expression: {expression_cols[:10]}...")
        
        # Transpose so samples are rows and genes are columns
        expression_data = expression_data.T
        expression_data.columns = df_train['Gene Description'].values
        
        # Convert index to integers for proper matching
        expression_data.index = expression_data.index.astype(int)
        
        print(f"📊 Expression data shape after transpose: {expression_data.shape}")
        print(f"📊 Expression sample IDs: {sorted(expression_data.index.tolist())}")
        print(f"📊 Label patient IDs: {sorted(labels_df['patient'].tolist())}")
        
        # Create a mapping between expression samples and labels
        expression_sample_ids = set(expression_data.index)
        label_patient_ids = set(labels_df['patient'])
        
        # Find common patient IDs
        common_ids = expression_sample_ids.intersection(label_patient_ids)
        
        print(f"✅ Common patient IDs: {len(common_ids)}")
        print(f"📊 Common IDs: {sorted(list(common_ids))}")
        
        if len(common_ids) > 0:
            # Filter both datasets to common IDs
            common_ids_sorted = sorted(list(common_ids))
            expression_data_filtered = expression_data.loc[common_ids_sorted]
            labels_df_filtered = labels_df[labels_df['patient'].isin(common_ids_sorted)].sort_values('patient')
            
            print(f"✅ Final data shape: {expression_data_filtered.shape}")
            print(f"✅ Final labels shape: {labels_df_filtered.shape}")
            print(f"✅ Label distribution: {labels_df_filtered['cancer'].value_counts().to_dict()}")
            
            return True
        else:
            print("❌ No matching patient IDs found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 Data loading test PASSED!")
    else:
        print("\n💥 Data loading test FAILED!")
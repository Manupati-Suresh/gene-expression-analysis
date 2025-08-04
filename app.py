import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Gene Expression Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧬 Advanced Gene Expression Analysis Platform</h1>', unsafe_allow_html=True)

# Introduction with expandable sections
with st.expander("📖 About This Application", expanded=False):
    st.markdown("""
    ### Welcome to the Advanced Gene Expression Analysis Platform!
    
    This comprehensive tool performs sophisticated analysis on gene expression data for cancer classification:
    
    **🎯 Key Features:**
    - **Multi-dimensional Analysis**: PCA, t-SNE, and clustering algorithms
    - **Interactive Visualizations**: 2D/3D plots with real-time parameter adjustment
    - **Machine Learning Integration**: Classification models with cross-validation
    - **Statistical Analysis**: Comprehensive variance analysis and feature importance
    - **Data Quality Assessment**: Missing value analysis and outlier detection
    - **Export Capabilities**: Multiple download formats and detailed reports
    
    **🔬 Dataset:**
    - ALL (Acute Lymphoblastic Leukemia) vs AML (Acute Myeloid Leukemia)
    - High-dimensional gene expression profiles
    - Clinical classification labels
    """)

with st.expander("🚀 Quick Start Guide", expanded=False):
    st.markdown("""
    1. **Explore Data**: Check the dataset overview and quality metrics
    2. **Configure Analysis**: Use the sidebar to adjust parameters
    3. **Visualize Results**: Interact with 2D/3D plots and statistical charts
    4. **Compare Methods**: Try different dimensionality reduction techniques
    5. **Evaluate Models**: Test machine learning classifiers
    6. **Download Results**: Export your analysis in various formats
    """)

@st.cache_data
def load_and_process_data():
    """Load and process the gene expression data with enhanced error handling"""
    try:
        with st.spinner("🔄 Loading and processing gene expression data..."):
            # Load the training data
            df_train = pd.read_csv('data_set_ALL_AML_train.csv')
            
            # Load the labels
            labels_df = pd.read_csv('actual.csv')
            
            # Process the gene expression data
            gene_data = df_train.iloc[:, 2:]
            
            # Extract only the expression values (skip 'call' columns)
            # The pattern is: number, 'call', number, 'call', etc.
            expression_cols = []
            for i, col in enumerate(gene_data.columns):
                if i % 2 == 0:  # Even indices are expression values, odd are 'call'
                    expression_cols.append(col)
            
            expression_data = gene_data[expression_cols]
            
            # Transpose so samples are rows and genes are columns
            expression_data = expression_data.T
            expression_data.columns = df_train['Gene Description'].values
            
            # Convert index to integers for proper matching
            expression_data.index = expression_data.index.astype(int)
            
            # Debug information
            st.info(f"🔍 Data dimensions: Expression data has {len(expression_data)} samples, Labels have {len(labels_df)} entries")
            st.info(f"🔍 Expression sample IDs: {sorted(expression_data.index.tolist())}")
            st.info(f"🔍 Label patient IDs: {sorted(labels_df['patient'].tolist())}")
            
            # Create a mapping between expression samples and labels
            # Match based on patient ID
            expression_sample_ids = set(expression_data.index)
            label_patient_ids = set(labels_df['patient'])
            
            # Find common patient IDs
            common_ids = expression_sample_ids.intersection(label_patient_ids)
            
            if len(common_ids) == 0:
                st.error("❌ No matching patient IDs found between expression data and labels")
                return None, None, None
            
            st.info(f"✅ Found {len(common_ids)} matching samples")
            
            # Filter both datasets to common IDs
            common_ids_sorted = sorted(list(common_ids))
            expression_data_filtered = expression_data.loc[common_ids_sorted]
            labels_df_filtered = labels_df[labels_df['patient'].isin(common_ids_sorted)].sort_values('patient')
            
            # Reset indices
            expression_data_filtered.reset_index(drop=True, inplace=True)
            labels_df_filtered.reset_index(drop=True, inplace=True)
            
            # Add labels and patient IDs
            expression_data_filtered['label'] = labels_df_filtered['cancer'].values
            expression_data_filtered['patient_id'] = labels_df_filtered['patient'].values
            
            # Use the filtered data
            expression_data = expression_data_filtered
            
            # Data quality checks
            missing_values = expression_data.drop(columns=['label', 'patient_id']).isnull().sum().sum()
            if missing_values > 0:
                st.warning(f"⚠️ Found {missing_values} missing values in the dataset")
            
            # Additional data validation
            unique_labels = expression_data['label'].unique()
            if len(unique_labels) < 2:
                st.error("❌ Dataset must contain at least 2 different class labels")
                return None, None, None
            
            # Check for numeric data in expression columns
            numeric_cols = expression_data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = [col for col in expression_data.columns if col not in numeric_cols and col not in ['label', 'patient_id']]
            
            if non_numeric_cols:
                st.warning(f"⚠️ Found {len(non_numeric_cols)} non-numeric expression columns. These will be excluded from analysis.")
                # Remove non-numeric columns except label and patient_id
                expression_data = expression_data.drop(columns=non_numeric_cols)
            
            # Final validation
            if len(expression_data.columns) <= 2:  # Only label and patient_id left
                st.error("❌ No valid numeric expression data found")
                return None, None, None
            
            return expression_data, df_train, {
                'total_samples': len(expression_data),
                'total_genes': len(expression_data.columns) - 2,  # Exclude label and patient_id
                'missing_values': missing_values,
                'class_distribution': expression_data['label'].value_counts().to_dict(),
                'alignment_info': f"Matched {len(common_ids)} samples by patient ID",
                'common_patient_ids': len(common_ids),
                'expression_samples': len(expression_sample_ids),
                'label_samples': len(label_patient_ids)
            }
        
    except FileNotFoundError as e:
        st.error(f"📁 File not found: {str(e)}")
        st.info("Please ensure both 'data_set_ALL_AML_train.csv' and 'actual.csv' are in the repository")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("This might be due to data format issues or dimension mismatches")
        return None, None, None

@st.cache_data
def inspect_data_files():
    """Inspect the structure of data files for debugging"""
    inspection_results = {}
    
    try:
        # Inspect gene expression file
        df_train = pd.read_csv('data_set_ALL_AML_train.csv')
        gene_data = df_train.iloc[:, 2:]
        # Extract only the expression values (skip 'call' columns)
        expression_cols = []
        for i, col in enumerate(gene_data.columns):
            if i % 2 == 0:  # Even indices are expression values, odd are 'call'
                expression_cols.append(col)
        
        inspection_results['expression_file'] = {
            'total_columns': len(df_train.columns),
            'gene_columns': len(gene_data.columns),
            'expression_columns': len(expression_cols),
            'samples_in_expression': len(expression_cols),
            'genes_count': len(df_train),
            'sample_columns': expression_cols[:5] + ['...'] + expression_cols[-5:] if len(expression_cols) > 10 else expression_cols
        }
        
        # Inspect labels file
        labels_df = pd.read_csv('actual.csv')
        inspection_results['labels_file'] = {
            'total_rows': len(labels_df),
            'columns': list(labels_df.columns),
            'unique_labels': labels_df['cancer'].unique().tolist() if 'cancer' in labels_df.columns else [],
            'label_distribution': labels_df['cancer'].value_counts().to_dict() if 'cancer' in labels_df.columns else {}
        }
        
        return inspection_results
        
    except Exception as e:
        return {'error': str(e)}

@st.cache_data
def detect_outliers(data, method='iqr'):
    """Detect outliers in the dataset"""
    features = data.drop(columns=['label', 'patient_id'])
    
    if method == 'iqr':
        Q1 = features.quantile(0.25)
        Q3 = features.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)
    elif method == 'zscore':
        z_scores = np.abs((features - features.mean()) / features.std())
        outliers = (z_scores > 3).any(axis=1)
    
    return outliers

@st.cache_data
def get_feature_statistics(data):
    """Calculate comprehensive feature statistics"""
    features = data.drop(columns=['label', 'patient_id'])
    
    stats = {
        'mean_expression': features.mean().mean(),
        'std_expression': features.std().mean(),
        'min_expression': features.min().min(),
        'max_expression': features.max().max(),
        'zero_variance_genes': (features.var() == 0).sum(),
        'low_variance_genes': (features.var() < 0.1).sum()
    }
    
    return stats

def get_scaler(scaler_type):
    """Get the appropriate scaler based on user selection"""
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    return scalers[scaler_type]

@st.cache_data
def perform_pca(data, n_components=2, scaler_type='StandardScaler'):
    """Perform PCA on the data with enhanced options"""
    try:
        # Separate features and labels
        labels = data['label']
        patient_ids = data['patient_id']
        features = data.drop(columns=['label', 'patient_id'])
        
        # Ensure we have numeric data
        features = features.select_dtypes(include=[np.number])
        
        if features.empty:
            st.error("❌ No numeric features found for PCA analysis")
            return None, None, None, []
        
        # Handle zero variance features
        zero_var_cols = features.columns[features.var() == 0]
        if len(zero_var_cols) > 0:
            features = features.drop(columns=zero_var_cols)
            st.warning(f"⚠️ Removed {len(zero_var_cols)} zero-variance genes")
        
        # Handle infinite or very large values
        features = features.replace([np.inf, -np.inf], np.nan)
        if features.isnull().any().any():
            st.warning("⚠️ Found infinite or missing values, filling with column means")
            features = features.fillna(features.mean())
        
        # Ensure we have enough features for PCA
        if len(features.columns) < n_components:
            st.error(f"❌ Not enough features ({len(features.columns)}) for {n_components} components")
            return None, None, None, []
        
        # Standardize features
        scaler = get_scaler(scaler_type)
        scaled_features = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_features)
        
        # Create DataFrame for PCA results
        pca_df = pd.DataFrame(
            data=principal_components, 
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        pca_df['label'] = labels.values
        pca_df['patient_id'] = patient_ids.values
        
        return pca_df, pca, scaler, features.columns.tolist()
        
    except Exception as e:
        st.error(f"❌ Error in PCA analysis: {str(e)}")
        return None, None, None, []

@st.cache_data
def perform_tsne(data, n_components=2, perplexity=30, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    labels = data['label']
    patient_ids = data['patient_id']
    features = data.drop(columns=['label', 'patient_id'])
    
    # Remove zero variance features
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        features = features.drop(columns=zero_var_cols)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    tsne_components = tsne.fit_transform(scaled_features)
    
    # Create DataFrame for t-SNE results
    tsne_df = pd.DataFrame(
        data=tsne_components,
        columns=[f'tSNE{i+1}' for i in range(n_components)]
    )
    tsne_df['label'] = labels.values
    tsne_df['patient_id'] = patient_ids.values
    
    return tsne_df

@st.cache_data
def perform_clustering(data, n_clusters=2, algorithm='kmeans'):
    """Perform clustering analysis"""
    features = data.drop(columns=['label', 'patient_id'])
    
    # Remove zero variance features
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        features = features.drop(columns=zero_var_cols)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    if algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    cluster_labels = clusterer.fit_predict(scaled_features)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    
    return cluster_labels, silhouette_avg

@st.cache_data
def train_classifiers(data):
    """Train multiple classifiers and return performance metrics"""
    labels = data['label']
    features = data.drop(columns=['label', 'patient_id'])
    
    # Remove zero variance features
    zero_var_cols = features.columns[features.var() == 0]
    if len(zero_var_cols) > 0:
        features = features.drop(columns=zero_var_cols)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, clf in classifiers.items():
        # Perform cross-validation
        cv_scores = cross_val_score(clf, scaled_features, labels, cv=5, scoring='accuracy')
        results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    return results

# Load data
data, raw_data, data_info = load_and_process_data()

if data is not None:
    # Data inspection section (for debugging)
    with st.expander("🔍 Data Inspection (Debug Info)", expanded=False):
        inspection = inspect_data_files()
        if 'error' not in inspection:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Expression File Structure:**")
                exp_info = inspection['expression_file']
                st.write(f"• Total columns: {exp_info['total_columns']}")
                st.write(f"• Expression samples: {exp_info['samples_in_expression']}")
                st.write(f"• Total genes: {exp_info['genes_count']}")
                st.write(f"• Sample columns: {exp_info['sample_columns']}")
            
            with col2:
                st.markdown("**Labels File Structure:**")
                label_info = inspection['labels_file']
                st.write(f"• Total samples: {label_info['total_rows']}")
                st.write(f"• Columns: {label_info['columns']}")
                st.write(f"• Unique labels: {label_info['unique_labels']}")
                st.write(f"• Distribution: {label_info['label_distribution']}")
            
            if data_info.get('alignment_info'):
                st.info(f"📊 {data_info['alignment_info']}")
                st.write(f"• Expression samples available: {data_info.get('expression_samples', 'N/A')}")
                st.write(f"• Label samples available: {data_info.get('label_samples', 'N/A')}")
                st.write(f"• Successfully matched: {data_info.get('common_patient_ids', 'N/A')}")
        else:
            st.error(f"Error inspecting files: {inspection['error']}")

    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Analysis Configuration")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "📊 Analysis Type",
        ["PCA Analysis", "t-SNE Analysis", "Clustering Analysis", "Classification Models", "Comparative Analysis"]
    )
    
    # Common parameters
    st.sidebar.markdown("### 🔧 Parameters")
    
    if analysis_type in ["PCA Analysis", "Comparative Analysis"]:
        n_components = st.sidebar.slider("Number of Components", 2, 10, 2)
        scaler_type = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "RobustScaler", "MinMaxScaler"])
    
    if analysis_type == "t-SNE Analysis":
        tsne_components = st.sidebar.slider("t-SNE Components", 2, 3, 2)
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    
    if analysis_type == "Clustering Analysis":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 2)
        clustering_algorithm = st.sidebar.selectbox("Clustering Algorithm", ["kmeans"])
    
    # Data quality section
    st.sidebar.markdown("### 📈 Data Quality")
    show_outliers = st.sidebar.checkbox("Show Outlier Detection", False)
    if show_outliers:
        outlier_method = st.sidebar.selectbox("Outlier Detection Method", ["iqr", "zscore"])
    
    # Dataset Overview
    with st.expander("📊 Dataset Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Samples", data_info['total_samples'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Genes", data_info['total_genes'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Missing Values", data_info['missing_values'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            all_count = data_info['class_distribution'].get('ALL', 0)
            aml_count = data_info['class_distribution'].get('AML', 0)
            st.metric("ALL:AML Ratio", f"{all_count}:{aml_count}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature statistics
        feature_stats = get_feature_statistics(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Expression Statistics:**")
            st.write(f"• Mean Expression: {feature_stats['mean_expression']:.2f}")
            st.write(f"• Std Expression: {feature_stats['std_expression']:.2f}")
            st.write(f"• Expression Range: [{feature_stats['min_expression']:.2f}, {feature_stats['max_expression']:.2f}]")
        
        with col2:
            st.markdown("**Gene Quality:**")
            st.write(f"• Zero Variance Genes: {feature_stats['zero_variance_genes']}")
            st.write(f"• Low Variance Genes: {feature_stats['low_variance_genes']}")
            
            if feature_stats['zero_variance_genes'] > 0:
                st.markdown('<div class="warning-box">⚠️ Some genes have zero variance and will be excluded from analysis</div>', unsafe_allow_html=True)
    
    # Outlier detection
    if show_outliers:
        with st.expander("🔍 Outlier Detection", expanded=False):
            outliers = detect_outliers(data, method=outlier_method)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                st.markdown(f'<div class="warning-box">⚠️ Detected {outlier_count} potential outlier samples using {outlier_method.upper()} method</div>', unsafe_allow_html=True)
                
                outlier_samples = data[outliers][['patient_id', 'label']]
                st.dataframe(outlier_samples, use_container_width=True)
            else:
                st.markdown('<div class="success-box">✅ No outliers detected in the dataset</div>', unsafe_allow_html=True)
    
    # Main Analysis Section
    if analysis_type == "PCA Analysis":
        st.markdown("## 🔬 Principal Component Analysis")
        
        # Perform PCA
        pca_result = perform_pca(data, n_components, scaler_type)
        
        if pca_result[0] is not None:
            pca_df, pca_model, scaler, feature_names = pca_result
            
            # Main visualization
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if n_components >= 2:
                    # Enhanced 2D plot
                    fig = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2', 
                        color='label',
                        hover_data=['patient_id'],
                        title=f'PCA Visualization ({scaler_type})<br>PC1: {pca_model.explained_variance_ratio_[0]:.2%} variance, PC2: {pca_model.explained_variance_ratio_[1]:.2%} variance',
                        color_discrete_map={'ALL': '#e74c3c', 'AML': '#3498db'},
                        template='plotly_white'
                    )
                    fig.update_traces(marker=dict(size=8, opacity=0.7))
                    fig.update_layout(height=500, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 3D plot
                if n_components >= 3:
                    fig_3d = px.scatter_3d(
                        pca_df, 
                        x='PC1', 
                        y='PC2', 
                        z='PC3',
                        color='label',
                        hover_data=['patient_id'],
                        title=f'3D PCA Visualization',
                        color_discrete_map={'ALL': '#e74c3c', 'AML': '#3498db'},
                        template='plotly_white'
                    )
                    fig_3d.update_traces(marker=dict(size=5, opacity=0.8))
                    fig_3d.update_layout(height=600)
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 PCA Summary")
                
                # Explained variance
                for i, var_ratio in enumerate(pca_model.explained_variance_ratio_):
                    st.metric(f"PC{i+1} Variance", f"{var_ratio:.2%}")
                
                total_variance = sum(pca_model.explained_variance_ratio_)
                st.metric("Total Explained", f"{total_variance:.2%}")
                
                # Class separation quality
                pc1_separation = abs(pca_df[pca_df['label']=='ALL']['PC1'].mean() - 
                                   pca_df[pca_df['label']=='AML']['PC1'].mean())
                pc2_separation = abs(pca_df[pca_df['label']=='ALL']['PC2'].mean() - 
                                   pca_df[pca_df['label']=='AML']['PC2'].mean())
                
                st.markdown("**Class Separation:**")
                st.write(f"PC1: {pc1_separation:.2f}")
                st.write(f"PC2: {pc2_separation:.2f}")
            
            # Component analysis
            st.markdown("### 📈 Component Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced scree plot
                pca_full = PCA()
                features_for_full_pca = data.drop(columns=['label', 'patient_id'])
                zero_var_cols = features_for_full_pca.columns[features_for_full_pca.var() == 0]
                if len(zero_var_cols) > 0:
                    features_for_full_pca = features_for_full_pca.drop(columns=zero_var_cols)
                
                scaler_full = get_scaler(scaler_type)
                scaled_features_full = scaler_full.fit_transform(features_for_full_pca)
                pca_full.fit(scaled_features_full)
                
                n_components_plot = min(20, len(pca_full.explained_variance_ratio_))
                
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Scatter(
                    x=list(range(1, n_components_plot + 1)),
                    y=pca_full.explained_variance_ratio_[:n_components_plot],
                    mode='lines+markers',
                    name='Explained Variance',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8)
                ))
                fig_scree.update_layout(
                    title="Scree Plot - Explained Variance by Component",
                    xaxis_title="Principal Component",
                    yaxis_title="Explained Variance Ratio",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_scree, use_container_width=True)
            
            with col2:
                # Cumulative variance
                cumsum_var = np.cumsum(pca_full.explained_variance_ratio_[:n_components_plot])
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=list(range(1, len(cumsum_var) + 1)),
                    y=cumsum_var,
                    mode='lines+markers',
                    name='Cumulative Variance',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8),
                    fill='tonexty'
                ))
                fig_cum.add_hline(y=0.95, line_dash="dash", line_color="gray", 
                                 annotation_text="95% Variance")
                fig_cum.update_layout(
                    title="Cumulative Explained Variance",
                    xaxis_title="Number of Components",
                    yaxis_title="Cumulative Explained Variance",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.error("❌ PCA analysis failed. Please check your data and parameters.")
    
    elif analysis_type == "t-SNE Analysis":
        st.markdown("## 🎯 t-SNE Analysis")
        
        try:
            with st.spinner("🔄 Computing t-SNE embedding..."):
                tsne_df = perform_tsne(data, tsne_components, perplexity)
            
            if tsne_components == 2:
                fig = px.scatter(
                    tsne_df,
                    x='tSNE1',
                    y='tSNE2',
                    color='label',
                    hover_data=['patient_id'],
                    title=f't-SNE Visualization (Perplexity: {perplexity})',
                    color_discrete_map={'ALL': '#e74c3c', 'AML': '#3498db'},
                    template='plotly_white'
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif tsne_components == 3:
                fig_3d = px.scatter_3d(
                    tsne_df,
                    x='tSNE1',
                    y='tSNE2',
                    z='tSNE3',
                    color='label',
                    hover_data=['patient_id'],
                    title=f'3D t-SNE Visualization (Perplexity: {perplexity})',
                    color_discrete_map={'ALL': '#e74c3c', 'AML': '#3498db'},
                    template='plotly_white'
                )
                fig_3d.update_traces(marker=dict(size=5, opacity=0.8))
                fig_3d.update_layout(height=700)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown('<div class="info-box">💡 t-SNE is particularly good at revealing local structure and clusters in high-dimensional data</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ t-SNE analysis failed: {str(e)}")
            st.info("Try adjusting the perplexity parameter or check your data quality.")
    
    elif analysis_type == "Clustering Analysis":
        st.markdown("## 🎯 Clustering Analysis")
        
        cluster_labels, silhouette_avg = perform_clustering(data, n_clusters, clustering_algorithm)
        
        # Add cluster labels to PCA for visualization
        pca_df, pca_model, scaler, feature_names = perform_pca(data, 2, 'StandardScaler')
        pca_df['cluster'] = cluster_labels
        pca_df['cluster'] = pca_df['cluster'].astype(str)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Clustering visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Clusters', 'True Labels'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Cluster plot
            for cluster in pca_df['cluster'].unique():
                cluster_data = pca_df[pca_df['cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['PC1'],
                        y=cluster_data['PC2'],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8, opacity=0.7)
                    ),
                    row=1, col=1
                )
            
            # True labels plot
            for label in pca_df['label'].unique():
                label_data = pca_df[pca_df['label'] == label]
                fig.add_trace(
                    go.Scatter(
                        x=label_data['PC1'],
                        y=label_data['PC2'],
                        mode='markers',
                        name=label,
                        marker=dict(size=8, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Clustering Metrics")
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            # Calculate ARI if we have true labels
            true_labels_numeric = [0 if label == 'ALL' else 1 for label in data['label']]
            ari_score = adjusted_rand_score(true_labels_numeric, cluster_labels)
            st.metric("Adjusted Rand Index", f"{ari_score:.3f}")
            
            st.markdown("**Cluster Sizes:**")
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                st.write(f"Cluster {cluster}: {count} samples")
    
    elif analysis_type == "Classification Models":
        st.markdown("## 🤖 Machine Learning Classification")
        
        with st.spinner("🔄 Training classification models..."):
            classification_results = train_classifiers(data)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance comparison
            models = list(classification_results.keys())
            accuracies = [classification_results[model]['mean_accuracy'] for model in models]
            std_devs = [classification_results[model]['std_accuracy'] for model in models]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=accuracies,
                error_y=dict(type='data', array=std_devs),
                marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                text=[f"{acc:.3f}" for acc in accuracies],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Cross-Validation Accuracy Comparison",
                xaxis_title="Classifier",
                yaxis_title="Accuracy",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Model Performance")
            
            for model, results in classification_results.items():
                st.markdown(f"**{model}:**")
                st.write(f"• Accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
                
                # Show individual CV scores
                with st.expander(f"CV Scores - {model}"):
                    for i, score in enumerate(results['cv_scores']):
                        st.write(f"Fold {i+1}: {score:.3f}")
    
    elif analysis_type == "Comparative Analysis":
        st.markdown("## 🔄 Comparative Analysis")
        
        # Perform both PCA and t-SNE
        pca_df, pca_model, scaler, feature_names = perform_pca(data, 2, scaler_type)
        
        with st.spinner("🔄 Computing t-SNE for comparison..."):
            tsne_df = perform_tsne(data, 2, 30)
        
        # Side-by-side comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PCA', 't-SNE'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # PCA plot
        for label in pca_df['label'].unique():
            label_data = pca_df[pca_df['label'] == label]
            fig.add_trace(
                go.Scatter(
                    x=label_data['PC1'],
                    y=label_data['PC2'],
                    mode='markers',
                    name=f'PCA-{label}',
                    marker=dict(size=8, opacity=0.7)
                ),
                row=1, col=1
            )
        
        # t-SNE plot
        for label in tsne_df['label'].unique():
            label_data = tsne_df[tsne_df['label'] == label]
            fig.add_trace(
                go.Scatter(
                    x=label_data['tSNE1'],
                    y=label_data['tSNE2'],
                    mode='markers',
                    name=f'tSNE-{label}',
                    marker=dict(size=8, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 PCA Characteristics")
            st.write("• Linear dimensionality reduction")
            st.write("• Preserves global structure")
            st.write("• Interpretable components")
            st.write(f"• Explained variance: {sum(pca_model.explained_variance_ratio_):.2%}")
        
        with col2:
            st.markdown("### 📊 t-SNE Characteristics")
            st.write("• Non-linear dimensionality reduction")
            st.write("• Preserves local structure")
            st.write("• Better for cluster visualization")
            st.write("• Stochastic (results may vary)")
    
    # Enhanced Download Section
    st.markdown("## 📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if analysis_type in ["PCA Analysis", "Comparative Analysis"]:
            # PCA results
            pca_csv = pca_df.to_csv(index=False)
            st.download_button(
                label="📊 Download PCA Results",
                data=pca_csv,
                file_name=f"pca_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if analysis_type == "t-SNE Analysis" or analysis_type == "Comparative Analysis":
            # t-SNE results
            if 'tsne_df' in locals():
                tsne_csv = tsne_df.to_csv(index=False)
                st.download_button(
                    label="🎯 Download t-SNE Results",
                    data=tsne_csv,
                    file_name=f"tsne_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        if analysis_type == "Classification Models":
            # Classification results
            results_json = json.dumps(classification_results, indent=2)
            st.download_button(
                label="🤖 Download ML Results",
                data=results_json,
                file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Analysis report
    with st.expander("📋 Generate Analysis Report", expanded=False):
        st.markdown("### 📄 Comprehensive Analysis Report")
        
        report = f"""
# Gene Expression Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Samples: {data_info['total_samples']}
- Total Genes: {data_info['total_genes']}
- Missing Values: {data_info['missing_values']}
- Class Distribution: {data_info['class_distribution']}

## Analysis Configuration
- Analysis Type: {analysis_type}
"""
        
        if analysis_type in ["PCA Analysis", "Comparative Analysis"]:
            report += f"""
## PCA Results
- Number of Components: {n_components}
- Scaling Method: {scaler_type}
- Total Explained Variance: {sum(pca_model.explained_variance_ratio_):.2%}
- Component Variances: {[f"{var:.2%}" for var in pca_model.explained_variance_ratio_]}
"""
        
        if analysis_type == "Classification Models":
            report += f"""
## Classification Results
"""
            for model, results in classification_results.items():
                report += f"- {model}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}\n"
        
        st.text_area("Report Content", report, height=300)
        
        st.download_button(
            label="📄 Download Full Report",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

else:
    st.markdown('<div class="warning-box">⚠️ Unable to load data. Please ensure the data files are present in the repository.</div>', unsafe_allow_html=True)
    st.info("Expected files: `data_set_ALL_AML_train.csv` and `actual.csv`")
    
    # Show detailed file inspection
    with st.expander("🔍 File Inspection Results", expanded=True):
        inspection = inspect_data_files()
        if 'error' not in inspection:
            st.json(inspection)
        else:
            st.error(f"Cannot inspect files: {inspection['error']}")
    
    # Provide sample data format
    with st.expander("📋 Expected Data Format", expanded=True):
        st.markdown("""
        **data_set_ALL_AML_train.csv** should contain:
        - Gene Description (column 1)
        - Gene Accession Number (column 2)  
        - Expression values for each sample (remaining columns)
        
        **actual.csv** should contain:
        - patient: Patient ID
        - cancer: Cancer type (ALL or AML)
        
        **Common Issues:**
        - Dimension mismatch between expression samples and labels
        - Missing or corrupted data files
        - Incorrect file format or encoding
        - Column naming inconsistencies
        """)
    
    # Offer to create sample data for testing
    with st.expander("🧪 Generate Sample Data for Testing", expanded=False):
        if st.button("Create Sample Dataset"):
            # Create sample gene expression data
            np.random.seed(42)
            n_samples = 50
            n_genes = 100
            
            # Generate sample expression data
            sample_expression = np.random.randn(n_genes, n_samples) * 2 + 5
            
            # Create sample gene descriptions
            gene_descriptions = [f"Gene_{i:03d}" for i in range(n_genes)]
            gene_accessions = [f"ACC_{i:03d}" for i in range(n_genes)]
            
            # Create expression DataFrame
            sample_df = pd.DataFrame({
                'Gene Description': gene_descriptions,
                'Gene Accession Number': gene_accessions
            })
            
            # Add expression columns
            for i in range(n_samples):
                sample_df[str(i+1)] = sample_expression[:, i]
                sample_df[f'{i+1}_call'] = 'P'  # Present call
            
            # Create labels DataFrame
            labels = ['ALL'] * 25 + ['AML'] * 25
            sample_labels = pd.DataFrame({
                'patient': list(range(1, n_samples + 1)),
                'cancer': labels
            })
            
            # Save sample files
            sample_df.to_csv('sample_expression_data.csv', index=False)
            sample_labels.to_csv('sample_labels.csv', index=False)
            
            st.success("✅ Sample data created! Files saved as 'sample_expression_data.csv' and 'sample_labels.csv'")
            st.info("You can rename these files to match the expected names and refresh the app.")
            
            # Show download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📊 Download Sample Expression Data",
                    sample_df.to_csv(index=False),
                    "sample_expression_data.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "🏷️ Download Sample Labels",
                    sample_labels.to_csv(index=False),
                    "sample_labels.csv",
                    "text/csv"
                )
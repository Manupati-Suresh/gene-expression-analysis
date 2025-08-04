# 🚀 Deployment Guide - Advanced Gene Expression Analysis App

## 📋 Pre-Deployment Checklist

### ✅ Files Required
- [x] `app.py` - Main Streamlit application
- [x] `requirements.txt` - Python dependencies
- [x] `data_set_ALL_AML_train.csv` - Gene expression data
- [x] `actual.csv` - Sample labels
- [x] `.streamlit/config.toml` - UI configuration
- [x] `setup.sh` - Deployment setup script
- [x] `Procfile` - Heroku configuration
- [x] `README.md` - Documentation
- [x] `DOCUMENTATION.md` - User guide

### ✅ Data Validation
Run the validation script to ensure everything works:
```bash
python validate_app.py
```

Expected output:
```
🎉 App validation PASSED! The Streamlit app should work correctly.
```

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended)

#### Prerequisites
- GitHub account
- Repository with all files

#### Steps
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy advanced gene expression analysis app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Configuration**:
   - The app will automatically use `.streamlit/config.toml`
   - Dependencies from `requirements.txt` will be installed automatically
   - Data files will be loaded from the repository

#### Expected Deployment Time: 2-5 minutes

### 2. Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Steps
1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create Heroku App**:
   ```bash
   heroku create your-gene-analysis-app
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

4. **Open App**:
   ```bash
   heroku open
   ```

#### Expected Deployment Time: 3-7 minutes

### 3. Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
# Build image
docker build -t gene-expression-app .

# Run container
docker run -p 8501:8501 gene-expression-app
```

### 4. Local Development

#### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd gene-expression-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 🔧 Configuration Options

### Environment Variables
Set these for production deployments:

```bash
# Optional: Set custom port
export PORT=8501

# Optional: Set Streamlit configuration
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### Streamlit Configuration
The app uses `.streamlit/config.toml` for UI customization:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
port = 8501
```

## 📊 Performance Optimization

### For Large Datasets
- **Memory**: Ensure at least 2GB RAM available
- **CPU**: Multi-core recommended for t-SNE and ML models
- **Storage**: ~100MB for app + data files

### Caching Configuration
The app uses Streamlit's caching for:
- Data loading (`@st.cache_data`)
- PCA computations (`@st.cache_data`)
- t-SNE embeddings (`@st.cache_data`)
- ML model training (`@st.cache_data`)

### Performance Tips
1. **First Load**: May take 30-60 seconds for initial data processing
2. **Subsequent Loads**: Cached results load in <5 seconds
3. **Parameter Changes**: Real-time updates with minimal delay
4. **Large Component Numbers**: May increase computation time

## 🛠️ Troubleshooting

### Common Issues

#### 1. Data Loading Errors
**Symptoms**: "Length of values does not match length of index"
**Solution**: The app now automatically handles dimension mismatches

#### 2. Memory Issues
**Symptoms**: App crashes or becomes unresponsive
**Solutions**:
- Reduce number of PCA components
- Use smaller perplexity values for t-SNE
- Clear browser cache

#### 3. Slow Performance
**Symptoms**: Long loading times
**Solutions**:
- Check internet connection
- Avoid changing parameters rapidly
- Use recommended parameter ranges

#### 4. Visualization Issues
**Symptoms**: Plots not displaying correctly
**Solutions**:
- Update browser to latest version
- Disable ad blockers
- Check JavaScript is enabled

### Debug Mode
Enable debug information by expanding the "🔍 Data Inspection" section in the app.

### Log Files
For Heroku deployments, check logs:
```bash
heroku logs --tail
```

## 📈 Monitoring and Maintenance

### Health Checks
The app includes automatic health monitoring:
- Data validation on startup
- Error handling for all major functions
- User-friendly error messages

### Updates
To update the deployed app:
1. Make changes locally
2. Test with `python validate_app.py`
3. Commit and push changes
4. Deployment platforms will auto-update

### Backup
Important files to backup:
- Source code (version controlled)
- Data files (if modified)
- Configuration files

## 🎯 Success Metrics

### Deployment Success Indicators
- ✅ App loads without errors
- ✅ Data inspection shows correct dimensions
- ✅ All 5 analysis types work
- ✅ Visualizations render correctly
- ✅ Download functions work

### Performance Benchmarks
- **Initial Load**: < 60 seconds
- **Parameter Updates**: < 5 seconds
- **Analysis Switching**: < 10 seconds
- **Export Operations**: < 15 seconds

## 📞 Support

### Self-Help Resources
1. Check `DOCUMENTATION.md` for detailed usage guide
2. Review error messages in the app
3. Use the data inspection feature for debugging
4. Run `validate_app.py` for local testing

### Platform-Specific Help
- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **Docker**: [docs.docker.com](https://docs.docker.com)

---

## 🎉 Ready to Deploy!

Your advanced gene expression analysis app is now ready for deployment with:
- ✅ Robust error handling
- ✅ Automatic data alignment
- ✅ Multiple analysis methods
- ✅ Professional UI/UX
- ✅ Comprehensive documentation
- ✅ Production-ready configuration

Choose your preferred deployment method and follow the steps above. The app will provide an excellent platform for gene expression analysis and education!
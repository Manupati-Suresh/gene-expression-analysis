@echo off
echo 🚀 Pushing Gene Expression Analysis App to GitHub
echo.
echo Please make sure you've created a repository on GitHub first!
echo.
set /p REPO_URL="Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git): "
echo.
echo Adding remote origin...
git remote add origin %REPO_URL%
echo.
echo Pushing to GitHub...
git branch -M main
git push -u origin main
echo.
echo ✅ Successfully pushed to GitHub!
echo.
echo Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Connect your GitHub account
echo 3. Select your repository
echo 4. Set main file to: app.py
echo 5. Deploy!
echo.
pause
@echo off
rem PlasticFEM v4 post-processing viewer launcher.
rem Double-click this file. It starts the app and opens your browser.
rem Close the black window to stop the app.
cd /d "%~dp0"
"C:\Users\gimme\anaconda3\python.exe" -m streamlit run "app\post_app.py"
pause

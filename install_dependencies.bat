@echo off
echo Installing EDinai Backend Dependencies...
echo =====================================

echo.
echo 1. Installing Python packages...
pip install -r requirements.txt

echo.
echo 2. Installing additional packages for PDF generation...
pip install reportlab

echo.
echo 3. Checking Tesseract installation...
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract found at C:\Program Files\Tesseract-OCR\tesseract.exe
) else (
    echo ❌ Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki
)

echo.
echo 4. Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "generated_pdfs" mkdir generated_pdfs

echo.
echo 5. Checking .env file...
if exist ".env" (
    echo ✅ .env file exists
) else (
    echo ⚠️ .env file not found. Copying from .env.example...
    copy .env.example .env
    echo ❗ Please edit .env file and add your GROQ_API_KEY
)

echo.
echo =====================================
echo Installation completed!
echo.
echo Next steps:
echo 1. Edit .env file and add your GROQ_API_KEY
echo 2. Run: python -m uvicorn app.main:app --reload --port 8014
echo =====================================
pause

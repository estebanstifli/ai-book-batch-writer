@echo off
setlocal

set "BUILD_VENV=.build-venv"

if not exist "%BUILD_VENV%\Scripts\python.exe" (
  python -m venv "%BUILD_VENV%"
  if errorlevel 1 goto :error
)

"%BUILD_VENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

"%BUILD_VENV%\Scripts\python.exe" -m pip install . "pyinstaller>=6.11,<7"
if errorlevel 1 goto :error

"%BUILD_VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onefile ^
  --name "AI Book Batch Writer" ^
  --paths "src" ^
  --add-data "locales;locales" ^
  --collect-all customtkinter ^
  --hidden-import langchain_anthropic ^
  --hidden-import langchain_google_genai ^
  --hidden-import langchain_openai ^
  --hidden-import langchain_ollama ^
  --hidden-import langchain_openrouter ^
  src\ai_book_batch_writer\main.py
if errorlevel 1 goto :error

echo.
echo Build complete: dist\AI Book Batch Writer.exe
pause
exit /b 0

:error
echo.
echo Build failed.
pause
exit /b 1

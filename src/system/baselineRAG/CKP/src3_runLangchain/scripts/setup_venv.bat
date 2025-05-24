@echo off
echo Setting up Python virtual environment for FlashRAG...

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call .\venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt



echo Setup completed successfully!
echo To activate the environment, run: venv\Scripts\activate.bat
pause 
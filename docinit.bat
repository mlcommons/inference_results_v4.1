@echo off

REM Check if the 'docs' directory exists
if not exist docs (
    git clone https://github.com/GATEOverflow/inference_results_visualization_template.git docs
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)

REM Install Python requirements
python -m pip install -r docs\requirements.txt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM Check if 'overrides' directory exists, if not, copy it
if not exist overrides (
    xcopy /e /i /y docs\overrides overrides
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)

REM Check if 'config.js' exists, and create it if needed
if not exist docs\javascripts\config.js (
    if defined INFERENCE_RESULTS_VERSION (
        echo const results_version="%INFERENCE_RESULTS_VERSION%"; > docs\javascripts\config.js
        for /f "delims=" %%i in ('echo %INFERENCE_RESULTS_VERSION% ^| findstr /r /c:"[0-9]"') do set ver_num=%%i
        echo const dbVersion="%ver_num%"; >> docs\javascripts\config.js
    ) else (
        echo Please export INFERENCE_RESULTS_VERSION=v4.1 or the corresponding version
        exit /b 1
    )
)

REM Check if 'tablesorter' exists in the 'thirdparty' folder
if not exist docs\thirdparty\tablesorter (
    pushd docs\thirdparty
    git clone https://github.com/Mottie/tablesorter.git
    popd
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)

REM Run the Python scripts
python process.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python process_results_table.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

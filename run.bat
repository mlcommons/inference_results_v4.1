@echo off

REM Set environment variables with default values if not already set
IF NOT DEFINED INFERENCE_RESULTS_REPO_OWNER SET INFERENCE_RESULTS_REPO_OWNER=mlcommons
SET INFERENCE_RESULTS_VERSION=v4.1
SET INFERENCE_RESULTS_REPO_BRANCH=main
SET INFERENCE_RESULTS_REPO_NAME=inference_results_v4.1

REM Call the docinit.bat script
call docinit.bat

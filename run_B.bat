@echo off
setlocal

echo [1/4] Train unet_B...
python train.py --model_name="unet_B" --save_dir="./checkpoints/B" --epochs=200
if errorlevel 1 goto :error
echo Waiting 30 minutes before next command...
timeout /t 1800 /nobreak >nul

echo [2/4] Evaluate unet_B...
python evaluate.py --model_name="unet_B" --weights="./checkpoints/B/best_model.pth" --save_dir="./predictions/B"
if errorlevel 1 goto :error
echo Waiting 30 minutes before next command...
timeout /t 1800 /nobreak >nul

echo All commands finished successfully.
exit /b 0

:error
echo Command failed. Exit code: %errorlevel%
exit /b %errorlevel%

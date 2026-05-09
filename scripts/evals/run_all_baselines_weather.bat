@echo off
REM Run all baselines for Weather with corrected plausibility
REM Usage: run_all_baselines_weather.bat

set PYTHONPATH=C:\Users\Dell_Bou_Chrà\Desktop\counterfactual-forecasting-rl

echo ================================================================================
echo   RUNNING ALL BASELINES FOR WEATHER (with corrected plausibility)
echo ================================================================================
echo.

echo ================================================================================
echo   BaseNN - iTransformer
echo ================================================================================
python baselines/BaseNN/run_basenn.py --dataset weather --model itransformer --seeds 1
echo.

echo ================================================================================
echo   BaseNN - DLinear
echo ================================================================================
python baselines/BaseNN/run_basenn.py --dataset weather --model dlinear --seeds 1
echo.

echo ================================================================================
echo   BaseGrad - iTransformer
echo ================================================================================
python baselines/BaseGrad/run_basegrad.py --dataset weather --model itransformer --seeds 1
echo.

echo ================================================================================
echo   BaseGrad - DLinear
echo ================================================================================
python baselines/BaseGrad/run_basegrad.py --dataset weather --model dlinear --seeds 1
echo.

echo ================================================================================
echo   ForecastCF - iTransformer
echo ================================================================================
python baselines/ForecastCF_PyTorch/run_forecastcf_pt.py --dataset weather --model itransformer --seeds 1
echo.

echo ================================================================================
echo   ForecastCF - DLinear
echo ================================================================================
python baselines/ForecastCF_PyTorch/run_forecastcf_pt.py --dataset weather --model dlinear --seeds 1
echo.

echo ================================================================================
echo   ALL BASELINES COMPLETED
echo ================================================================================

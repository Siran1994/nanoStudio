@echo off
cd /d "%~dp0"
echo ================================================================
echo               nanoStudio - Git Push Launcher
echo ================================================================
echo.
echo Pushing local commits to GitHub (origin/main)...
echo.
git push origin main
if errorlevel 1 (
    echo.
    echo [Notice] Push failed or requires authorization in browser.
) else (
    echo.
    echo [Success] Successfully pushed to https://github.com/Siran1994/nanoStudio !
)
echo.
pause

@echo off
rem Instalacion de GestureFlow: entorno de Python, dependencias y acceso directo GestureFlow.lnk.
rem Se puede repetir sin problema (actualiza lo que falte).
cd /d "%~dp0"

set "PY="
py -3.13 -c "" >nul 2>nul && set "PY=py -3.13"
if not defined PY py -3 -c "" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "" >nul 2>nul && set "PY=python"
if not defined PY (
  echo No se encontro Python. Instala Python 3.13 desde https://www.python.org/downloads/ y vuelve a ejecutar setup.bat
  pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando el entorno .venv con %PY% ...
  %PY% -m venv .venv || goto :error
)
echo Instalando dependencias (la primera vez tarda unos minutos)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :error

powershell -NoProfile -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0GestureFlow.lnk');" ^
  "$s.TargetPath='%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments='-m gestureflow.app';" ^
  "$s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0gestureflow\ui\gestureflow.ico,0'; $s.Save()" || goto :error

echo.
echo Listo. Abre GestureFlow con doble clic en GestureFlow.lnk (o run.bat).
echo La primera vez descarga el modelo de manos de MediaPipe (~7.8 MB) y abre el Tutorial.
exit /b 0

:error
echo.
echo La instalacion fallo. Revisa el mensaje de arriba.
pause
exit /b 1

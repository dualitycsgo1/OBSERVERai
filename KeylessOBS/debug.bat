@echo off
echo CS2 Auto Observer - DEBUG MODE
echo ===============================
echo.
echo Dette program vil logge al HTTP trafik fra CS2
echo for at diagnosticere problemer med gamestate integration.
echo.
echo VIGTIGE TING AT CHECKE:
echo 1. Er gamestate_integration_obs.cfg i CS2 cfg mappen?
echo 2. Har du genstartet CS2 efter at have tilføjet filen?
echo 3. Er du observer/spectator i en aktiv match?
echo.
pause
echo Starter debug server...
py debug_server.py
pause

@echo off
echo CS2 Auto Observer - Duel Detection
echo ===================================
echo.
echo Denne avancerede version bruger spillerpositioner til at finde dueller!
echo.
echo Funktioner:
echo - Analyserer afstand mellem spillere af forskellige teams
echo - Tjekker om spillere kigger mod hinanden
echo - Forudsiger hvem der mest sandsynligt vinder duellen
echo - Skifter automatisk til den mest interessante spiller
echo.
echo VIGTIGT: Kør som administrator for tastatur input!
echo.
pause
echo Starter duel detector...
py cs2_duel_detector.py
pause

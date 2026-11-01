# OBSERVERai source

See the [repository README](../README.md) for current setup, training, geometry,
validation commands, and measured limitations. Historical accuracy claims and
standalone EXE instructions from earlier revisions do not validate this source.

From this directory, build the required parser with Go 1.27 or newer:

```powershell
go build -o demo_parser_with_positions.exe .
python ai_trainer_gui.py
python cs2_duel_detector.py
```

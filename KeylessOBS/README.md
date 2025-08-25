# CS2 Auto Observer

En intelligent baggrundsapplikation til Counter-Strike 2 observers der automatisk skifter til den spiller som mest sandsynligt får et kill.

## Funktioner

- 🎯 **Smart spilleranalyse**: Analyserer spillerdata for at forudsige hvem der får næste kill
- ⌨️ **Automatisk tastaturkontrol**: Trykker automatisk på tallene 1-0 for at skifte spiller
- 📊 **Intelligent scoring**: Baseret på health, våben, ammunition og tidligere kills
- 🔧 **Konfigurerbar**: Justerbare vægte og cooldown-tider
- 📝 **Logging**: Detaljeret logging af alle beslutninger

## Installation

1. **Installer Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Konfigurer CS2 Gamestate Integration:**
   - Kopier `gamestate_integration_obs.cfg` til din CS2 cfg mappe:
   - `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`
   - Eller opdater din eksisterende gamestate fil til at pege på `http://192.168.1.45:8082`

## Brug

1. **Start programmet:**
```bash
python cs2_auto_observer.py
```

2. **Start CS2 og gå ind som observer/GOTV**

3. **Kontroller programmet:**
   - `toggle` - Tænd/sluk automatisk skift
   - `status` - Vis nuværende status
   - `cooldown X` - Sæt cooldown til X sekunder
   - `quit` - Afslut program

## Algoritme

Programmet beregner en score for hver spiller baseret på:

### Health (30% vægt)
- Høj health = højere sandsynlighed for at overleve og få kill

### Våben kvalitet (40% vægt)
- AWP: 100% score
- AK47: 90% score  
- M4A4/M4A1: 80% score
- Andre rifles: 60% score
- Pistoler: 20% score

### Ammunition (10% vægt)
- Fuld ammo clip = højere score

### Round kills (20% vægt)
- Spillere der allerede har kills denne runde

### Eksklusion
- Døde spillere får score 0
- Minimum threshold på 0.1 for at udløse skift

## Konfiguration

Rediger `config.json` for at justere:
- Cooldown mellem skift
- Vægte for forskellige faktorer
- Våben scores
- Log niveau

## Gamestate Integration

Din CS2 skal sende data til programmet. Sørg for at din gamestate integration fil indeholder:

```properties
"Observer All Players v.1"
{
 "uri" "http://192.168.1.45:8082"
 "timeout" "5.0"
 "buffer"  "0.1"
 "throttle" "0.1"
 "heartbeat" "30.0"
 "data"
 {
   "allplayers_id"       "1"
   "allplayers_state"    "1"      
   "allplayers_match_stats"  "1"  
   "allplayers_weapons"  "1"      
   "allplayers_position" "1"
   "phase_countdowns"    "1"
   "round"               "1"
 }
}
```

## Fejlfinding

- **Ingen data modtages**: Check at CS2 gamestate integration er konfigureret korrekt
- **Ingen tastaturinput**: Kør programmet som administrator
- **Forkerte skift**: Juster vægtene i config.json
- **For mange skift**: Øg cooldown tiden

## Logs

Alle handlinger logges til `cs2_observer.log` for fejlfinding og optimering.

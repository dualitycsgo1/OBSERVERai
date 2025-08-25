# Installation Guide - CS2 Auto Observer

## Trin 1: Installer Python

1. **Download Python:**
   - Gå til https://python.org
   - Download Python 3.9+ (anbefalet: Python 3.11)
   - **VIGTIGT:** Vælg "Add Python to PATH" under installationen

2. **Verificer installation:**
   - Åbn Command Prompt (cmd)
   - Skriv: `python --version`
   - Du skulle se noget som "Python 3.11.x"

## Trin 2: Konfigurer CS2 Gamestate Integration

1. **Kopier gamestate fil:**
   - Kopier `gamestate_integration_obs.cfg` til:
   - `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\`

2. **Opdater IP adresse:**
   - Åbn `gamestate_integration_obs.cfg`
   - Ændre IP adressen til din computers IP:
   ```
   "uri" "http://DIN_IP_ADRESSE:8082"
   ```
   - Find din IP med: `ipconfig` i Command Prompt

## Trin 3: Start Programmet

1. **Som Administrator:**
   - Højreklik på `start_admin.bat`
   - Vælg "Kør som administrator"

2. **Eller manuelt:**
   - Åbn Command Prompt som administrator
   - Naviger til mappen: `cd "c:\Users\dlndm\Desktop\KeylessOBS"`
   - Kør: `python cs2_auto_observer_standalone.py`

## Trin 4: Test i CS2

1. **Start CS2**
2. **Join en match som observer/GOTV**
3. **Test kommandoer i programmet:**
   - `test 1` - Tester om tast '1' virker
   - `status` - Viser program status
   - `toggle` - Tænd/sluk auto-switch

## Fejlfinding

### "Python blev ikke fundet"
- Geninstaller Python med "Add to PATH" markeret
- Genstart computeren efter installation

### "Ingen data modtages"
- Check at gamestate_integration_obs.cfg er i den rigtige mappe
- Verificer IP adressen i cfg filen
- Sørg for at CS2 kører og du er observer

### "Tastatur input virker ikke"
- Kør programmet som Administrator
- Test med `test 1` kommandoen
- Sørg for at CS2 er det aktive vindue

### "Programmet crasher"
- Check log filen: `cs2_observer.log`
- Kontakt support med log indholdet

## Avanceret Konfiguration

Rediger `config.json` for at justere:

```json
{
    "observer": {
        "switch_cooldown": 1.5,        // Sekunder mellem skift
        "min_kill_probability": 0.2,   // Minimum score for skift
        "enable_auto_switch": true
    },
    "weights": {
        "health": 0.3,                 // Vægt for spillers health
        "weapon_quality": 0.4,         // Vægt for våben kvalitet
        "ammo": 0.1,                   // Vægt for ammunition
        "round_kills": 0.2             // Vægt for kills denne runde
    }
}
```

## Tips til Observing

1. **Optimal Settings:**
   - Cooldown: 1.5-2.0 sekunder
   - Threshold: 0.15-0.25
   - Kør altid som administrator

2. **Best Practices:**
   - Test systemet i warmup runder
   - Juster threshold baseret på spil tempo
   - Brug `toggle` til at deaktivere under pauser

3. **Troubleshooting:**
   - Check `cs2_observer.log` for detaljer
   - Test keyboard input med `test` kommandoen
   - Restart programmet hvis det hænger

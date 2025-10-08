package mainpackage main



import (import (

	"encoding/json""encoding/json"

	"fmt""fmt"

	"log""log"

	"os""os"

	"path/filepath""path/filepath"



	dem "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"dem "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"

	common "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/common"common "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/common"

	events "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"events "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"

))



type Position struct {type Position struct {

	X float64 `json:"x"`X float64 `json:"x"`

	Y float64 `json:"y"`Y float64 `json:"y"`

	Z float64 `json:"z"`Z float64 `json:"z"`

}}



type KillEvent struct {type KillEvent struct {

	Tick            int64    `json:"tick"`Tick            int64    `json:"tick"`

	AttackerName    string   `json:"attacker_name"`AttackerName    string   `json:"attacker_name"`

	AttackerSteamID string   `json:"attacker_steam_id"`AttackerSteamID string   `json:"attacker_steam_id"`

	AttackerPos     Position `json:"attacker_pos"`AttackerPos     Position `json:"attacker_pos"`

	VictimName      string   `json:"victim_name"`VictimName      string   `json:"victim_name"`

	VictimSteamID   string   `json:"victim_steam_id"`VictimSteamID   string   `json:"victim_steam_id"`

	VictimPos       Position `json:"victim_pos"`VictimPos       Position `json:"victim_pos"`

	Weapon          string   `json:"weapon"`Weapon          string   `json:"weapon"`

	IsHeadshot      bool     `json:"is_headshot"`IsHeadshot      bool     `json:"is_headshot"`

	AttackerTeam    string   `json:"attacker_team"`AttackerTeam    string   `json:"attacker_team"`

	VictimTeam      string   `json:"victim_team"`VictimTeam      string   `json:"victim_team"`

	MapName         string   `json:"map_name"`MapName         string   `json:"map_name"`

	RoundNumber     int      `json:"round_number"`RoundNumber     int      `json:"round_number"`

	Distance        float64  `json:"distance"`Distance        float64  `json:"distance"`

	AngleDiff       float64  `json:"angle_diff"`AngleDiff       float64  `json:"angle_diff"`

	AttackerViewX   float64  `json:"attacker_view_x"`AttackerViewX   float64  `json:"attacker_view_x"`

	AttackerViewY   float64  `json:"attacker_view_y"`AttackerViewY   float64  `json:"attacker_view_y"`

	VictimViewX     float64  `json:"victim_view_x"`VictimViewX     float64  `json:"victim_view_x"`

	VictimViewY     float64  `json:"victim_view_y"`VictimViewY     float64  `json:"victim_view_y"`

	DemoFile        string   `json:"demo_file"`DemoFile        string   `json:"demo_file"`

}}



type DemoResult struct {type DemoResult struct {

	DemoFile  string      `json:"demo_file"`DemoFile  string      `json:"demo_file"`

	MapName   string      `json:"map_name"`MapName   string      `json:"map_name"`

	Kills     []KillEvent `json:"kills"`Kills     []KillEvent `json:"kills"`

	Error     string      `json:"error,omitempty"`Error     string      `json:"error,omitempty"`

	Success   bool        `json:"success"`Success   bool        `json:"success"`

	KillCount int         `json:"kill_count"`KillCount int         `json:"kill_count"`

}}



func teamToString(team common.Team) string {func teamToString(team common.Team) string {

	switch team {switch team {

	case common.TeamTerrorists:case common.TeamTerrorists:

		return "T"return "T"

	case common.TeamCounterTerrorists:case common.TeamCounterTerrorists:

		return "CT"return "CT"

	default:default:

		return "UNKNOWN"return "UNKNOWN"

	}}

}}



func calculateDistance(pos1, pos2 Position) float64 {func calculateDistance(pos1, pos2 Position) float64 {

	dx := pos1.X - pos2.Xdx := pos1.X - pos2.X

	dy := pos1.Y - pos2.Ydy := pos1.Y - pos2.Y

	dz := pos1.Z - pos2.Zdz := pos1.Z - pos2.Z

	return dx*dx + dy*dy + dz*dzreturn dx*dx + dy*dy + dz*dz

}}



func calculateAngleDifference(viewAngle1, viewAngle2 float32) float64 {func calculateAngleDifference(viewAngle1, viewAngle2 float32) float64 {

	diff := float64(viewAngle1 - viewAngle2)diff := float64(viewAngle1 - viewAngle2)

	if diff > 180 {if diff > 180 {

		diff -= 360diff -= 360

	} else if diff < -180 {} else if diff < -180 {

		diff += 360diff += 360

	}}

	if diff < 0 {if diff < 0 {

		diff = -diffdiff = -diff

	}}

	return diffreturn diff

}}



func parseDemoFile(filePath string) *DemoResult {func parseDemoFile(filePath string) *DemoResult {

	result := &DemoResult{result := &DemoResult{

		DemoFile: filepath.Base(filePath),DemoFile: filepath.Base(filePath),

		Success:  false,Success:  false,

		Kills:    []KillEvent{},Kills:    []KillEvent{},

	}}



	f, err := os.Open(filePath)f, err := os.Open(filePath)

	if err != nil {if err != nil {

		result.Error = fmt.Sprintf("Failed to open demo file: %v", err)result.Error = fmt.Sprintf("Failed to open demo file: %v", err)

		return resultreturn result

	}}

	defer f.Close()defer f.Close()



	p := dem.NewParser(f)p := dem.NewParser(f)

	defer p.Close()defer p.Close()



	var currentRound int = 1var currentRound int = 1

	var mapName string = "unknown"var mapName string



	// Start parsing to initialize game state	// Parse header - use gamestate after parsing begins

	p.RegisterEventHandler(func(e events.MatchStart) {	// For now, set map name as unknown until we can extract it from events

		// Try to get map name from game state when match starts	mapName = "unknown"

		gs := p.GameState()	result.MapName = mapNamep.RegisterEventHandler(func(e events.RoundStart) {

		if gs != nil {currentRound++

			mapName = "match_started_no_convars" // Default fallback})

		}

	})p.RegisterEventHandler(func(e events.Kill) {

if e.Killer == nil || e.Victim == nil {

	p.RegisterEventHandler(func(e events.RoundStart) {return

		currentRound++}

	})

killerPos := e.Killer.Position()

	p.RegisterEventHandler(func(e events.Kill) {victimPos := e.Victim.Position()

		if e.Killer == nil || e.Victim == nil {

			returndistance := calculateDistance(

		}Position{X: float64(killerPos.X), Y: float64(killerPos.Y), Z: float64(killerPos.Z)},

Position{X: float64(victimPos.X), Y: float64(victimPos.Y), Z: float64(victimPos.Z)},

		killerPos := e.Killer.Position())

		victimPos := e.Victim.Position()

angleDiff := calculateAngleDifference(e.Killer.ViewDirectionX(), e.Victim.ViewDirectionX())

		distance := calculateDistance(

			Position{X: float64(killerPos.X), Y: float64(killerPos.Y), Z: float64(killerPos.Z)},killEvent := KillEvent{

			Position{X: float64(victimPos.X), Y: float64(victimPos.Y), Z: float64(victimPos.Z)},Tick:            int64(p.GameState().IngameTick()),

		)AttackerName:    e.Killer.Name,

AttackerSteamID: fmt.Sprintf("%d", e.Killer.SteamID64),

		angleDiff := calculateAngleDifference(e.Killer.ViewDirectionX(), e.Victim.ViewDirectionX())AttackerPos: Position{

X: float64(killerPos.X),

		killEvent := KillEvent{Y: float64(killerPos.Y),

			Tick:            int64(p.GameState().IngameTick()),Z: float64(killerPos.Z),

			AttackerName:    e.Killer.Name,},

			AttackerSteamID: fmt.Sprintf("%d", e.Killer.SteamID64),VictimName:      e.Victim.Name,

			AttackerPos: Position{VictimSteamID:   fmt.Sprintf("%d", e.Victim.SteamID64),

				X: float64(killerPos.X),VictimPos: Position{

				Y: float64(killerPos.Y),X: float64(victimPos.X),

				Z: float64(killerPos.Z),Y: float64(victimPos.Y),

			},Z: float64(victimPos.Z),

			VictimName:      e.Victim.Name,},

			VictimSteamID:   fmt.Sprintf("%d", e.Victim.SteamID64),Weapon:        e.Weapon.String(),

			VictimPos: Position{IsHeadshot:    e.IsHeadshot,

				X: float64(victimPos.X),AttackerTeam:  teamToString(e.Killer.Team),

				Y: float64(victimPos.Y),VictimTeam:    teamToString(e.Victim.Team),

				Z: float64(victimPos.Z),MapName:       mapName,

			},RoundNumber:   currentRound,

			Weapon:        e.Weapon.String(),Distance:      distance,

			IsHeadshot:    e.IsHeadshot,AngleDiff:     angleDiff,

			AttackerTeam:  teamToString(e.Killer.Team),AttackerViewX: float64(e.Killer.ViewDirectionX()),

			VictimTeam:    teamToString(e.Victim.Team),AttackerViewY: float64(e.Killer.ViewDirectionY()),

			MapName:       mapName,VictimViewX:   float64(e.Victim.ViewDirectionX()),

			RoundNumber:   currentRound,VictimViewY:   float64(e.Victim.ViewDirectionY()),

			Distance:      distance,DemoFile:      result.DemoFile,

			AngleDiff:     angleDiff,}

			AttackerViewX: float64(e.Killer.ViewDirectionX()),

			AttackerViewY: float64(e.Killer.ViewDirectionY()),result.Kills = append(result.Kills, killEvent)

			VictimViewX:   float64(e.Victim.ViewDirectionX()),})

			VictimViewY:   float64(e.Victim.ViewDirectionY()),

			DemoFile:      result.DemoFile,err = p.ParseToEnd()

		}if err != nil {

result.Error = fmt.Sprintf("Error parsing demo: %v", err)

		result.Kills = append(result.Kills, killEvent)return result

	})}



	err = p.ParseToEnd()result.Success = true

	if err != nil {result.KillCount = len(result.Kills)

		result.Error = fmt.Sprintf("Error parsing demo: %v", err)return result

		return result}

	}

func main() {

	result.Success = trueif len(os.Args) < 2 {

	result.KillCount = len(result.Kills)log.Fatal("Usage: demo_parser_with_positions.exe <demo_file_path>")

	result.MapName = mapName}

	return result

}demoPath := os.Args[1]



func main() {if _, err := os.Stat(demoPath); os.IsNotExist(err) {

	if len(os.Args) < 2 {result := &DemoResult{

		log.Fatal("Usage: demo_parser_with_positions.exe <demo_file_path>")DemoFile: filepath.Base(demoPath),

	}Success:  false,

Error:    fmt.Sprintf("Demo file does not exist: %s", demoPath),

	demoPath := os.Args[1]Kills:    []KillEvent{},

}

	if _, err := os.Stat(demoPath); os.IsNotExist(err) {output, _ := json.MarshalIndent(result, "", "  ")

		result := &DemoResult{fmt.Println(string(output))

			DemoFile: filepath.Base(demoPath),return

			Success:  false,}

			Error:    fmt.Sprintf("Demo file does not exist: %s", demoPath),

			Kills:    []KillEvent{},result := parseDemoFile(demoPath)

		}

		output, _ := json.MarshalIndent(result, "", "  ")output, err := json.MarshalIndent(result, "", "  ")

		fmt.Println(string(output))if err != nil {

		returnlog.Printf("Error marshaling JSON: %v", err)

	}fmt.Println(`{"success": false, "error": "Failed to create JSON output"}`)

} else {

	result := parseDemoFile(demoPath)fmt.Println(string(output))

}

	output, err := json.MarshalIndent(result, "", "  ")}

	if err != nil {
		log.Printf("Error marshaling JSON: %v", err)
		fmt.Println(`{"success": false, "error": "Failed to create JSON output"}`)
	} else {
		fmt.Println(string(output))
	}
}
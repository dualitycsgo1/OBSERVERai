// Demo parser and causal next-kill training snapshot exporter.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	dem "github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v6/pkg/demoinfocs/msg"
)

const horizonSeconds = 2.0
const sampleIntervalSeconds = 0.5

type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type KillEvent struct {
	Tick            int      `json:"tick"`
	Time            float64  `json:"time"`
	AttackerName    string   `json:"attacker_name"`
	AttackerSteamID string   `json:"attacker_steam_id"`
	AttackerPos     Position `json:"attacker_pos"`
	VictimName      string   `json:"victim_name"`
	VictimSteamID   string   `json:"victim_steam_id"`
	VictimPos       Position `json:"victim_pos"`
	Weapon          string   `json:"weapon"`
	IsHeadshot      bool     `json:"is_headshot"`
	Wallbang        bool     `json:"wallbang"`
	AttackerTeam    string   `json:"attacker_team"`
	VictimTeam      string   `json:"victim_team"`
	MapName         string   `json:"map_name"`
	RoundNumber     int      `json:"round_number"`
	Distance        float64  `json:"distance"`
	AngleDiff       float64  `json:"angle_diff"`
	AttackerViewX   float64  `json:"attacker_view_x"`
	AttackerViewY   float64  `json:"attacker_view_y"`
	VictimViewX     float64  `json:"victim_view_x"`
	VictimViewY     float64  `json:"victim_view_y"`
	DemoFile        string   `json:"demo_file"`
}

type PlayerState struct {
	Health int `json:"health"`
	Armor  int `json:"armor"`
}

type WeaponState struct {
	Name     string `json:"name"`
	State    string `json:"state"`
	AmmoClip int    `json:"ammo_clip"`
}

type PlayerSnapshot struct {
	Name     string                 `json:"name"`
	Team     string                 `json:"team"`
	Position string                 `json:"position"`
	Forward  string                 `json:"forward"`
	State    PlayerState            `json:"state"`
	Weapons  map[string]WeaponState `json:"weapons"`
}

type MapSnapshot struct {
	Name  string `json:"name"`
	Round int    `json:"round"`
}

type GameSnapshot struct {
	AllPlayers map[string]PlayerSnapshot `json:"allplayers"`
	Map        MapSnapshot               `json:"map"`
	Round      map[string]string         `json:"round"`
	Provider   map[string]float64        `json:"provider"`
}

type PredictionSample struct {
	DemoID         string       `json:"demo_id"`
	SampleTick     int          `json:"sample_tick"`
	SampleTime     float64      `json:"sample_time"`
	HorizonSeconds float64      `json:"horizon_seconds"`
	GameState      GameSnapshot `json:"gamestate"`
	PlayerID       string       `json:"player_id"`
	Label          int          `json:"label"`
	Target         string       `json:"target"`
	SnapshotPhase  string       `json:"snapshot_phase"`
}

type pendingSnapshot struct {
	Tick  int
	Time  float64
	Round int
	State GameSnapshot
}

type DemoResult struct {
	DemoFile          string                 `json:"demo_file"`
	DemoID            string                 `json:"demo_id"`
	MapName           string                 `json:"map_name"`
	ServerName        string                 `json:"server_name,omitempty"`
	PatchVersion      *int32                 `json:"patch_version,omitempty"`
	BuildNum          *int32                 `json:"build_num,omitempty"`
	DemoVersionName   string                 `json:"demo_version_name,omitempty"`
	DemoType          string                 `json:"demo_type"`
	Kills             []KillEvent            `json:"kills"`
	PredictionSamples []PredictionSample     `json:"prediction_samples"`
	DataQuality       map[string]interface{} `json:"data_quality"`
	Error             string                 `json:"error,omitempty"`
	Success           bool                   `json:"success"`
	KillCount         int                    `json:"kill_count"`
}

func teamToString(team common.Team) string {
	switch team {
	case common.TeamTerrorists:
		return "T"
	case common.TeamCounterTerrorists:
		return "CT"
	default:
		return "UNKNOWN"
	}
}

func calculateDistance(a, b Position) float64 {
	dx, dy, dz := a.X-b.X, a.Y-b.Y, a.Z-b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func aimYawError(attacker, target Position, yaw float64) float64 {
	targetYaw := math.Atan2(target.Y-attacker.Y, target.X-attacker.X) * 180 / math.Pi
	return math.Abs(math.Mod(targetYaw-yaw+540, 360) - 180)
}

func forwardVector(yawDegrees, pitchDegrees float64) string {
	yaw, pitch := yawDegrees*math.Pi/180, pitchDegrees*math.Pi/180
	return fmt.Sprintf("%.8f, %.8f, %.8f", math.Cos(pitch)*math.Cos(yaw), math.Cos(pitch)*math.Sin(yaw), -math.Sin(pitch))
}

func weaponName(name string) string {
	clean := strings.ToLower(strings.NewReplacer("-", "", " ", "", "_", "").Replace(name))
	aliases := map[string]string{
		"usps": "usp_silencer", "m4a1": "m4a1_silencer", "m4a1s": "m4a1_silencer",
		"m4a4": "m4a1", "sg553": "sg556", "deserteagle": "deagle", "glock18": "glock",
		"dualberettas": "elite", "p2000": "hkp2000", "ppbizon": "bizon",
		"r8revolver": "revolver", "cz75auto": "cz75a", "decoygrenade": "decoy",
		"incendiarygrenade": "incgrenade", "zeusx27": "taser",
	}
	if alias, ok := aliases[clean]; ok {
		clean = alias
	}
	return "weapon_" + clean
}

// Labels are derived only after the entire horizon has been observed. Death events
// at/before the snapshot tick are excluded, including events in its own frame.
func labelSnapshot(snapshot pendingSnapshot, kills []KillEvent, demoID string) []PredictionSample {
	samples := make([]PredictionSample, 0, len(snapshot.State.AllPlayers))
	for playerID := range snapshot.State.AllPlayers {
		label := 0
		for _, kill := range kills {
			if kill.RoundNumber == snapshot.Round && kill.Tick > snapshot.Tick &&
				kill.Time > snapshot.Time && kill.Time <= snapshot.Time+horizonSeconds &&
				kill.AttackerSteamID == playerID && kill.AttackerTeam != kill.VictimTeam {
				label = 1
				break
			}
		}
		samples = append(samples, PredictionSample{
			DemoID: demoID, SampleTick: snapshot.Tick, SampleTime: snapshot.Time, HorizonSeconds: horizonSeconds,
			GameState: snapshot.State, PlayerID: playerID, Label: label,
			Target: "kill_within_horizon", SnapshotPhase: "pre_event",
		})
	}
	return samples
}

func matureSnapshots(pending []pendingSnapshot, kills []KillEvent, demoID string, now float64) ([]PredictionSample, []pendingSnapshot) {
	samples := []PredictionSample{}
	remaining := []pendingSnapshot{}
	for _, snapshot := range pending {
		if now >= snapshot.Time+horizonSeconds {
			samples = append(samples, labelSnapshot(snapshot, kills, demoID)...)
		} else {
			remaining = append(remaining, snapshot)
		}
	}
	return samples, remaining
}

func parseDemoFile(filePath string) (result *DemoResult) {
	result = &DemoResult{
		DemoFile: filepath.Base(filePath), DemoType: "CS2",
		Kills: []KillEvent{}, PredictionSamples: []PredictionSample{},
		DataQuality: map[string]interface{}{
			"sample_interval_seconds": sampleIntervalSeconds, "horizon_seconds": horizonSeconds,
			"snapshot_phase": "pre_event", "partial_horizons": "discarded",
			"positions": "demo_player_feet", "angles": "demo_view_angles",
			"timestamp_basis": "demo_elapsed_seconds", "parser_version": "demoinfocs/v6.0.0-rc.1",
		},
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			result.Success = false
			result.Error = fmt.Sprintf("Parser panic: %v", recovered)
			result.PredictionSamples = []PredictionSample{}
		}
	}()
	file, err := os.Open(filePath)
	if err != nil {
		result.Error = fmt.Sprintf("Open demo: %v", err)
		return result
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err = io.Copy(hasher, file); err != nil {
		result.Error = fmt.Sprintf("Hash demo: %v", err)
		return result
	}
	result.DemoID = hex.EncodeToString(hasher.Sum(nil))
	if _, err = file.Seek(0, io.SeekStart); err != nil {
		result.Error = err.Error()
		return result
	}
	parser := dem.NewParser(file)
	defer parser.Close()
	// Source 2 headers are exposed through net-message handlers.
	parser.RegisterNetMessageHandler(func(header *msg.CDemoFileHeader) {
		result.MapName = header.GetMapName()
		result.ServerName = header.GetServerName()
		result.PatchVersion = header.PatchVersion
		result.BuildNum = header.BuildNum
		result.DemoVersionName = header.GetDemoVersionName()
		if header.PatchVersion != nil {
			result.DataQuality["patch_version"] = *header.PatchVersion
		}
		if header.BuildNum != nil {
			result.DataQuality["build_num"] = *header.BuildNum
		}
		if result.DemoVersionName != "" {
			result.DataQuality["demo_version_name"] = result.DemoVersionName
		}
	})
	roundLive := false
	currentRound := 1
	pending := []pendingSnapshot{}
	lastSampleTime := math.Inf(-1)
	discarded := 0

	// Only complete horizons wholly inside one live round are retained.
	mature := func(now float64) {
		samples, remaining := matureSnapshots(pending, result.Kills, result.DemoID, now)
		result.PredictionSamples = append(result.PredictionSamples, samples...)
		pending = remaining
	}
	startLiveRound := func() {
		if roundLive {
			return
		}
		roundLive = true
		currentRound = parser.GameState().TotalRoundsPlayed() + 1
		lastSampleTime = math.Inf(-1)
	}
	parser.RegisterEventHandler(func(event events.RoundFreezetimeEnd) {
		startLiveRound()
	})
	parser.RegisterEventHandler(func(event events.RoundFreezetimeChanged) {
		// Source 2's mimicked events use this state transition rather than
		// always delivering the legacy RoundFreezetimeEnd event.
		if event.OldIsFreezetime && !event.NewIsFreezetime {
			startLiveRound()
		} else if event.NewIsFreezetime {
			discarded += len(pending)
			pending = nil
			roundLive = false
		}
	})
	parser.RegisterEventHandler(func(event events.RoundEnd) {
		mature(parser.CurrentTime().Seconds())
		discarded += len(pending)
		pending = nil
		roundLive = false
	})
	parser.RegisterEventHandler(func(event events.RoundStart) {
		discarded += len(pending)
		pending = nil
		roundLive = false
		currentRound = parser.GameState().TotalRoundsPlayed() + 1
	})
	parser.RegisterEventHandler(func(event events.Kill) {
		if event.Killer == nil || event.Victim == nil || event.Killer == event.Victim {
			return
		}
		killer, victim := event.Killer, event.Victim
		kp, vp := killer.Position(), victim.Position()
		attackerPos := Position{kp.X, kp.Y, kp.Z}
		victimPos := Position{vp.X, vp.Y, vp.Z}
		weapon := "unknown"
		if event.Weapon != nil {
			weapon = weaponName(event.Weapon.String())
		}
		result.Kills = append(result.Kills, KillEvent{
			Tick: parser.GameState().IngameTick(), Time: parser.CurrentTime().Seconds(),
			AttackerName: killer.Name, AttackerSteamID: strconv.FormatUint(killer.SteamID64, 10),
			AttackerPos: attackerPos, VictimName: victim.Name,
			VictimSteamID: strconv.FormatUint(victim.SteamID64, 10), VictimPos: victimPos,
			Weapon: weapon, IsHeadshot: event.IsHeadshot, Wallbang: event.PenetratedObjects > 0,
			AttackerTeam: teamToString(killer.Team), VictimTeam: teamToString(victim.Team),
			MapName: result.MapName, RoundNumber: currentRound,
			Distance:      calculateDistance(attackerPos, victimPos),
			AngleDiff:     aimYawError(attackerPos, victimPos, float64(killer.ViewDirectionX())),
			AttackerViewX: float64(killer.ViewDirectionX()), AttackerViewY: float64(killer.ViewDirectionY()),
			VictimViewX: float64(victim.ViewDirectionX()), VictimViewY: float64(victim.ViewDirectionY()),
			DemoFile: result.DemoFile,
		})
	})
	parser.RegisterEventHandler(func(event events.FrameDone) {
		gs := parser.GameState()
		now := parser.CurrentTime().Seconds()
		if !roundLive || gs.IsWarmupPeriod() || gs.IsFreezetimePeriod() {
			return
		}
		mature(now)
		if now-lastSampleTime < sampleIntervalSeconds {
			return
		}
		players := map[string]PlayerSnapshot{}
		for _, player := range gs.Participants().Playing() {
			if player == nil || player.PlayerPawnEntity() == nil || !player.IsAlive() || player.SteamID64 == 0 ||
				(player.Team != common.TeamTerrorists && player.Team != common.TeamCounterTerrorists) {
				continue
			}
			pos := player.Position()
			weapons := map[string]WeaponState{}
			if active := player.ActiveWeapon(); active != nil {
				weapons["weapon_0"] = WeaponState{Name: weaponName(active.String()), State: "active", AmmoClip: active.AmmoInMagazine()}
			}
			playerID := strconv.FormatUint(player.SteamID64, 10)
			players[playerID] = PlayerSnapshot{
				Name: player.Name, Team: teamToString(player.Team),
				Position: fmt.Sprintf("%.8f, %.8f, %.8f", pos.X, pos.Y, pos.Z),
				Forward:  forwardVector(float64(player.ViewDirectionX()), float64(player.ViewDirectionY())),
				State:    PlayerState{Health: player.Health(), Armor: player.Armor()}, Weapons: weapons,
			}
		}
		hasT, hasCT := false, false
		for _, player := range players {
			hasT = hasT || player.Team == "T"
			hasCT = hasCT || player.Team == "CT"
		}
		if !hasT || !hasCT {
			return
		}
		pending = append(pending, pendingSnapshot{
			Tick: gs.IngameTick(), Time: now, Round: currentRound,
			State: GameSnapshot{AllPlayers: players, Map: MapSnapshot{Name: result.MapName, Round: currentRound - 1},
				Round: map[string]string{"phase": "live"}, Provider: map[string]float64{"timestamp": now}},
		})
		lastSampleTime = now
	})
	if err = parser.ParseToEnd(context.Background()); err != nil {
		result.Error = fmt.Sprintf("Parse demo: %v", err)
		result.PredictionSamples = []PredictionSample{}
		return result
	}
	// No RoundEnd event is guaranteed at EOF. These horizons are censored.
	discarded += len(pending)
	result.DataQuality["discarded_partial_snapshots"] = discarded
	result.KillCount = len(result.Kills)
	result.Success = true
	return result
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: demo_parser_with_positions.exe <demo_file_path>")
		os.Exit(2)
	}
	result := parseDemoFile(os.Args[1])
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if !result.Success {
		os.Exit(1)
	}
}

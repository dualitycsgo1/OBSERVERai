package main

import (
	"math"
	"testing"
)

func TestDistanceAndAim(t *testing.T) {
	if got := calculateDistance(Position{}, Position{3, 4, 12}); got != 13 {
		t.Fatalf("distance must be Euclidean units, got %v", got)
	}
	if got := aimYawError(Position{}, Position{1, 0, 0}, 359); math.Abs(got-1) > 1e-8 {
		t.Fatalf("wrapped angle error = %v", got)
	}
}

func TestCausalLabelsAndNegatives(t *testing.T) {
	snapshot := pendingSnapshot{Tick: 100, Time: 10, Round: 2,
		State: GameSnapshot{AllPlayers: map[string]PlayerSnapshot{"1": {}, "2": {}}}}
	kills := []KillEvent{
		{Tick: 100, Time: 10, RoundNumber: 2, AttackerSteamID: "2", AttackerTeam: "T", VictimTeam: "CT"},
		{Tick: 101, Time: 10.1, RoundNumber: 1, AttackerSteamID: "2", AttackerTeam: "T", VictimTeam: "CT"},
		{Tick: 102, Time: 10.2, RoundNumber: 2, AttackerSteamID: "2", AttackerTeam: "T", VictimTeam: "T"},
		{Tick: 110, Time: 11, RoundNumber: 2, AttackerSteamID: "1", AttackerTeam: "T", VictimTeam: "CT"},
		{Tick: 130, Time: 13, RoundNumber: 2, AttackerSteamID: "2", AttackerTeam: "T", VictimTeam: "CT"},
	}
	samples := labelSnapshot(snapshot, kills, "demo-sha")
	if len(samples) != 2 {
		t.Fatal("must label every alive candidate")
	}
	for _, sample := range samples {
		if (sample.PlayerID == "1" && sample.Label != 1) || (sample.PlayerID == "2" && sample.Label != 0) {
			t.Fatalf("causal label wrong: %+v", sample)
		}
		if sample.Target != "kill_within_horizon" || sample.SnapshotPhase != "pre_event" || sample.DemoID != "demo-sha" {
			t.Fatalf("missing provenance: %+v", sample)
		}
		if sample.SampleTime != 10 {
			t.Fatalf("replay time must be the snapshot time, got %v", sample.SampleTime)
		}
	}
}

func TestHorizonBoundaryInclusive(t *testing.T) {
	snapshot := pendingSnapshot{Tick: 1, Time: 1, Round: 1,
		State: GameSnapshot{AllPlayers: map[string]PlayerSnapshot{"1": {}}}}
	samples := labelSnapshot(snapshot, []KillEvent{{Tick: 129, Time: 3, RoundNumber: 1,
		AttackerSteamID: "1", AttackerTeam: "T", VictimTeam: "CT"}}, "demo")
	if samples[0].Label != 1 {
		t.Fatal("horizon endpoint must be included")
	}
}

func TestIncompleteHorizonsNeverReceiveLabels(t *testing.T) {
	pending := []pendingSnapshot{
		{Tick: 1, Time: 10, State: GameSnapshot{AllPlayers: map[string]PlayerSnapshot{"1": {}}}},
		{Tick: 2, Time: 11, State: GameSnapshot{AllPlayers: map[string]PlayerSnapshot{"1": {}}}},
	}
	samples, remaining := matureSnapshots(pending, nil, "demo", 12)
	if len(samples) != 1 || samples[0].SampleTick != 1 || len(remaining) != 1 || remaining[0].Tick != 2 {
		t.Fatalf("only the complete horizon should mature: samples=%+v remaining=%+v", samples, remaining)
	}
}

func TestForwardUsesSourcePitchConvention(t *testing.T) {
	if got := forwardVector(0, 90); got != "0.00000000, 0.00000000, -1.00000000" {
		t.Fatalf("positive Source pitch points down, got %q", got)
	}
}

func TestParserWeaponNamesMatchGSI(t *testing.T) {
	for name, expected := range map[string]string{
		"Desert Eagle": "weapon_deagle", "Glock-18": "weapon_glock",
		"AK-47": "weapon_ak47", "M4A1": "weapon_m4a1_silencer", "M4A4": "weapon_m4a1",
		"Dual Berettas": "weapon_elite", "USP-S": "weapon_usp_silencer", "P2000": "weapon_hkp2000",
	} {
		if actual := weaponName(name); actual != expected {
			t.Fatalf("%s: got %s want %s", name, actual, expected)
		}
	}
}

func TestMissingDemoFails(t *testing.T) {
	result := parseDemoFile("this-demo-does-not-exist.dem")
	if result.Success || result.Error == "" || len(result.PredictionSamples) != 0 {
		t.Fatalf("must report failure without fabricated samples: %+v", result)
	}
}

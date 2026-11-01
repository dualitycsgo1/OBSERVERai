"""Versioned, shared feature contract for *future* kill prediction.

Only quantities available in a live GSI snapshot are used. Kill-event records,
headshot labels and post-event player health are not prediction examples.
"""
import math

SCHEMA_VERSION = 1
TARGET = "kill_within_horizon"
HORIZON_SECONDS = 2.0
FEATURE_NAMES = (
    "health", "armor", "ammo_clip", "enemies_alive", "teammates_alive",
    "nearest_enemy_distance", "nearest_enemy_height_delta", "nearest_enemy_angle",
    "best_aim_angle", "best_aim_distance", "weapon_rifle", "weapon_sniper",
    "weapon_smg", "weapon_shotgun", "weapon_pistol", "weapon_knife", "weapon_other",
)


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite snapshot value")
    return result


def vector(value):
    if isinstance(value, str):
        value = value.split(",")
    elif isinstance(value, dict):
        value = [value[axis] for axis in ("x", "y", "z")]
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("Expected a three-dimensional vector")
    return tuple(_number(component) for component in value)


def _forward(player):
    if player.get("forward") is not None:
        direction = vector(player["forward"])
    else:
        # Explicit Source view angles are in degrees; positive pitch looks down.
        angles = player["view_angles"]
        yaw, pitch = math.radians(_number(angles["yaw"])), math.radians(_number(angles["pitch"]))
        direction = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    length = math.sqrt(sum(v * v for v in direction))
    if length <= 1e-9:
        raise ValueError("Missing view direction")
    return tuple(v / length for v in direction)


def _weapon(player):
    weapons = player.get("weapons", {})
    for weapon in weapons.values():
        if weapon.get("state") == "active":
            return weapon
    return {}


def feature_vector(gamestate, player_id):
    """Return the same ordered features for training and inference, or None.

    Missing optional armor/ammo values are represented by -1, never invented
    healthy-player or loaded-weapon values. Position, team, health and view are
    required. Dead players are excluded on both sides of the engagement.
    """
    try:
        players = gamestate["allplayers"]
        player = players[player_id]
        team = player["team"]
        health = _number(player["state"]["health"])
        if team not in ("T", "CT") or health <= 0:
            return None
        position, forward = vector(player["position"]), _forward(player)
        enemies, teammates = [], 0
        for other_id, other in players.items():
            if (other_id in (player_id, 'total') or not isinstance(other, dict)
                    or other.get("team") not in ("T", "CT")):
                continue
            # Unknown health cannot establish that another playing-team player
            # is dead. Abstain instead of silently reducing the alive counts.
            if _number(other["state"]["health"]) <= 0:
                continue
            if other["team"] == team:
                teammates += 1
                continue
            target = vector(other["position"])
            delta = tuple(b - a for a, b in zip(position, target))
            distance = math.sqrt(sum(v * v for v in delta))
            if distance <= 1e-9:
                continue
            cosine = sum(a * b for a, b in zip(forward, delta)) / distance
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            enemies.append((distance, delta[2], angle))
        if not enemies:
            return None
        nearest = min(enemies)  # Stable tie-break independent of GSI/map key order.
        aimed = min(enemies, key=lambda enemy: (enemy[2], enemy[0]))
        weapon = _weapon(player)
        name = weapon.get("name", "").lower().removeprefix("weapon_")
        categories = (
            {"ak47", "m4a1", "m4a1_silencer", "m4a4", "aug", "sg556", "sg553", "famas", "galilar"},
            {"awp", "ssg08", "scar20", "g3sg1"},
            {"mp9", "mp7", "mp5sd", "ump45", "p90", "bizon", "mac10"},
            {"nova", "xm1014", "mag7", "sawedoff"},
            {"glock", "hkp2000", "usp_silencer", "p250", "deagle", "elite", "fiveseven", "tec9", "cz75a", "revolver"},
        )
        flags = [float(name in category) for category in categories]
        flags.append(float("knife" in name or name == "bayonet"))
        flags.append(float(not any(flags)))
        return [health, _number(player["state"].get("armor", -1)), _number(weapon.get("ammo_clip", -1)),
                float(len(enemies)), float(teammates), *nearest, aimed[2], aimed[0], *flags]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def validate_model_bundle(bundle):
    """Reject legacy headshot estimators and incompatible feature orders."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported observer model schema; retrain on pre-event snapshots")
    if bundle.get("target") != TARGET or bundle.get("horizon_seconds") != HORIZON_SECONDS:
        raise ValueError("Model does not predict the supported future-kill target")
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Observer model feature schema mismatch")
    model = bundle.get("model")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model lacks predict_proba")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError("A kill model requires both negative and positive examples")
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("Estimator feature count mismatch")
    for field in ("train_demo_ids", "validation_demo_ids"):
        identities = bundle.get(field)
        if (not isinstance(identities, (list, tuple)) or not identities or
                any(not isinstance(identity, str) or not identity for identity in identities)):
            raise ValueError("Model requires non-empty training and validation demo identities")
    if set(bundle["train_demo_ids"]) & set(bundle["validation_demo_ids"]):
        raise ValueError("Training/validation demos overlap")
    evaluation = bundle.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("promotion_passed") is not True:
        raise ValueError("Model has not passed held-out evaluation")
    return bundle


def predict_probabilities(bundle, gamestate, player_ids):
    """Score all eligible players in one estimator call per observer update."""
    validate_model_bundle(bundle)
    identities, matrix = [], []
    for player_id in dict.fromkeys(player_ids):
        features = feature_vector(gamestate, player_id)
        if features is not None:
            identities.append(player_id)
            matrix.append(features)
    if not matrix:
        return {}
    predictions = bundle["model"].predict_proba(matrix)
    if len(predictions) != len(identities):
        raise ValueError("Estimator returned a mismatched number of probabilities")
    result = {}
    for player_id, prediction in zip(identities, predictions):
        if len(prediction) != 2:
            raise ValueError("Estimator returned an invalid probability row")
        score = float(prediction[1])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid model probability")
        result[player_id] = score
    return result


def predict_probability(bundle, gamestate, player_id):
    return predict_probabilities(bundle, gamestate, [player_id]).get(player_id)

# Case Schema Guide

This document explains how prebuilt case files in `cases/prebuilt/*.json` are used by the server.

## Layers

Each case file has two layers.

1. Briefing and UI data
- `selection_card`
- `documents`
- `suspect_profile.default_personality`
- `suspect_profile.mental_state`

2. Interrogation engine data
- `overview`
- `suspect`
- `false_statement`
- `truth_slots`
- `evidences`
- `contradictions`

## suspect_profile

### default_personality
- Stores the case file's default HEXACO values
- Useful as UI starting values or designer reference values
- The live interrogation engine uses the player's `selected_personality`

### mental_state
- Stores the starting PAD values for the interrogation
- `pleasure`
- `arousal`
- `dominance`

## evidences

Each evidence item includes:
- `id`
- `name`
- `description`
- `aliases`

`aliases` are used for question analysis and evidence mention matching.

## contradictions

Each contradiction item includes:
- `id`
- `description`
- `related_evidence`
- `slot`
- `truth_value`
- `contradiction_type`

These rows are the basis for hard contradiction scoring.

## Personality Flow

The case file stores `default_personality` as reference data.
Before interrogation starts, Unreal sends all 6 HEXACO values to the server.
The server stores them as `selected_personality` and uses that for live interrogation calculations.

HEXACO trait keys:
- `honesty_humility`
- `emotionality`
- `extraversion`
- `agreeableness`
- `conscientiousness`
- `openness_to_experience`

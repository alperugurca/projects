# LUCKY SLOTS

Demo slot machine game built with Godot Engine 4.7. For entertainment only - virtual currency, no real money.

## Features

- 3-reel slot machine with classic Vegas theme (gold/red)
- 8 normal symbols + Wild + Scatter
- Wild symbol (star) substitutes any normal symbol
- 3 Scatter symbols trigger 10 Free Spins (+5 extra on retrigger)
- Save system (balance, total spins, biggest win) via ConfigFile
- Daily bonus (+500) when out of credits
- Procedural sound effects + background music
- Mute toggle
- Mobile-optimized (portrait 720x1280)

## Symbols & Payouts (3 of a kind)

| Symbol | Multiplier |
|--------|------------|
| 7 | 50x |
| diamond | 30x |
| bell | 20x |
| BAR | 15x |
| grape | 10x |
| orange | 8x |
| lemon | 5x |
| cherry | 3x |
| WILD (star) | 50x (3 wilds) |
| SCATTER | triggers free spins |

- 2 matching symbols pay 20% of the 3-match payout (Wild substitutes).
- 3 Scatter = 10 free spins (no bet deducted). Retrigger = +5 spins.

## Requirements

- [Godot Engine 4.7+](https://godotengine.org/download)

## How to Run

1. Open Godot Engine
2. Import -> select `project.godot`
3. Press F5 (or Play button)

## Project Structure

```
SlotMachineGame/
├── project.godot
├── icon.svg
├── export_presets.cfg
├── scenes/
│   └── main.tscn          # Main scene
├── scripts/
│   ├── slot_machine.gd    # Game orchestrator
│   ├── reel.gd            # Reel animation
│   ├── symbol_set.gd      # Symbol data + registry
│   ├── save_manager.gd    # Autoload: save/load
│   └── audio_manager.gd   # Autoload: procedural sound
├── assets/
│   └── symbols/           # 10 SVG symbol textures
└── README.md
```

## Autoloads

- `SaveManager` - `res://scripts/save_manager.gd`
- `AudioManager` - `res://scripts/audio_manager.gd`

## Save Data

Stored at `user://slot_save.cfg` (per-platform user data dir):
- balance
- total_spins
- biggest_win
- free_spins_remaining

## Legal Notice

For educational and entertainment purposes only. No real money gambling. Virtual currency has no real-world value.

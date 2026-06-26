# QuitTheOdds

QuitTheOdds is a small Android app that turns a slot-machine simulation into a reality check about gambling odds. It lets users spin, lose virtual money, view a loss report, and see short facts about casino math, near-miss effects, and gambling risk.

## Features

- Weighted slot-machine simulation with intentionally unfavorable odds
- Live wallet, spin, win, and loss tracking
- Reality-check popups with gambling facts
- Loss report screen with simple financial stats
- Built with Kotlin, Jetpack Compose, Material 3, and Navigation Compose

## Build

Open the project in Android Studio, sync Gradle, then run the `app` configuration.

Or build from the terminal:

```powershell
.\gradlew.bat :app:assembleDebug
```

The debug APK is generated at:

```text
app/build/outputs/apk/debug/QuitTheOdds-debug.apk
```

## Purpose

This project is a prototype for showing how gambling can feel playable while still being mathematically designed to drain the player over time.

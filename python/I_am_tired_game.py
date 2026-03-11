import random


ROOMS = [
    {
        "name": "Quiet Cave",
        "text": "A sleepy cave hums softly. Three doors wait.",
        "good": 2,
    },
    {
        "name": "Snack Hall",
        "text": "A hallway smells faintly of toast and improbable victory.",
        "good": 1,
    },
    {
        "name": "Cloud Deck",
        "text": "You step onto a platform floating above a pillow-shaped cloud.",
        "good": 3,
    },
]

TREASURES = [
    "a warm cup of tea",
    "an extra hour of sleep",
    "a suspiciously perfect blanket",
    "a tiny medal labeled 'still trying'",
    "a pocket-sized sunset",
]


def ask_choice() -> int:
    while True:
        choice = input("Pick a door (1-3): ").strip()
        if choice in {"1", "2", "3"}:
            return int(choice)
        print("That is not a door. Tired rules still require numbers.")


def play_round(score: int) -> int:
    room = random.choice(ROOMS)
    print(f"\n{room['name']}")
    print(room["text"])
    choice = ask_choice()

    if choice == room["good"]:
        treasure = random.choice(TREASURES)
        print(f"You found {treasure}. Excellent work for a low-battery human.")
        return score + 1

    print("A duck wearing boots judges you briefly, then lets you continue.")
    return score


def main() -> None:
    print("Tiny Door Game")
    print("Find the cozy door three times to win.\n")

    score = 0
    rounds = 0

    while score < 3:
        rounds += 1
        score = play_round(score)
        print(f"Score: {score}/3")

    print(f"\nYou won in {rounds} rounds.")
    print("Prescription: drink water, stretch once, and be less rude to yourself.")


if __name__ == "__main__":
    main()

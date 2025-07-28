"""
Problem Definition: Smart assistant with gemini
    - Aim: To create a smart assistant that can help with tasks and answer questions, gemini api
    - Human natural language understand
    - Our maintenance system will provide us with features such as a calendar and exercises, enabling us to extract summary information by comparing notes and activities.

Model Definition: Gemini
    - Gemini 2.0 flash

Api Definition: https://aistudio.google.com/apikey

Plan
    - Assistant: Gemini chatbot
    - database: sqlite database, notes and activities
    - main: all together


Install lib

Import Lib
"""
# assistant.py file take 
from assistant import get_gemini_response, detect_intent

# database.py file process functions
from database import initialize_db, add_event, add_note, get_notes, get_events

# initialize the database
initialize_db()
print("Welcome")
print("Commands: add a note | add an event | view notes | view events | chat | exit")

# infinite loop for user input

while True:
    command = input("Enter your command: ").strip().lower() # take command

    if command == "add a note":
        content = input("Enter your note: ")
        add_note(content)
        print("Note added successfully")
    elif command == "add an event":
        event = input("Enter your event: ")
        date = input("Enter the event date: ")
        add_event(event, date)
        print("Event added successfully")
    elif command == "view notes":
        notes = get_notes() # get notes from database
        if notes:
            print("Saved Notes:")
            for content, created_at in notes:
                print(f"- [{created_at} - {content}]")
        else:
            print("No notes found")
    elif command == "view events":
        events = get_events() # get events from database
        if events:
            print("Saved Events:")
            for event, event_date in events:
                print(f"- [{event_date} - {event}]")
        else:
            print("No events found")

    elif command == "chat":
        message = input("User Enter your prompt: ").strip()
        intent = detect_intent(message) # detect intent from message

        if intent == "notes_summary":
            notes = get_notes()
            if not notes:
                print("No notes found")
                continue
            
            all_notes_text = "\n".join([f"{note[0]}" for note in notes]) # merge all notes
            prompt = f"Summarize the following notes: \n\n{all_notes_text}"
            summary = get_gemini_response(prompt) # get summary from gemini
            print(f"Notes Summary:\n{summary}")

        elif intent == "events_summary":
            events = get_events() # get events from database
            if not events:
                print("No events found")
                continue
            
            all_events_text = "\n".join([f"{e[0]}: {e[1]}" for e in events]) # merge all events
            prompt = f"According to user request summarize the following events: \n\n{all_events_text}\n\nUser request: {message}"
            summary = get_gemini_response(prompt) # get summary from gemini
            print(f"Events Summary:\n{summary}")
        else:
            reply = get_gemini_response(message) # get reply from gemini
            print(f"Assistant: {reply}")

    elif command == "exit":
        print("Goodbye!")
        break
    else:
        print("Invalid command")
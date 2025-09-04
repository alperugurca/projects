import sqlite3 #import sqlite3 library
import os #import os library

# database path data/assistant.db
DB_PATH = os.path.join('data', 'assistant.db')

def initialize_db():
    # check if the database file exists
    os.makedirs('data', exist_ok=True)

    # connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # if notes table does not exist, create it
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, -- auto create primary key
                        content TEXT NOT NULL, -- note content
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- note creation time
                    )
                """)


    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calendar (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, -- auto create primary key
                        event TEXT NOT NULL, -- event description,
                        event_date TEXT NOT NULL -- event date
                    )
                """)

    # commit the changes
    conn.commit()

    # close the connection
    conn.close()


# add a note to the database
def add_note(content):
    # connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # add content to the notes table
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))

    # commit the changes
    conn.commit()

    # close the connection
    conn.close()

# new function to add an event to the calendar
def add_event(event, event_date):
    # connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # add event to the calendar table
    cursor.execute("INSERT INTO calendar (event, event_date) VALUES (?, ?)", (event, event_date))

    # commit the changes
    conn.commit()

    # close the connection
    conn.close()

# all notes from the database, ordered by creation time
def get_notes():
    # connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # get all notes from the notes table
    cursor.execute("SELECT content, created_at FROM notes ORDER BY created_at DESC")

    # result list
    notes = cursor.fetchall()

    # close the connection
    conn.close()
    return notes    

# all events from the database, ordered by event date
def get_events():
    # connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # get all events from the calendar table
    cursor.execute("SELECT event, event_date FROM calendar ORDER BY event_date")

    # catch results
    events = cursor.fetchall()
    
    conn.close()
    return events

if __name__ == "__main__":
    initialize_db()
    add_note("This is a test note")
    add_event("This is a test event", "2025-01-01")

    print(f"Notes: {get_notes()}")
    print(f"Events: {get_events()}")
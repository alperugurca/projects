import { useEffect, useState } from "react";
import {
  collection,
  onSnapshot,
  addDoc,
  updateDoc,
  deleteDoc,
  doc,
  serverTimestamp,
  query,
  orderBy,
} from "firebase/firestore";
import { db } from "../lib/firebase";
import { KanbanColumn } from "./components/KanbanColumn";
import type { Note, Column } from "./types";

const COLUMNS: Column[] = [
  { id: "todo", title: "To Do", color: "bg-gray-500", noteColor: "bg-gray-200" },
  { id: "inprogress", title: "In Progress", color: "bg-yellow-500", noteColor: "bg-yellow-200" },
  { id: "done", title: "Done", color: "bg-green-500", noteColor: "bg-green-200" },
];

export default function App() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);

  // Listen for real-time changes from Firestore
  useEffect(() => {
    const q = query(collection(db, "notes"), orderBy("createdAt", "asc"));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const fetched: Note[] = snapshot.docs.map((d) => ({
        id: d.id,
        text: d.data().text,
        columnId: d.data().columnId,
      }));
      setNotes(fetched);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const addNote = async (columnId: string) => {
    await addDoc(collection(db, "notes"), {
      text: "",
      columnId,
      createdAt: serverTimestamp(),
    });
  };

  const moveNote = async (noteId: string, targetColumnId: string) => {
    await updateDoc(doc(db, "notes", noteId), { columnId: targetColumnId });
  };

  const editNote = async (noteId: string, text: string) => {
    await updateDoc(doc(db, "notes", noteId), { text });
  };

  const deleteNote = async (noteId: string) => {
    await deleteDoc(doc(db, "notes", noteId));
  };

  if (loading) {
    return (
      <div className="size-full bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-gray-500 text-lg font-medium animate-pulse">Loading board...</div>
      </div>
    );
  }

  return (
    <div className="size-full bg-gradient-to-br from-blue-50 to-indigo-100 p-8 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-800 mb-8 text-center">
          Kanban Board
        </h1>
        <div className="flex gap-6">
          {COLUMNS.map((column) => (
            <KanbanColumn
              key={column.id}
              title={column.title}
              columnId={column.id}
              notes={notes.filter((note) => note.columnId === column.id)}
              columns={COLUMNS}
              onAddNote={() => addNote(column.id)}
              onMoveNoteTo={moveNote}
              onEditNote={editNote}
              onDeleteNote={deleteNote}
              color={column.color}
              noteColor={column.noteColor}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

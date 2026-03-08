import { StickyNote } from "./StickyNote";
import type { Note, Column } from "../types";

interface KanbanColumnProps {
  title: string;
  columnId: string;
  notes: Note[];
  columns: Column[];
  onAddNote: () => void;
  onMoveNoteTo: (noteId: string, columnId: string) => void;
  onEditNote: (noteId: string, text: string) => void;
  onDeleteNote: (noteId: string) => void;
  color: string;
  noteColor: string;
}

export function KanbanColumn({
  title,
  columnId,
  notes,
  columns,
  onAddNote,
  onMoveNoteTo,
  onEditNote,
  onDeleteNote,
  color,
  noteColor,
}: KanbanColumnProps) {
  return (
    <div className="flex-1 min-w-[300px] bg-gray-100 rounded-xl p-6 shadow-lg">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${color}`}></span>
          {title}
          <span className="text-sm text-gray-500 font-normal">({notes.length})</span>
        </h2>
        <button
          onClick={onAddNote}
          className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium shadow-md hover:shadow-lg"
        >
          + Add Note
        </button>
      </div>
      <div className="space-y-4 min-h-[400px]">
        {notes.map((note) => (
          <StickyNote
            key={note.id}
            id={note.id}
            text={note.text}
            currentColumn={columnId}
            columns={columns}
            backgroundColor={noteColor}
            onMoveTo={(targetColumnId) => onMoveNoteTo(note.id, targetColumnId)}
            onEdit={(text) => onEditNote(note.id, text)}
            onDelete={() => onDeleteNote(note.id)}
          />
        ))}
      </div>
    </div>
  );
}

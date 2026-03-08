import type { Column } from "../types";

interface StickyNoteProps {
  id: string;
  text: string;
  currentColumn: string;
  columns: Column[];
  backgroundColor: string;
  onMoveTo: (columnId: string) => void;
  onEdit: (text: string) => void;
  onDelete: () => void;
}

export function StickyNote({ id, text, currentColumn, columns, backgroundColor, onMoveTo, onEdit, onDelete }: StickyNoteProps) {
  const availableColumns = columns.filter((col) => col.id !== currentColumn);

  return (
    <div className={`${backgroundColor} p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow relative group`}>
      <textarea
        value={text}
        onChange={(e) => onEdit(e.target.value)}
        className="w-full bg-transparent resize-none outline-none text-gray-800 min-h-[80px] text-base leading-relaxed"
        placeholder="Type your note..."
      />
      <div className="flex gap-2 mt-2 justify-end opacity-0 group-hover:opacity-100 transition-opacity flex-wrap">
        {availableColumns.map((column) => (
          <button
            key={column.id}
            onClick={() => onMoveTo(column.id)}
            className="text-xs bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 transition-colors"
          >
            → {column.title}
          </button>
        ))}
        <button
          onClick={onDelete}
          className="text-xs bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600 transition-colors"
        >
          Delete
        </button>
      </div>
    </div>
  );
}

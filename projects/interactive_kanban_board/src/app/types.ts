export interface Note {
  id: string;
  text: string;
  columnId: string;
}

export interface Column {
  id: string;
  title: string;
  color: string;
  noteColor: string;
}

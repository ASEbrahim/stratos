import React from 'react';
import { SuggestionChips } from './SuggestionChips';
import { DirectorNoteBar } from './DirectorNoteBar';
import { ChatInput } from './ChatInput';
import { Suggestion } from '../../lib/types';

interface ChatInputSectionProps {
  suggestions: Suggestion[];
  directorNote: string;
  lastUsedNote: string;
  isStreaming: boolean;
  accentColor: string | undefined;
  onSend: (text: string) => void;
  onNoteChange: (note: string) => void;
  onReuseNote: () => void;
  onClearNote: () => void;
}

export const ChatInputSection = React.memo(function ChatInputSection({
  suggestions,
  directorNote,
  lastUsedNote,
  isStreaming,
  accentColor,
  onSend,
  onNoteChange,
  onReuseNote,
  onClearNote,
}: ChatInputSectionProps) {
  return (
    <>
      <SuggestionChips suggestions={suggestions} onSelect={onSend} accentColor={accentColor} />

      <DirectorNoteBar
        note={directorNote}
        lastUsedNote={lastUsedNote}
        onNoteChange={onNoteChange}
        onReuse={onReuseNote}
        onClear={onClearNote}
        accentColor={accentColor}
      />

      <ChatInput onSend={onSend} disabled={isStreaming} accentColor={accentColor} />
    </>
  );
});

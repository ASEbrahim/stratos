import { USE_MOCKS } from '../constants/config';
import { API_BASE } from '../constants/config';
import { getToken } from './api';
import { ChatMessage, CharacterCard, Suggestion } from './types';
import { MOCK_SUGGESTIONS, generateId } from './mock';

export async function streamMessage(
  sessionId: string, message: string, persona: 'roleplay' | 'gaming',
  characterCard: CharacterCard | null,
  onChunk: (text: string) => void, onDone: () => void,
): Promise<void> {
  if (USE_MOCKS) {
    await mockStream(characterCard, onChunk, onDone);
    return;
  }
  const token = await getToken();
  const response = await fetch(`${API_BASE}/api/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ message, persona, session_id: sessionId, character_card: characterCard }),
  });
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  if (!reader) { onDone(); return; }
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) { onDone(); break; }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.content) onChunk(data.content);
          if (data.done) onDone();
        } catch { /* partial */ }
      }
    }
  }
}

async function mockStream(
  character: CharacterCard | null, onChunk: (text: string) => void, onDone: () => void,
): Promise<void> {
  const name = character?.name ?? 'Character';
  const responses = [
    `*${name} considers your words carefully.*\n\n"An interesting proposition," *they say.* "You have my attention — for now."`,
    `*A faint smile plays at ${name}'s lips.*\n\n"You remind me of someone. Someone who asked too many questions."`,
    `*${name} is quiet for a long moment.*\n\n"I've heard prettier words," *they finally say.* "Walk with me. I want to show you something."`,
  ];
  const response = responses[Math.floor(Math.random() * responses.length)];
  await new Promise(r => setTimeout(r, 600));
  const words = response.split(' ');
  for (let i = 0; i < words.length; i++) {
    await new Promise(r => setTimeout(r, 30 + Math.random() * 40));
    onChunk(words[i] + (i < words.length - 1 ? ' ' : ''));
  }
  onDone();
}

export async function getSuggestions(
  sessionId: string, persona: 'roleplay' | 'gaming', lastMessage: string,
): Promise<Suggestion[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 500));
    return [...MOCK_SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 3);
  }
  const token = await getToken();
  const response = await fetch(`${API_BASE}/api/agent/suggest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ session_id: sessionId, persona, last_message: lastMessage }),
  });
  return response.json();
}

export function createMessageId(): string {
  return generateId();
}

/**
 * Shared SSE stream parser with timeout, abort, and cleanup.
 * Used by streamMessage() and streamRegenerate() in chat.ts.
 */

import { Config } from '../constants/config';
import { reportError } from './utils';

const TOTAL_TIMEOUT_MS = Config.STREAM_TIMEOUT_MS;
const STALL_TIMEOUT_MS = Config.STALL_TIMEOUT_MS;

export interface SSECallbacks {
  onToken: (text: string) => void;
  onComplete: () => void;
  onError: (error: Error) => void;
}

/**
 * Parse an SSE response stream. Handles both ReadableStream and text fallback.
 * Accepts an optional AbortSignal to support external cancellation.
 */
export async function parseSSEStream(
  response: Response,
  { onToken, onComplete, onError }: SSECallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const reader = response.body?.getReader();

  // Total timeout — abort if entire stream takes too long
  const totalTimer = setTimeout(() => {
    reader?.cancel();
    onError(new Error('Response timed out'));
  }, TOTAL_TIMEOUT_MS);

  // Stall timer — abort if no data arrives for too long
  let stallTimer: ReturnType<typeof setTimeout> | null = null;
  const resetStallTimer = () => {
    if (stallTimer) clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      reader?.cancel();
      onError(new Error('Connection stalled — no data received'));
    }, STALL_TIMEOUT_MS);
  };

  // Listen for external abort
  const abortHandler = () => { reader?.cancel(); };
  signal?.addEventListener('abort', abortHandler);

  const cleanup = () => {
    clearTimeout(totalTimer);
    if (stallTimer) clearTimeout(stallTimer);
    signal?.removeEventListener('abort', abortHandler);
  };

  try {
    if (reader) {
      const decoder = new TextDecoder();
      let buffer = '';
      resetStallTimer();
      while (true) {
        if (signal?.aborted) { reader.cancel(); break; }
        const { done, value } = await reader.read();
        if (done) { onComplete(); break; }
        resetStallTimer();
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.content || data.token) onToken(data.content || data.token);
              if (data.done) { onComplete(); return; }
            } catch { /* partial JSON — skip */ }
          }
        }
      }
    } else {
      // Fallback: read full response as text and parse SSE lines
      const text = await response.text();
      const lines = text.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.content || data.token) onToken(data.content || data.token);
          } catch { /* partial */ }
        }
      }
      onComplete();
    }
  } catch (err) {
    if (signal?.aborted) return; // Expected — don't report
    onError(err instanceof Error ? err : new Error(String(err)));
  } finally {
    cleanup();
    // Ensure reader is always cancelled
    try { reader?.cancel(); } catch (err) { reportError('parseSSEStream:readerCancel', err); }
  }
}

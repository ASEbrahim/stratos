/**
 * RP Expansion API functions.
 *
 * Edit, swipe, branch, feedback, director's note, image generation.
 * These call the backend endpoints built in Sprints 2-6.
 */

import { apiFetch, getToken } from './api';
import { API_BASE, USE_MOCKS } from '../constants/config';

// ── Swipe: regenerate last assistant message ──
export async function regenerateMessage(
  sessionId: string, branchId: string = 'main', cardId?: string
) {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 300));
    return { done: true, swipe_group_id: `swipe-mock-${Date.now()}`, swipe_count: 2, message_id: 0 };
  }
  return apiFetch<{
    done: boolean;
    swipe_group_id: string;
    swipe_count: number;
    message_id: number;
  }>('/api/rp/regenerate', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, branch_id: branchId, character_card_id: cardId }),
  });
}

// ── Select a swipe alternative ──
export async function selectSwipe(messageId: number, swipeGroupId: string) {
  if (USE_MOCKS) return { ok: true, selected: messageId };
  return apiFetch<{ ok: boolean; selected: number }>('/api/rp/swipe', {
    method: 'POST',
    body: JSON.stringify({ message_id: messageId, swipe_group_id: swipeGroupId }),
  });
}

// ── Edit AI message (DPO pair) ──
export async function editMessage(
  messageId: number, editedContent: string, editReason?: string
) {
  if (USE_MOCKS) return { ok: true, category: 'refinement', reason: editReason };
  return apiFetch<{ ok: boolean; category: string; reason?: string }>('/api/rp/edit', {
    method: 'POST',
    body: JSON.stringify({
      message_id: messageId,
      edited_content: editedContent,
      edit_reason: editReason,
    }),
  });
}

// ── Branch from earlier message (returns SSE stream) ──
export async function createBranch(
  sessionId: string, branchId: string, atTurn: number,
  content: string, cardId?: string, persona: string = 'roleplay'
) {
  if (USE_MOCKS) {
    return { ok: true, branch_id: `branch-mock-${Date.now()}`, from_branch: branchId };
  }
  const token = await getToken();
  const response = await fetch(`${API_BASE}/api/rp/branch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      session_id: sessionId, branch_id: branchId,
      at_turn: atTurn, content, character_card_id: cardId, persona,
    }),
  });
  // This is an SSE endpoint — caller handles streaming
  return response;
}

// ── Director's note ──
export async function setDirectorNote(sessionId: string, note: string) {
  if (USE_MOCKS) return { ok: true, note };
  return apiFetch<{ ok: boolean; note: string }>('/api/rp/director-note', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, note }),
  });
}

// ── Feedback (thumbs up/down) ──
export async function sendFeedback(
  messageId: number, type: 'thumbs_up' | 'thumbs_down'
) {
  if (USE_MOCKS) return { ok: true };
  return apiFetch<{ ok: boolean }>('/api/rp/feedback', {
    method: 'POST',
    body: JSON.stringify({ message_id: messageId, feedback_type: type }),
  });
}

// ── Get conversation history with branches ──
export async function getHistory(sessionId: string, branchId: string = 'main') {
  if (USE_MOCKS) return { session_id: sessionId, branch_id: branchId, messages: [], branches: [] };
  return apiFetch<{
    session_id: string;
    branch_id: string;
    messages: any[];
    branches: any[];
  }>(`/api/rp/history/${sessionId}?branch=${branchId}`);
}

// ── List branches ──
export async function getBranches(sessionId: string) {
  if (USE_MOCKS) return { branches: [{ id: 'main', parent_branch_id: null, turn_count: 0, is_active: true }] };
  return apiFetch<{ branches: any[] }>(`/api/rp/branches/${sessionId}`);
}

// ── Free-form text-to-image (CHROMA — single model, SFW+NSFW) ──
export async function generateImage(params: {
  prompt: string;
  width?: number;
  height?: number;
  seed?: number;
  steps?: number;
  negative_prompt?: string;
}): Promise<{ success: boolean; image_id?: string; error?: string }> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 2000));
    return { success: true, image_id: `mock-img-${Date.now()}` };
  }
  try {
    return await apiFetch<{ success: boolean; image_id?: string; error?: string }>(
      '/api/image/generate',
      { method: 'POST', body: JSON.stringify(params) }
    );
  } catch {
    return { success: false, error: 'Image generation unavailable — ComfyUI is not running.' };
  }
}

// ── Character portrait generation ──
export async function generateCharacterPortrait(params: {
  character_name: string;
  physical_description: string;
  scenario?: string;
  style?: 'anime' | 'realistic' | 'illustration';
  nsfw?: boolean;
  character_card_id?: string;
}): Promise<{ success: boolean; image_id?: string; error?: string }> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 2000));
    return { success: true, image_id: `mock-portrait-${Date.now()}` };
  }
  try {
    return await apiFetch<{ success: boolean; image_id?: string; error?: string }>(
      '/api/image/character-portrait',
      { method: 'POST', body: JSON.stringify(params) }
    );
  } catch {
    return { success: false, error: 'Image generation unavailable — ComfyUI is not running.' };
  }
}

// ── Get image URL ──
export function getImageUrl(imageId: string): string {
  return `${API_BASE}/api/image/${imageId}`;
}

// ── Image gallery ──
export async function getImageGallery() {
  if (USE_MOCKS) return { images: [] };
  return apiFetch<{ images: any[] }>('/api/image/gallery');
}

// ── Delete image ──
export async function deleteImage(imageId: string) {
  if (USE_MOCKS) return { ok: true, deleted: imageId };
  return apiFetch<{ ok: boolean; deleted: string }>(`/api/image/${imageId}`, {
    method: 'DELETE',
  });
}

// ── Privacy opt-in ──
export async function setTrainingOptIn(optedIn: boolean) {
  if (USE_MOCKS) return { ok: true };
  return apiFetch<{ ok: boolean }>('/api/rp/opt-in', {
    method: 'POST',
    body: JSON.stringify({ opted_in: optedIn }),
  });
}

// ── Rate a character card ──
export async function rateCard(cardId: string, rating: number) {
  if (USE_MOCKS) return { ok: true };
  return apiFetch<{ ok: boolean }>(`/api/cards/${cardId}/rate`, {
    method: 'POST',
    body: JSON.stringify({ rating }),
  });
}

// ── Publish a card ──
export async function publishCard(cardId: string) {
  if (USE_MOCKS) return { ok: true };
  return apiFetch<{ ok: boolean }>(`/api/cards/${cardId}/publish`, { method: 'POST' });
}

// ── Save someone else's card ──
export async function saveCard(cardId: string) {
  if (USE_MOCKS) return { ok: true, card_id: `saved-${cardId}` };
  return apiFetch<{ ok: boolean; card_id: string }>(`/api/cards/${cardId}/save`, { method: 'POST' });
}

import { USE_MOCKS } from '../constants/config';
import { API_BASE } from '../constants/config';
import { getToken, getDeviceId } from './api';
import { ChatMessage, CharacterCard, Suggestion } from './types';
import { MOCK_SUGGESTIONS, generateId } from './mock';
import { parseSSEStream } from './sse';
import { reportError } from './utils';

// Format + length hint — appended for cards without speech_pattern
const FORMAT_HINT = '[OOC: Use *asterisks* for actions and "quotes" for speech. Keep response length proportional to input — short input = short reply. Vary your response structure — don\'t always start with *action*.]';

// Active AbortController for current stream — exposed for cancellation
let _activeAbort: AbortController | null = null;

/** Cancel the currently active stream (if any). */
export function cancelStream(): void {
  _activeAbort?.abort();
  _activeAbort = null;
}

export async function streamMessage(
  sessionId: string, message: string, persona: 'roleplay' | 'gaming',
  characterCard: CharacterCard | null,
  onChunk: (text: string) => void, onDone: (doneData?: Record<string, unknown>) => void,
  directorNote?: string, sessionContext?: string,
): Promise<void> {
  if (USE_MOCKS) {
    await mockStream(message, persona, characterCard, onChunk, onDone);
    return;
  }

  const abort = new AbortController();
  _activeAbort = abort;

  try {
    const token = await getToken();
    const deviceId = await getDeviceId();
    const needsHint = persona === 'roleplay' && characterCard && !characterCard.speech_pattern?.trim();
    const content = needsHint ? `${message}\n\n${FORMAT_HINT}` : message;
    const response = await fetch(`${API_BASE}/api/rp/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Device-Id': deviceId,
        ...(token ? { 'X-Auth-Token': token } : {}),
      },
      body: JSON.stringify({
        content,
        persona,
        session_id: sessionId,
        character_card_id: characterCard?.id || undefined,
        first_message: characterCard?.first_message || undefined,
        ...(directorNote ? { director_note: directorNote } : {}),
        ...(sessionContext ? { session_context: sessionContext } : {}),
      }),
      signal: abort.signal,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      const errorMsg = `API error ${response.status}: ${body}`;
      reportError('streamMessage', new Error(errorMsg));
      onChunk(`\n\n*[Connection error — server returned ${response.status}. Try again.]*`);
      onDone();
      return;
    }

    await parseSSEStream(response, {
      onToken: onChunk,
      onComplete: (doneData) => onDone(doneData),
      onError: (err) => {
        reportError('streamMessage:sse', err);
        onChunk(`\n\n*[${err.message}]*`);
        onDone();
      },
    }, abort.signal);
  } catch (err) {
    if (abort.signal.aborted) { onDone(); return; }
    reportError('streamMessage', err);
    onChunk('\n\n*[Connection lost — check your server and try again.]*');
    onDone();
  } finally {
    if (_activeAbort === abort) _activeAbort = null;
  }
}

/**
 * Stream a regenerated (swipe) response from /api/rp/regenerate.
 * Does NOT insert a new user message — reuses existing conversation.
 */
export async function streamRegenerate(
  sessionId: string, persona: 'roleplay' | 'gaming',
  characterCard: CharacterCard | null,
  onChunk: (text: string) => void, onDone: () => void,
): Promise<void> {
  if (USE_MOCKS) {
    await mockStream('regenerate', persona, characterCard, onChunk, onDone);
    return;
  }

  const abort = new AbortController();
  _activeAbort = abort;

  try {
    const token = await getToken();
    const deviceId = await getDeviceId();
    const response = await fetch(`${API_BASE}/api/rp/regenerate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Device-Id': deviceId,
        ...(token ? { 'X-Auth-Token': token } : {}),
      },
      body: JSON.stringify({
        session_id: sessionId,
        branch_id: 'main',
        character_card_id: characterCard?.id || undefined,
        persona,
      }),
      signal: abort.signal,
    });

    if (!response.ok) {
      reportError('streamRegenerate', new Error(`HTTP ${response.status}`));
      onDone();
      return;
    }

    await parseSSEStream(response, {
      onToken: onChunk,
      onComplete: onDone,
      onError: (err) => {
        reportError('streamRegenerate:sse', err);
        onDone();
      },
    }, abort.signal);
  } catch (err) {
    if (abort.signal.aborted) { onDone(); return; }
    reportError('streamRegenerate', err);
    onDone();
  } finally {
    if (_activeAbort === abort) _activeAbort = null;
  }
}

// ─── Context-aware mock responses ───
// Reads user message + character card to produce semi-coherent replies
async function mockStream(
  userMessage: string,
  persona: 'roleplay' | 'gaming',
  character: CharacterCard | null,
  onChunk: (text: string) => void,
  onDone: () => void,
): Promise<void> {
  const name = character?.name ?? 'Character';
  const msg = userMessage.toLowerCase();

  let response: string;

  if (persona === 'gaming') {
    response = pickGamingResponse(msg);
  } else {
    response = pickRPResponse(msg, name, character);
  }

  await new Promise(r => setTimeout(r, 500));
  const words = response.split(' ');
  for (let i = 0; i < words.length; i++) {
    await new Promise(r => setTimeout(r, 25 + Math.random() * 35));
    onChunk(words[i] + (i < words.length - 1 ? ' ' : ''));
  }
  onDone();
}

function pickRPResponse(msg: string, name: string, char: CharacterCard | null): string {
  if (msg.match(/\b(hi|hello|hey|greetings|good morning|good evening)\b/)) {
    return [
      `*${name} inclines their head slightly, a measured gesture that reveals nothing and promises everything.*\n\n"You have a way of appearing at interesting moments," *they say, a trace of warmth beneath the careful tone.* "I wasn't expecting company — but I find I don't mind it."`,
      `*${name}'s gaze lifts to meet yours, and for a moment something unguarded crosses their features before it's smoothed away.*\n\n"Hello." *The word is simple, but the way they say it carries the weight of someone who hasn't said it to anyone in a while.* "Come in. Or don't. The choice, as always, is yours."`,
    ][Math.floor(Math.random() * 2)];
  }

  if (msg.includes('?') || msg.match(/\b(who|what|where|when|why|how|tell me|explain)\b/)) {
    return [
      `*${name} pauses, weighing how much truth to offer and how much to keep behind their teeth.*\n\n"That's not a simple question," *they say slowly.* "But I suppose you already knew that, or you wouldn't have asked."\n\n*They lean back, eyes distant for a moment.* "What I can tell you is this — things are rarely what they seem here. Least of all me."`,
      `*Something shifts behind ${name}'s expression — a flicker of the person beneath the performance.*\n\n"You want to know?" *A pause. Then, quieter:* "Most people don't actually want the answer. They want the version that lets them sleep at night."\n\n*They study you, as if deciding which version you can handle.* "But you... you might be different."`,
      `*${name} tilts their head, considering you with renewed interest.*\n\n"Curious," *they murmur.* "No one has asked me that in a very long time."\n\n*Their fingers trace an absent pattern on the surface beside them.* "I'll answer. But you might wish I hadn't."`,
    ][Math.floor(Math.random() * 3)];
  }

  if (msg.includes('*') || msg.match(/\b(walk|move|go|take|grab|draw|reach|look|turn|sit|stand|run|fight|attack)\b/)) {
    return [
      `*${name}'s eyes track your movement with the precision of someone who has learned the hard way never to let their guard down completely.*\n\n*For a beat, neither of you moves. The air between you hums with something unspoken — tension, maybe. Or recognition.*\n\n"Interesting," *they murmur, more to themselves than to you.* "Most people hesitate. You didn't."`,
      `*Your movement draws ${name}'s attention like a spark in darkness. They go very still — the kind of stillness that comes before a decision.*\n\n*Then, slowly, they respond in kind. Matching your energy, not escalating it. Testing.*\n\n"So that's how it is," *they say, and there's a ghost of approval in their voice.* "Good. I was getting tired of pretending."`,
    ][Math.floor(Math.random() * 2)];
  }

  if (msg.match(/\b(feel|love|hate|miss|afraid|scared|angry|sad|happy|sorry|forgive|trust)\b/)) {
    return `*The word hangs in the air between you, and you watch something move across ${name}'s face — not quite pain, not quite relief, but the complicated space between them.*\n\n*When they finally speak, their voice is rougher than before.* "You say that like it's easy. Like the word doesn't cost anything."\n\n*A breath. Their hands find something to hold onto — an anchor against whatever tide is pulling at them.* "It does. Trust me, it does."`;
  }

  const speechHint = char?.speech_pattern ? ` Their voice carries ${char.speech_pattern.split('.')[0].toLowerCase()}.` : '';
  return [
    `*${name} is quiet for a long moment — not the awkward silence of someone with nothing to say, but the deliberate silence of someone choosing their words like a surgeon choosing instruments.*\n\n"I've been thinking about what you said," *they admit.${speechHint}* "And I'm not sure you realize what you've set in motion."\n\n*They look at you — really look at you — and something behind their careful composure shifts.* "But we're here now. So let's see where this goes."`,
    `*Something in your words seems to reach ${name} in a place they keep carefully guarded. Their expression doesn't change — they're too practiced for that — but their breathing does.*\n\n"You know," *they say, with the careful lightness of someone defusing a bomb,* "there was a time I would have walked away from a conversation like this."\n\n*A pause. The faintest smile.* "Clearly, I'm getting worse at self-preservation."`,
    `*${name} watches you with an expression you're beginning to recognize — the one that means they're weighing something important against something personal.*\n\n"I want to be honest with you," *they say, and the simplicity of it is almost startling.* "That's... not something I say often. Or easily."\n\n*They hold your gaze.* "But here we are."`,
  ][Math.floor(Math.random() * 3)];
}

function pickGamingResponse(msg: string): string {
  if (msg.match(/\b1\b/) || msg.match(/reinforce|defend|hold|fight/)) {
    return `You rally your forces to the eastern wall. Soldiers scramble into position, reinforcing the crumbling stonework with timber and shields.\n\nThe next volley of boulders strikes — the wall shudders but holds. Your lieutenant shouts over the chaos: "It's working! But they're moving siege towers into position!"\n\nFrom the ramparts, you spot movement in the enemy camp. Reinforcements? Or something worse.\n\n1. Focus all archers on the siege towers\n2. Send scouts to investigate the enemy movement\n3. Prepare oil and fire for the siege towers\n4. Sound the retreat to the inner keep`;
  }
  if (msg.match(/\b2\b/) || msg.match(/evacuate|escape|tunnel|flee/)) {
    return `You order the evacuation. Civilians stream into the siege tunnels — families clutching what they can carry, children held close.\n\nThe tunnel is dark and narrow. Torchlight flickers against wet stone. Your rear guard reports: "Commander, the enemy has noticed the movement. Cavalry is riding for the southern exit."\n\nYou have maybe twenty minutes before they reach the tunnel mouth.\n\n1. Speed up the evacuation — leave supplies behind\n2. Collapse the tunnel behind the last group\n3. Set an ambush at the southern exit\n4. Split forces — half evacuate, half create a diversion`;
  }
  if (msg.match(/\b3\b/) || msg.match(/sortie|attack|strike|charge/)) {
    return `Under cover of darkness, you lead forty of your best fighters through the siege tunnels. The enemy camp sprawls before you — cook fires, supply wagons, sleeping soldiers.\n\nYour scout whispers: "The commander's tent is to the north. Supply depot is east. Siege engine workshop is south."\n\nYou have the element of surprise, but not for long.\n\n1. Strike the commander's tent — cut off the head\n2. Burn the supply depot — starve them out\n3. Destroy the siege engines — buy time for the walls\n4. Split into three teams and hit all targets at once`;
  }
  return `The situation evolves rapidly. Your forces hold their position, watching and waiting for your next command.\n\nA messenger arrives, out of breath: "Reports from the western watchtower — dust clouds on the horizon. Could be reinforcements... ours or theirs. We won't know for an hour."\n\nYour advisors look to you.\n\n1. Send fast riders to identify the approaching force\n2. Prepare defenses assuming they're hostile\n3. Send a signal fire to any allied forces in the region\n4. Use this distraction to launch an offensive`;
}

export async function getSuggestions(
  sessionId: string, persona: 'roleplay' | 'gaming', lastMessage: string,
): Promise<Suggestion[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 500));
    if (persona === 'gaming') {
      return [
        { label: 'Reinforce defenses', prompt: '1. Reinforce the defenses and hold position' },
        { label: 'Scout ahead', prompt: '*sends scouts to investigate the surroundings*' },
        { label: 'Rally troops', prompt: '"Everyone, to me! We make our stand here."' },
      ];
    }
    return [...MOCK_SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 3);
  }
  const token = await getToken();
  const deviceId = await getDeviceId();
  try {
    const response = await fetch(`${API_BASE}/api/suggest-context`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Device-Id': deviceId,
        ...(token ? { 'X-Auth-Token': token } : {}),
      },
      body: JSON.stringify({ session_id: sessionId, persona, last_message: lastMessage }),
    });
    if (!response.ok) return [];
    const data = await response.json();
    return data.suggestions || data || [];
  } catch (err) {
    reportError('getSuggestions', err);
    return [];
  }
}

export function createMessageId(): string {
  return generateId();
}

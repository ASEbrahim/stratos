import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Share, Keyboard, Modal, TextInput, TouchableWithoutFeedback } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Pencil, RefreshCw } from 'lucide-react-native';
import { useThemedAlert } from '../../components/shared/ThemedAlert';
import { useChatStore } from '../../stores/chatStore';
import { SessionHeader } from '../../components/chat/SessionHeader';
import { MessageBubble, StreamingBubble } from '../../components/chat/MessageBubble';
import { TypingIndicator } from '../../components/chat/TypingIndicator';
import { SuggestionChips } from '../../components/chat/SuggestionChips';
import { ChatInput } from '../../components/chat/ChatInput';
import { FeedbackButtons } from '../../components/chat/FeedbackButtons';
import { SwipeIndicator } from '../../components/chat/SwipeIndicator';
import { DirectorNoteBar } from '../../components/chat/DirectorNoteBar';
import { EditSheet } from '../../components/chat/EditSheet';
import { BranchSelector } from '../../components/chat/BranchSelector';
import { TrainingOptInPopup, useTrainingOptInCheck } from '../../components/chat/TrainingOptIn';
import { ChevronDown } from 'lucide-react-native';
import { ChatMessage } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList>(null);
  const prevStreamRef = useRef(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [showSaved, setShowSaved] = useState(false);
  const tc = useThemeStore(s => s.colors);
  const { character, messages, suggestions, isStreaming, streamingContent, sendMessage, persistSession, startSession, clearSession, regenerateLastMessage, setSessionContext } = useChatStore();
  const accentColor = character ? getGenreColor(character.genre_tags?.[0] ?? 'default') : undefined;

  // ── RP Expansion state ──
  const [directorNote, setDirectorNote] = useState('');
  const [lastUsedNote, setLastUsedNote] = useState('');
  const [branches, setBranches] = useState<any[]>([{ id: 'main', parent_branch_id: null, turn_count: 0, is_active: true }]);
  const [currentBranch, setCurrentBranch] = useState('main');
  const [swipeCount, setSwipeCount] = useState(0);
  const [swipeIndex, setSwipeIndex] = useState(0);
  const [editTarget, setEditTarget] = useState<{ id: string; content: string } | null>(null);
  const [showContextModal, setShowContextModal] = useState(false);
  const [contextInput, setContextInput] = useState('');
  const [isRegenerating, setIsRegenerating] = useState(false);
  const { showOptIn, dismiss: dismissOptIn } = useTrainingOptInCheck();
  const { alert: showAlert, AlertComponent } = useThemedAlert();

  // Persist session when leaving
  useFocusEffect(useCallback(() => { return () => { persistSession().catch(() => {}); }; }, []));

  useEffect(() => {
    if (isStreaming && !prevStreamRef.current) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    if (!isStreaming && prevStreamRef.current) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      persistSession().catch(() => {});
      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 2000);
      // Track director's note usage
      if (directorNote) {
        setLastUsedNote(directorNote);
        setDirectorNote('');
      }
    }
    prevStreamRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    if (messages.length > 0) setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
  }, [messages.length, streamingContent]);

  useEffect(() => {
    const sub = Keyboard.addListener('keyboardDidShow', () => {
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 150);
    });
    return () => sub.remove();
  }, []);

  const handleRegenerate = async () => {
    setIsRegenerating(true);
    try {
      await regenerateLastMessage();
      setSwipeCount(prev => prev + 1);
      setSwipeIndex(prev => prev + 1);
    } finally {
      setIsRegenerating(false);
    }
  };

  const handleEditSaved = async (newContent: string) => {
    if (!editTarget) return;
    const isUserEdit = (editTarget as any).isUser;

    if (isUserEdit) {
      // User message edit → replace message and regenerate AI response (creates branch)
      const store = useChatStore.getState();
      const idx = store.messages.findIndex(m => m.id === editTarget.id);
      if (idx >= 0) {
        // Trim to the edited message, update it, then send for new AI response
        const trimmed = store.messages.slice(0, idx);
        const editedMsg: ChatMessage = { ...store.messages[idx], content: newContent };
        useChatStore.setState({ messages: [...trimmed, editedMsg] });
        // Send the edited message as if it's new — will generate fresh response
        await sendMessage(newContent);
      }
    } else {
      // Assistant message edit — update in place (DPO training pair)
      const store = useChatStore.getState();
      const idx = store.messages.findIndex(m => m.id === editTarget.id);
      if (idx >= 0) {
        const updated = [...store.messages];
        updated[idx] = { ...updated[idx], content: newContent };
        useChatStore.setState({ messages: updated });
      }
    }
    setEditTarget(null);
  };

  const handleSendWithNote = (text: string) => {
    sendMessage(text, directorNote || undefined);
  };

  const handleUpdateCharacter = async () => {
    if (!character?.id) return;
    try {
      const { apiFetch } = await import('../../lib/api');
      const { mapCardFromBackend } = await import('../../lib/mappers');
      const raw = await apiFetch<any>(`/api/cards/${character.id}`);
      const updated = mapCardFromBackend(raw);
      useChatStore.setState({ character: updated });
      showAlert('Updated', `${updated.name}'s card has been refreshed. Changes will apply to the next message.`);
    } catch (e) {
      showAlert('Error', 'Failed to update character card.');
    }
  };

  const handleImportContext = () => {
    if (!contextInput.trim()) return;
    setSessionContext(contextInput.trim());
    setShowContextModal(false);
    setContextInput('');
    showAlert('Context Applied', 'This context will be referenced by the AI in every response for this session.');
  };

  const lastAssistantIdx = messages.length > 0 && messages[messages.length - 1]?.role === 'assistant'
    ? messages.length - 1 : -1;

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <SessionHeader
        characterName={character?.name ?? 'Chat'}
        accentColor={accentColor}
        characterId={character?.id}
        isTyping={isStreaming}
        onNewSession={character ? () => { startSession(character, 'roleplay'); } : undefined}
        onClearHistory={() => { clearSession(); }}
        onExportChat={messages.length > 1 ? () => {
          const text = messages
            .filter(m => m.role !== 'system')
            .map(m => `${m.role === 'user' ? 'You' : character?.name ?? 'Character'}: ${m.content}`)
            .join('\n\n---\n\n');
          Share.share({ message: `Chat with ${character?.name ?? 'Character'} (${messages.length} messages)\n\n` + text });
        } : undefined}
        onMenuOpen={(actions) => {
          // Insert Update Character and Import Context before Clear History
          const extraActions = [
            { text: 'Update Character', onPress: handleUpdateCharacter },
            { text: 'Import Context', onPress: () => setShowContextModal(true) },
          ];
          const clearIdx = actions.findIndex(a => a.style === 'destructive');
          const enriched = [...actions];
          enriched.splice(clearIdx >= 0 ? clearIdx : actions.length - 1, 0, ...extraActions);
          showAlert(character?.name ?? 'Chat', undefined, enriched.map(a => ({
            text: a.text,
            style: a.style as any,
            onPress: a.onPress,
          })));
        }}
      />

      {/* Branch selector (only visible when multiple branches exist) */}
      <BranchSelector branches={branches} currentBranch={currentBranch} onSelect={setCurrentBranch} accentColor={accentColor} />

      <KeyboardAvoidingView style={styles.chatArea} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={insets.top}>
        <FlatList
          ref={listRef}
          data={messages}
          renderItem={({ item, index }: { item: ChatMessage; index: number }) => {
            const isAssistant = item.role === 'assistant';
            const isLastAssistant = index === lastAssistantIdx;
            const isLastUser = item.role === 'user' && index === messages.length - 1 && !isStreaming;
            const isLastUserBeforeAssistant = item.role === 'user' && index < messages.length - 1 && messages[index + 1]?.role === 'assistant';

            return (
              <View>
                <MessageBubble message={item} accentColor={accentColor} />

                {/* Seen indicator */}
                {(isLastUser || isLastUserBeforeAssistant) && !isStreaming && (
                  <Text style={[styles.seenText, { color: tc.text.muted }]}>Seen ✓</Text>
                )}

                {/* Edit button on USER messages (enables branching) */}
                {!isAssistant && !isStreaming && index > 0 && (
                  <View style={[styles.msgActions, { justifyContent: 'flex-end' }]}>
                    <TouchableOpacity
                      style={styles.actionBtn}
                      onPress={() => setEditTarget({ id: item.id, content: item.content, isUser: true } as any)}
                      hitSlop={8}
                    >
                      <Pencil size={14} color={tc.text.faint} />
                    </TouchableOpacity>
                  </View>
                )}

                {/* Feedback + Regenerate + Edit on assistant messages */}
                {isAssistant && !isStreaming && (
                  <View style={styles.msgActions}>
                    <FeedbackButtons messageId={item.id} accentColor={accentColor} />
                    {isLastAssistant && (
                      <TouchableOpacity
                        style={styles.actionBtn}
                        onPress={handleRegenerate}
                        disabled={isRegenerating}
                        hitSlop={8}
                      >
                        <RefreshCw size={14} color={tc.text.faint} />
                      </TouchableOpacity>
                    )}
                    <TouchableOpacity
                      style={styles.actionBtn}
                      onPress={() => setEditTarget({ id: item.id, content: item.content })}
                      hitSlop={8}
                    >
                      <Pencil size={14} color={tc.text.faint} />
                    </TouchableOpacity>
                  </View>
                )}

                {/* Swipe indicator removed — regenerate is now inline with edit button */}
              </View>
            );
          }}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.msgList}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={character ? (
            messages.length <= 2 ? (
              <View style={[styles.charIntro, { backgroundColor: (accentColor ?? tc.accent.primary) + '08', borderColor: (accentColor ?? tc.accent.primary) + '20' }]}>
                <View style={[styles.charIntroAvatar, { backgroundColor: (accentColor ?? tc.accent.primary) + '15' }]}>
                  <Text style={[styles.charIntroLetter, { color: accentColor ?? tc.accent.primary }]}>{character.name[0]}</Text>
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.charIntroName, { color: tc.text.primary }]}>{character.name}</Text>
                  <Text style={[styles.charIntroDesc, { color: tc.text.secondary }]} numberOfLines={2}>{character.description}</Text>
                </View>
              </View>
            ) : messages.length > 2 ? (
              <Text style={[styles.resumeBanner, { color: tc.text.muted }]}>Conversation resumed · {messages.length} messages</Text>
            ) : null
          ) : null}
          onScroll={(e) => { const y = e.nativeEvent.contentOffset.y; const h = e.nativeEvent.contentSize.height - e.nativeEvent.layoutMeasurement.height; setShowScrollBtn(h - y > 200); }}
          scrollEventThrottle={100}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          ListFooterComponent={<View>
            {isStreaming && streamingContent ? <StreamingBubble content={streamingContent} accentColor={accentColor} /> : isStreaming ? <TypingIndicator characterName={character?.name} /> : null}
          </View>}
        />

        {/* Scroll to bottom FAB */}
        {showScrollBtn && (
          <TouchableOpacity style={[styles.scrollFab, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]} onPress={() => listRef.current?.scrollToEnd({ animated: true })} activeOpacity={0.8}>
            <ChevronDown size={18} color={tc.text.secondary} />
          </TouchableOpacity>
        )}

        {showSaved && <Text style={[styles.savedText, { color: tc.status.success }]}>Auto-saved ✓</Text>}

        <SuggestionChips suggestions={suggestions} onSelect={p => handleSendWithNote(p)} accentColor={accentColor} />

        {/* Director's Note bar */}
        <DirectorNoteBar
          note={directorNote}
          lastUsedNote={lastUsedNote}
          onNoteChange={setDirectorNote}
          onReuse={() => setDirectorNote(lastUsedNote)}
          onClear={() => { setDirectorNote(''); setLastUsedNote(''); }}
          accentColor={accentColor}
        />

        <ChatInput onSend={handleSendWithNote} disabled={isStreaming} accentColor={accentColor} />
      </KeyboardAvoidingView>

      {/* Training opt-in popup (shown once) */}
      {showOptIn && <TrainingOptInPopup onDismiss={dismissOptIn} />}

      {/* Edit bottom sheet */}
      {editTarget && (
        <EditSheet
          visible={!!editTarget}
          messageId={editTarget.id}
          originalContent={editTarget.content}
          onClose={() => setEditTarget(null)}
          onSaved={handleEditSaved}
          accentColor={accentColor}
        />
      )}

      {/* Import Context modal */}
      <Modal visible={showContextModal} transparent animationType="fade" onRequestClose={() => setShowContextModal(false)}>
        <TouchableWithoutFeedback onPress={() => setShowContextModal(false)}>
          <View style={styles.contextOverlay}>
            <TouchableWithoutFeedback>
              <View style={[styles.contextModal, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]}>
                <Text style={[styles.contextTitle, { color: tc.text.primary }]}>Import Context</Text>
                <Text style={[styles.contextDesc, { color: tc.text.muted }]}>Add backstory, world info, or instructions the AI should reference throughout this conversation.</Text>
                <TextInput
                  style={[styles.contextInput, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]}
                  value={contextInput}
                  onChangeText={setContextInput}
                  placeholder="e.g. Marcus is a ronin from Edo period Japan who lost his clan..."
                  placeholderTextColor={tc.text.muted}
                  multiline
                  textAlignVertical="top"
                  autoFocus
                />
                <View style={styles.contextBtnRow}>
                  <TouchableOpacity style={[styles.contextBtn, { backgroundColor: tc.bg.tertiary }]} onPress={() => setShowContextModal(false)}>
                    <Text style={[styles.contextBtnText, { color: tc.text.muted }]}>Cancel</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.contextBtn, { backgroundColor: (accentColor ?? tc.accent.primary) + '15' }]} onPress={handleImportContext} disabled={!contextInput.trim()}>
                    <Text style={[styles.contextBtnText, { color: accentColor ?? tc.accent.primary }]}>Apply</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>

      {/* Themed alert modal */}
      {AlertComponent}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  chatArea: { flex: 1 },
  msgList: { paddingVertical: spacing.md },
  msgActions: { flexDirection: 'row', alignItems: 'center' },
  editBtn: { padding: 4, marginLeft: 4 },
  actionBtn: { padding: 6, marginLeft: 6 },
  scrollFab: { position: 'absolute', right: spacing.lg, bottom: 225, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, zIndex: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  charIntro: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.md, borderRadius: 12, borderWidth: 1 },
  charIntroAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  charIntroLetter: { fontSize: 18, fontFamily: fonts.heading },
  charIntroName: { fontSize: 13, fontFamily: fonts.heading, marginBottom: 2 },
  charIntroDesc: { fontSize: 11, fontFamily: fonts.body, lineHeight: 15 },
  seenText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'right', paddingRight: spacing.lg, marginTop: 2 },
  resumeBanner: { fontSize: 10, fontFamily: fonts.body, textAlign: 'center', paddingVertical: spacing.sm, marginBottom: spacing.sm },
  savedText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'center', paddingVertical: 3 },
  contextOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center', padding: spacing.xl },
  contextModal: { width: '100%', maxWidth: 360, borderRadius: 16, padding: spacing.xl, borderWidth: 1, gap: spacing.md },
  contextTitle: { fontSize: 18, fontFamily: fonts.heading, textAlign: 'center' },
  contextDesc: { fontSize: 12, fontFamily: fonts.body, textAlign: 'center', lineHeight: 18 },
  contextInput: { borderRadius: 10, borderWidth: 1, paddingHorizontal: spacing.md, paddingVertical: spacing.md, fontSize: 14, fontFamily: fonts.body, minHeight: 120, textAlignVertical: 'top' },
  contextBtnRow: { flexDirection: 'row', gap: spacing.sm },
  contextBtn: { flex: 1, paddingVertical: spacing.md, borderRadius: 10, alignItems: 'center' },
  contextBtnText: { fontSize: 15, fontFamily: fonts.heading },
});

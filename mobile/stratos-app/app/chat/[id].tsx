import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Share, Keyboard, Modal, TextInput, TouchableWithoutFeedback } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { useThemedAlert } from '../../components/shared/ThemedAlert';
import { useChatStore } from '../../stores/chatStore';
import { SessionHeader } from '../../components/chat/SessionHeader';
import { ChatMessageList } from '../../components/chat/ChatMessageList';
import { ChatInputSection } from '../../components/chat/ChatInputSection';
import { BranchSelector } from '../../components/chat/BranchSelector';
import { EditSheet } from '../../components/chat/EditSheet';
import { TrainingOptInPopup, useTrainingOptInCheck } from '../../components/chat/TrainingOptIn';
import { ChatMessage } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { reportError } from '../../lib/utils';
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
  const [editTarget, setEditTarget] = useState<{ id: string; content: string; isUser?: boolean } | null>(null);
  const [showContextModal, setShowContextModal] = useState(false);
  const [contextInput, setContextInput] = useState('');
  const [isRegenerating, setIsRegenerating] = useState(false);
  const { showOptIn, dismiss: dismissOptIn } = useTrainingOptInCheck();
  const { alert: showAlert, AlertComponent } = useThemedAlert();

  // Persist session when leaving
  useFocusEffect(useCallback(() => { return () => { persistSession().catch(err => reportError('ChatScreen:onBlur:persistSession', err)); }; }, []));

  useEffect(() => {
    if (isStreaming && !prevStreamRef.current) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    if (!isStreaming && prevStreamRef.current) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      persistSession().catch(err => reportError('ChatScreen:streamEnd:persistSession', err));
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

  const handleRegenerate = useCallback(async () => {
    setIsRegenerating(true);
    try {
      await regenerateLastMessage();
      setSwipeCount(prev => prev + 1);
      setSwipeIndex(prev => prev + 1);
    } finally {
      setIsRegenerating(false);
    }
  }, [regenerateLastMessage]);

  const handleEditSaved = async (newContent: string) => {
    if (!editTarget) return;
    const isUserEdit = editTarget.isUser;

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

  const handleSendWithNote = useCallback((text: string) => {
    sendMessage(text, directorNote || undefined);
  }, [sendMessage, directorNote]);

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

  const handleEditUser = useCallback((msg: { id: string; content: string; isUser: boolean }) => {
    setEditTarget(msg);
  }, []);

  const handleEditAssistant = useCallback((msg: { id: string; content: string }) => {
    setEditTarget(msg);
  }, []);

  const handleScroll = useCallback((e: any) => {
    const y = e.nativeEvent.contentOffset.y;
    const h = e.nativeEvent.contentSize.height - e.nativeEvent.layoutMeasurement.height;
    setShowScrollBtn(h - y > 200);
  }, []);

  const handleScrollToEnd = useCallback(() => {
    listRef.current?.scrollToEnd({ animated: true });
  }, []);

  const handleReuseNote = useCallback(() => setDirectorNote(lastUsedNote), [lastUsedNote]);
  const handleClearNote = useCallback(() => { setDirectorNote(''); setLastUsedNote(''); }, []);

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
        <ChatMessageList
          listRef={listRef}
          messages={messages}
          isStreaming={isStreaming}
          streamingContent={streamingContent}
          character={character}
          accentColor={accentColor}
          isRegenerating={isRegenerating}
          showScrollBtn={showScrollBtn}
          showSaved={showSaved}
          onEditUser={handleEditUser}
          onEditAssistant={handleEditAssistant}
          onRegenerate={handleRegenerate}
          onScroll={handleScroll}
          onScrollToEnd={handleScrollToEnd}
        />

        <ChatInputSection
          suggestions={suggestions}
          directorNote={directorNote}
          lastUsedNote={lastUsedNote}
          isStreaming={isStreaming}
          accentColor={accentColor}
          onSend={handleSendWithNote}
          onNoteChange={setDirectorNote}
          onReuseNote={handleReuseNote}
          onClearNote={handleClearNote}
        />
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
  contextOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center', padding: spacing.xl },
  contextModal: { width: '100%', maxWidth: 360, borderRadius: 16, padding: spacing.xl, borderWidth: 1, gap: spacing.md },
  contextTitle: { fontSize: 18, fontFamily: fonts.heading, textAlign: 'center' },
  contextDesc: { fontSize: 12, fontFamily: fonts.body, textAlign: 'center', lineHeight: 18 },
  contextInput: { borderRadius: 10, borderWidth: 1, paddingHorizontal: spacing.md, paddingVertical: spacing.md, fontSize: 14, fontFamily: fonts.body, minHeight: 120, textAlignVertical: 'top' },
  contextBtnRow: { flexDirection: 'row', gap: spacing.sm },
  contextBtn: { flex: 1, paddingVertical: spacing.md, borderRadius: 10, alignItems: 'center' },
  contextBtnText: { fontSize: 15, fontFamily: fonts.heading },
});

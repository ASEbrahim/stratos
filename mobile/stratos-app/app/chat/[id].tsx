import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Alert, Share, Keyboard } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Pencil } from 'lucide-react-native';
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
  const { character, messages, suggestions, isStreaming, streamingContent, sendMessage, persistSession, startSession, clearSession, regenerateLastMessage } = useChatStore();
  const accentColor = character ? getGenreColor(character.genre_tags?.[0] ?? 'default') : undefined;

  // ── RP Expansion state ──
  const [directorNote, setDirectorNote] = useState('');
  const [lastUsedNote, setLastUsedNote] = useState('');
  const [branches, setBranches] = useState<any[]>([{ id: 'main', parent_branch_id: null, turn_count: 0, is_active: true }]);
  const [currentBranch, setCurrentBranch] = useState('main');
  const [swipeCount, setSwipeCount] = useState(0);
  const [swipeIndex, setSwipeIndex] = useState(0);
  const [editTarget, setEditTarget] = useState<{ id: string; content: string } | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);

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

  const handleEditSaved = (newContent: string) => {
    // Update the message in local state (store will handle persistence)
    if (editTarget) {
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
    // The director's note is passed alongside the message
    // In mock mode this has no effect, but when wired to backend
    // the chatStore.sendMessage could include it
    sendMessage(text);
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

                {/* Feedback + Edit + Swipe on assistant messages */}
                {isAssistant && !isStreaming && (
                  <View style={styles.msgActions}>
                    <FeedbackButtons messageId={item.id} accentColor={accentColor} />
                    <TouchableOpacity
                      style={styles.editBtn}
                      onPress={() => setEditTarget({ id: item.id, content: item.content })}
                      hitSlop={8}
                    >
                      <Pencil size={11} color={tc.text.faint} />
                    </TouchableOpacity>
                  </View>
                )}

                {/* Swipe indicator on last assistant message */}
                {isLastAssistant && !isStreaming && (
                  <SwipeIndicator
                    currentIndex={swipeIndex}
                    totalCount={Math.max(swipeCount, 1)}
                    onPrev={() => setSwipeIndex(Math.max(0, swipeIndex - 1))}
                    onNext={() => setSwipeIndex(Math.min(swipeCount - 1, swipeIndex + 1))}
                    onRegenerate={handleRegenerate}
                    isRegenerating={isRegenerating || isStreaming}
                    accentColor={accentColor}
                  />
                )}
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
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  chatArea: { flex: 1 },
  msgList: { paddingVertical: spacing.md },
  msgActions: { flexDirection: 'row', alignItems: 'center' },
  editBtn: { padding: 4, marginLeft: 4 },
  scrollFab: { position: 'absolute', right: spacing.lg, bottom: 80, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, zIndex: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  charIntro: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.md, borderRadius: 12, borderWidth: 1 },
  charIntroAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  charIntroLetter: { fontSize: 18, fontFamily: fonts.heading },
  charIntroName: { fontSize: 13, fontFamily: fonts.heading, marginBottom: 2 },
  charIntroDesc: { fontSize: 11, fontFamily: fonts.body, lineHeight: 15 },
  seenText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'right', paddingRight: spacing.lg, marginTop: 2 },
  resumeBanner: { fontSize: 10, fontFamily: fonts.body, textAlign: 'center', paddingVertical: spacing.sm, marginBottom: spacing.sm },
  savedText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'center', paddingVertical: 3 },
});

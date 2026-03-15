import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity, Alert, Share, Keyboard } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { useChatStore } from '../../stores/chatStore';
import { SessionHeader } from '../../components/chat/SessionHeader';
import { MessageBubble, StreamingBubble } from '../../components/chat/MessageBubble';
import { TypingIndicator } from '../../components/chat/TypingIndicator';
import { SuggestionChips } from '../../components/chat/SuggestionChips';
import { ChatInput } from '../../components/chat/ChatInput';
import { RefreshCw, ChevronDown } from 'lucide-react-native';
import { ChatMessage } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList>(null);
  const prevStreamRef = useRef(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [showSaved, setShowSaved] = useState(false);
  const tc = useThemeStore(s => s.colors);
  const { character, messages, suggestions, isStreaming, streamingContent, sendMessage, persistSession, startSession, clearSession, regenerateLastMessage } = useChatStore();
  const accentColor = character ? getGenreColor(character.genre_tags?.[0] ?? 'default') : undefined;

  // Persist session when leaving the screen
  useFocusEffect(useCallback(() => { return () => { persistSession(); }; }, []));

  useEffect(() => {
    if (isStreaming && !prevStreamRef.current) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    if (!isStreaming && prevStreamRef.current) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      persistSession();
      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 2000);
    }
    prevStreamRef.current = isStreaming;
  }, [isStreaming]);
  useEffect(() => { if (messages.length > 0) setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100); }, [messages.length, streamingContent]);
  // Auto-scroll when keyboard opens
  useEffect(() => {
    const sub = Keyboard.addListener('keyboardDidShow', () => {
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 150);
    });
    return () => sub.remove();
  }, []);

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
          const header = `Chat with ${character?.name ?? 'Character'} (${messages.length} messages)\n\n`;
          Share.share({ message: header + text });
        } : undefined}
      />
      <KeyboardAvoidingView style={styles.chatArea} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={insets.top}>
        <FlatList ref={listRef} data={messages} renderItem={({ item, index }: { item: ChatMessage; index: number }) => {
          const isLastUser = item.role === 'user' && index === messages.length - 1 && !isStreaming;
          const isLastUserBeforeAssistant = item.role === 'user' && index < messages.length - 1 && messages[index + 1]?.role === 'assistant';
          return (
            <View>
              <MessageBubble message={item} accentColor={accentColor} />
              {(isLastUser || isLastUserBeforeAssistant) && !isStreaming && (
                <Text style={[styles.seenText, { color: tc.text.muted }]}>Seen ✓</Text>
              )}
            </View>
          );
        }} keyExtractor={item => item.id} contentContainerStyle={styles.msgList} showsVerticalScrollIndicator={false}
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
            {!isStreaming && messages.length > 1 && messages[messages.length - 1]?.role === 'assistant' && (
              <TouchableOpacity style={styles.regenBtn} onPress={regenerateLastMessage} activeOpacity={0.7}>
                <RefreshCw size={12} color={tc.text.muted} />
                <Text style={[styles.regenText, { color: tc.text.muted }]}>Regenerate</Text>
              </TouchableOpacity>
            )}
          </View>} />
        {showScrollBtn && (
          <TouchableOpacity style={[styles.scrollFab, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]} onPress={() => listRef.current?.scrollToEnd({ animated: true })} activeOpacity={0.8}>
            <ChevronDown size={18} color={tc.text.secondary} />
          </TouchableOpacity>
        )}
        {showSaved && <Text style={[styles.savedText, { color: tc.status.success }]}>Auto-saved ✓</Text>}
        <SuggestionChips suggestions={suggestions} onSelect={p => sendMessage(p)} accentColor={accentColor} />
        <ChatInput onSend={sendMessage} disabled={isStreaming} accentColor={accentColor} />
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  chatArea: { flex: 1 },
  msgList: { paddingVertical: spacing.md },
  regenBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, marginTop: spacing.xs },
  regenText: { fontSize: 11, fontWeight: '500' },
  scrollFab: { position: 'absolute', right: spacing.lg, bottom: 80, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, zIndex: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
  charIntro: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.md, borderRadius: 12, borderWidth: 1 },
  charIntroAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  charIntroLetter: { fontSize: 18, fontWeight: '700' },
  charIntroName: { fontSize: 13, fontWeight: '700', marginBottom: 2 },
  charIntroDesc: { fontSize: 11, lineHeight: 15 },
  seenText: { fontSize: 9, textAlign: 'right', paddingRight: spacing.lg, marginTop: 2 },
  resumeBanner: { fontSize: 10, textAlign: 'center', paddingVertical: spacing.sm, marginBottom: spacing.sm },
  savedText: { fontSize: 9, textAlign: 'center', paddingVertical: 3, fontWeight: '600' },
});

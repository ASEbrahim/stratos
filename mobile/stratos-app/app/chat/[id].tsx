import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, Text, FlatList, StyleSheet, KeyboardAvoidingView, Platform, TouchableOpacity } from 'react-native';
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
  const tc = useThemeStore(s => s.colors);
  const { character, messages, suggestions, isStreaming, streamingContent, sendMessage, persistSession, startSession, clearSession, regenerateLastMessage } = useChatStore();
  const accentColor = character ? getGenreColor(character.genre_tags?.[0] ?? 'default') : undefined;

  // Persist session when leaving the screen
  useFocusEffect(useCallback(() => { return () => { persistSession(); }; }, []));

  useEffect(() => { if (isStreaming && !prevStreamRef.current) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); prevStreamRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { if (messages.length > 0) setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100); }, [messages.length, streamingContent]);

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <SessionHeader
        characterName={character?.name ?? 'Chat'}
        accentColor={accentColor}
        characterId={character?.id}
        onNewSession={character ? () => { startSession(character, 'roleplay'); } : undefined}
        onClearHistory={() => { clearSession(); }}
      />
      <KeyboardAvoidingView style={styles.chatArea} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={insets.top}>
        <FlatList ref={listRef} data={messages} renderItem={({ item }: { item: ChatMessage }) => <MessageBubble message={item} accentColor={accentColor} />} keyExtractor={item => item.id} contentContainerStyle={styles.msgList} showsVerticalScrollIndicator={false}
          onScroll={(e) => { const y = e.nativeEvent.contentOffset.y; const h = e.nativeEvent.contentSize.height - e.nativeEvent.layoutMeasurement.height; setShowScrollBtn(h - y > 200); }}
          scrollEventThrottle={100}
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
});

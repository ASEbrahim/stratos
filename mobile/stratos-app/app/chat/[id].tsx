import React, { useRef, useEffect } from 'react';
import { View, FlatList, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import { useChatStore } from '../../stores/chatStore';
import { SessionHeader } from '../../components/chat/SessionHeader';
import { MessageBubble, StreamingBubble } from '../../components/chat/MessageBubble';
import { TypingIndicator } from '../../components/chat/TypingIndicator';
import { SuggestionChips } from '../../components/chat/SuggestionChips';
import { ChatInput } from '../../components/chat/ChatInput';
import { ChatMessage } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { colors, spacing } from '../../constants/theme';

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList>(null);
  const prevStreamRef = useRef(false);
  const { character, messages, suggestions, isStreaming, streamingContent, sendMessage } = useChatStore();
  const accentColor = character ? getGenreColor(character.genre_tags[0] ?? 'default') : undefined;

  useEffect(() => { if (isStreaming && !prevStreamRef.current) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); prevStreamRef.current = isStreaming; }, [isStreaming]);
  useEffect(() => { if (messages.length > 0) setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100); }, [messages.length, streamingContent]);

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <SessionHeader characterName={character?.name ?? 'Chat'} accentColor={accentColor} />
      <KeyboardAvoidingView style={styles.chatArea} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={insets.top}>
        <FlatList ref={listRef} data={messages} renderItem={({ item }: { item: ChatMessage }) => <MessageBubble message={item} accentColor={accentColor} />} keyExtractor={item => item.id} contentContainerStyle={styles.msgList} showsVerticalScrollIndicator={false}
          ListFooterComponent={<View>{isStreaming && streamingContent ? <StreamingBubble content={streamingContent} accentColor={accentColor} /> : isStreaming ? <TypingIndicator /> : null}</View>} />
        <SuggestionChips suggestions={suggestions} onSelect={p => sendMessage(p)} accentColor={accentColor} />
        <ChatInput onSend={sendMessage} disabled={isStreaming} accentColor={accentColor} />
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  chatArea: { flex: 1 },
  msgList: { paddingVertical: spacing.md },
});

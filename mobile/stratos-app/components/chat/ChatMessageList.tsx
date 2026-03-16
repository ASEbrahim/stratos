import React from 'react';
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from 'react-native';
import { Pencil, RefreshCw, ChevronDown } from 'lucide-react-native';
import { MessageBubble, StreamingBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';
import { FeedbackButtons } from './FeedbackButtons';
import { ChatMessage, CharacterCard } from '../../lib/types';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

interface ChatMessageListProps {
  listRef: React.RefObject<FlatList | null>;
  messages: ChatMessage[];
  isStreaming: boolean;
  streamingContent: string;
  character: CharacterCard | null;
  accentColor: string | undefined;
  isRegenerating: boolean;
  showScrollBtn: boolean;
  showSaved: boolean;
  onEditUser: (msg: { id: string; content: string; isUser: boolean }) => void;
  onEditAssistant: (msg: { id: string; content: string }) => void;
  onRegenerate: () => void;
  onScroll: (e: any) => void;
  onScrollToEnd: () => void;
}

export const ChatMessageList = React.memo(function ChatMessageList({
  listRef,
  messages,
  isStreaming,
  streamingContent,
  character,
  accentColor,
  isRegenerating,
  showScrollBtn,
  showSaved,
  onEditUser,
  onEditAssistant,
  onRegenerate,
  onScroll,
  onScrollToEnd,
}: ChatMessageListProps) {
  const tc = useThemeStore(s => s.colors);

  const lastAssistantIdx = messages.length > 0 && messages[messages.length - 1]?.role === 'assistant'
    ? messages.length - 1 : -1;

  return (
    <>
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
                    onPress={() => onEditUser({ id: item.id, content: item.content, isUser: true })}
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
                      onPress={onRegenerate}
                      disabled={isRegenerating}
                      hitSlop={8}
                    >
                      <RefreshCw size={14} color={tc.text.faint} />
                    </TouchableOpacity>
                  )}
                  <TouchableOpacity
                    style={styles.actionBtn}
                    onPress={() => onEditAssistant({ id: item.id, content: item.content })}
                    hitSlop={8}
                  >
                    <Pencil size={14} color={tc.text.faint} />
                  </TouchableOpacity>
                </View>
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
        onScroll={onScroll}
        scrollEventThrottle={100}
        keyboardDismissMode="on-drag"
        keyboardShouldPersistTaps="handled"
        ListFooterComponent={<View>
          {isStreaming && streamingContent ? <StreamingBubble content={streamingContent} accentColor={accentColor} /> : isStreaming ? <TypingIndicator characterName={character?.name} /> : null}
        </View>}
      />

      {/* Scroll to bottom FAB */}
      {showScrollBtn && (
        <TouchableOpacity style={[styles.scrollFab, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]} onPress={onScrollToEnd} activeOpacity={0.8}>
          <ChevronDown size={18} color={tc.text.secondary} />
        </TouchableOpacity>
      )}

      {showSaved && <Text style={[styles.savedText, { color: tc.status.success }]}>Auto-saved ✓</Text>}
    </>
  );
});

const styles = StyleSheet.create({
  msgList: { paddingVertical: spacing.md },
  msgActions: { flexDirection: 'row', alignItems: 'center' },
  actionBtn: { padding: 6, marginLeft: 6 },
  seenText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'right', paddingRight: spacing.lg, marginTop: 2 },
  resumeBanner: { fontSize: 10, fontFamily: fonts.body, textAlign: 'center', paddingVertical: spacing.sm, marginBottom: spacing.sm },
  savedText: { fontSize: 9, fontFamily: fonts.body, textAlign: 'center', paddingVertical: 3 },
  charIntro: { flexDirection: 'row', alignItems: 'center', gap: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.md, borderRadius: 12, borderWidth: 1 },
  charIntroAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  charIntroLetter: { fontSize: 18, fontFamily: fonts.heading },
  charIntroName: { fontSize: 13, fontFamily: fonts.heading, marginBottom: 2 },
  charIntroDesc: { fontSize: 11, fontFamily: fonts.body, lineHeight: 15 },
  scrollFab: { position: 'absolute', right: spacing.lg, bottom: 225, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, zIndex: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
});

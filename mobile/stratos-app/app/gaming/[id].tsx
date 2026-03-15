import React, { useEffect, useState, useRef } from 'react';
import { View, FlatList, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import { SessionHeader } from '../../components/chat/SessionHeader';
import { MessageBubble, StreamingBubble } from '../../components/chat/MessageBubble';
import { TypingIndicator } from '../../components/chat/TypingIndicator';
import { ChatInput } from '../../components/chat/ChatInput';
import { StatBar } from '../../components/gaming/StatBar';
import { OptionButtons } from '../../components/gaming/OptionButtons';
import { GamingScenario, ChatMessage } from '../../lib/types';
import { getScenario, startGamingSession, parseOptions } from '../../lib/gaming';
import { streamMessage, createMessageId } from '../../lib/chat';
import { getGenreColor } from '../../constants/genres';
import { LoadingScreen } from '../../components/shared/LoadingScreen';
import { colors, spacing } from '../../constants/theme';

export default function GamingSessionScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList>(null);
  const [scenario, setScenario] = useState<GamingScenario | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [currentOptions, setCurrentOptions] = useState<string[]>([]);
  const [playerStats, setPlayerStats] = useState<Record<string, number>>({});
  const accentColor = scenario ? getGenreColor(scenario.genre) : undefined;

  useEffect(() => {
    if (!id) return;
    getScenario(id).then(s => {
      if (!s) return;
      setScenario(s);
      const player = s.entities.find(e => e.name === 'Player');
      if (player?.stats) setPlayerStats(player.stats);
      startGamingSession(id).then(sid => {
        setSessionId(sid);
        const parsed = parseOptions(s.initial_message);
        setMessages([{ id: createMessageId(), role: 'assistant', content: parsed.text, timestamp: new Date().toISOString() }]);
        setCurrentOptions(parsed.options);
      });
    });
  }, [id]);

  const handleSend = async (text: string) => {
    if (!sessionId) return;
    setMessages(prev => [...prev, { id: createMessageId(), role: 'user', content: text, timestamp: new Date().toISOString() }]);
    setIsStreaming(true); setStreamingContent(''); setCurrentOptions([]);
    let acc = '';
    await streamMessage(sessionId, text, 'gaming', null, chunk => { acc += chunk; setStreamingContent(acc); }, () => {
      const parsed = parseOptions(acc);
      setMessages(prev => [...prev, { id: createMessageId(), role: 'assistant', content: parsed.text, timestamp: new Date().toISOString() }]);
      setCurrentOptions(parsed.options); setIsStreaming(false); setStreamingContent('');
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    });
  };

  useEffect(() => { if (messages.length > 0) setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100); }, [messages.length, streamingContent]);
  if (!scenario) return <LoadingScreen />;

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <SessionHeader characterName={scenario.name} accentColor={accentColor} />
      {Object.keys(playerStats).length > 0 && <StatBar stats={playerStats} accentColor={accentColor} />}
      <KeyboardAvoidingView style={styles.chatArea} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <FlatList ref={listRef} data={messages} renderItem={({ item }: { item: ChatMessage }) => <MessageBubble message={item} accentColor={accentColor} />} keyExtractor={item => item.id} contentContainerStyle={styles.msgList} showsVerticalScrollIndicator={false}
          ListFooterComponent={<View>{isStreaming && streamingContent ? <StreamingBubble content={streamingContent} /> : isStreaming ? <TypingIndicator /> : null}{!isStreaming && currentOptions.length > 0 && <OptionButtons options={currentOptions} onSelect={(o, i) => handleSend(`${i + 1}. ${o}`)} accentColor={accentColor} />}</View>} />
        <ChatInput onSend={handleSend} disabled={isStreaming} accentColor={accentColor} />
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  chatArea: { flex: 1 },
  msgList: { paddingVertical: spacing.md },
});

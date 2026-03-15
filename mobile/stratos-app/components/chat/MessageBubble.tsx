import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, FadeIn } from 'react-native-reanimated';
import { ChatMessage } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';

interface MessageBubbleProps { message: ChatMessage; accentColor?: string; }

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    const h = d.getHours();
    const m = d.getMinutes().toString().padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    return `${h % 12 || 12}:${m} ${ampm}`;
  } catch { return ''; }
}

export function MessageBubble({ message, accentColor }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const [showTime, setShowTime] = useState(false);
  const tc = useThemeStore(s => s.colors);

  if (isSystem) return <View style={styles.systemContainer}><Text style={[styles.systemText, { color: tc.text.muted }]}>{message.content}</Text></View>;

  const handleLongPress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Message', undefined, [
      { text: 'Copy Text', onPress: () => {} },
      { text: 'Regenerate', onPress: () => {}, style: 'default' },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const handleTap = () => setShowTime(prev => !prev);

  const accent = accentColor || tc.accent.primary;

  return (
    <Animated.View entering={FadeIn.duration(200)} style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <TouchableOpacity
        style={[
          styles.bubble,
          isUser
            ? [styles.userBubble, { backgroundColor: accent + '20', borderColor: accent + '15' }]
            : [styles.assistantBubble, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]
        ]}
        onLongPress={handleLongPress}
        onPress={handleTap}
        activeOpacity={0.9}
        delayLongPress={400}
      >
        {renderFormattedText(message.content, isUser, tc)}
      </TouchableOpacity>
      {showTime && message.timestamp && (
        <Text style={[styles.timestamp, { color: tc.text.muted }, isUser ? styles.timestampRight : styles.timestampLeft]}>{formatTime(message.timestamp)}</Text>
      )}
    </Animated.View>
  );
}

function renderFormattedText(text: string, isUser: boolean, tc: any) {
  const textColor = tc.text.primary;
  const italicColor = tc.text.secondary;
  return text.split('\n\n').map((para, pi) => (
    <View key={pi} style={pi > 0 ? { marginTop: spacing.md } : undefined}>
      {para.split('\n').map((line, li) => <Text key={li} style={[isUser ? styles.userText : styles.assistantText, { color: textColor }]}>{renderLine(line, italicColor)}</Text>)}
    </View>
  ));
}

function renderLine(line: string, italicColor: string) {
  const parts: React.ReactNode[] = [];
  const regex = /\*([^*]+)\*/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(line)) !== null) {
    if (match.index > lastIndex) parts.push(<Text key={`t-${lastIndex}`}>{line.slice(lastIndex, match.index)}</Text>);
    parts.push(<Text key={`a-${match.index}`} style={{ fontStyle: 'italic', color: italicColor }}>{match[1]}</Text>);
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < line.length) parts.push(<Text key={`r-${lastIndex}`}>{line.slice(lastIndex)}</Text>);
  return parts.length === 0 ? line : parts;
}

export function StreamingBubble({ content, accentColor }: { content: string; accentColor?: string }) {
  const tc = useThemeStore(s => s.colors);
  return (
    <Animated.View entering={FadeIn.duration(150)} style={[styles.container, styles.assistantContainer]}>
      <View style={[styles.assistantBubble, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]}>{renderFormattedText(content, false, tc)}</View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.xs },
  userContainer: { alignItems: 'flex-end' },
  assistantContainer: { alignItems: 'flex-start' },
  systemContainer: { alignItems: 'center', paddingHorizontal: spacing.xxl, paddingVertical: spacing.sm },
  bubble: { maxWidth: '85%', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg },
  userBubble: { borderTopRightRadius: borderRadius.sm, borderWidth: 1 },
  assistantBubble: { borderTopLeftRadius: borderRadius.sm, maxWidth: '90%', borderWidth: 1 },
  userText: { ...typography.body },
  assistantText: { ...typography.body, lineHeight: 24 },
  systemText: { ...typography.caption, textAlign: 'center', fontStyle: 'italic' },
  timestamp: { ...typography.small, fontSize: 9, marginTop: 2 },
  timestampLeft: { marginLeft: spacing.sm },
  timestampRight: { marginRight: spacing.sm },
});

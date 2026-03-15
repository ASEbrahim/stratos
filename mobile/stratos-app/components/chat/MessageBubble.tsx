import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, FadeIn } from 'react-native-reanimated';
import { ChatMessage } from '../../lib/types';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

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

  if (isSystem) return <View style={styles.systemContainer}><Text style={styles.systemText}>{message.content}</Text></View>;

  const handleLongPress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Copy to clipboard would go here on native; for now show options
    Alert.alert('Message', undefined, [
      { text: 'Copy Text', onPress: () => {} },
      { text: 'Regenerate', onPress: () => {}, style: 'default' },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const handleTap = () => setShowTime(prev => !prev);

  return (
    <Animated.View entering={FadeIn.duration(200)} style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <TouchableOpacity
        style={[styles.bubble, isUser ? [styles.userBubble, accentColor ? { backgroundColor: accentColor + '20', borderColor: accentColor + '15' } : null] : styles.assistantBubble]}
        onLongPress={handleLongPress}
        onPress={handleTap}
        activeOpacity={0.9}
        delayLongPress={400}
      >
        {renderFormattedText(message.content, isUser)}
      </TouchableOpacity>
      {showTime && message.timestamp && (
        <Text style={[styles.timestamp, isUser ? styles.timestampRight : styles.timestampLeft]}>{formatTime(message.timestamp)}</Text>
      )}
    </Animated.View>
  );
}

function renderFormattedText(text: string, isUser: boolean) {
  return text.split('\n\n').map((para, pi) => (
    <View key={pi} style={pi > 0 ? { marginTop: spacing.md } : undefined}>
      {para.split('\n').map((line, li) => <Text key={li} style={isUser ? styles.userText : styles.assistantText}>{renderLine(line)}</Text>)}
    </View>
  ));
}

function renderLine(line: string) {
  const parts: React.ReactNode[] = [];
  const regex = /\*([^*]+)\*/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(line)) !== null) {
    if (match.index > lastIndex) parts.push(<Text key={`t-${lastIndex}`}>{line.slice(lastIndex, match.index)}</Text>);
    parts.push(<Text key={`a-${match.index}`} style={styles.actionText}>{match[1]}</Text>);
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < line.length) parts.push(<Text key={`r-${lastIndex}`}>{line.slice(lastIndex)}</Text>);
  return parts.length === 0 ? line : parts;
}

export function StreamingBubble({ content, accentColor }: { content: string; accentColor?: string }) {
  return (
    <Animated.View entering={FadeIn.duration(150)} style={[styles.container, styles.assistantContainer]}>
      <View style={styles.assistantBubble}>{renderFormattedText(content, false)}</View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.xs },
  userContainer: { alignItems: 'flex-end' },
  assistantContainer: { alignItems: 'flex-start' },
  systemContainer: { alignItems: 'center', paddingHorizontal: spacing.xxl, paddingVertical: spacing.sm },
  bubble: { maxWidth: '85%', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg },
  userBubble: { backgroundColor: colors.accent.primary + '25', borderTopRightRadius: borderRadius.sm, borderWidth: 1, borderColor: colors.accent.primary + '15' },
  assistantBubble: { backgroundColor: colors.bg.tertiary, borderTopLeftRadius: borderRadius.sm, maxWidth: '90%', borderWidth: 1, borderColor: colors.border.subtle },
  userText: { ...typography.body, color: colors.text.primary },
  assistantText: { ...typography.body, color: colors.text.primary, lineHeight: 24 },
  systemText: { ...typography.caption, color: colors.text.muted, textAlign: 'center', fontStyle: 'italic' },
  actionText: { fontStyle: 'italic', color: colors.text.secondary },
  timestamp: { ...typography.small, color: colors.text.muted, fontSize: 9, marginTop: 2 },
  timestampLeft: { marginLeft: spacing.sm },
  timestampRight: { marginRight: spacing.sm },
});

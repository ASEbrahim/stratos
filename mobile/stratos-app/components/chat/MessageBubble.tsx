import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ChatMessage } from '../../lib/types';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface MessageBubbleProps { message: ChatMessage; accentColor?: string; }

export function MessageBubble({ message, accentColor }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  if (isSystem) return <View style={styles.systemContainer}><Text style={styles.systemText}>{message.content}</Text></View>;
  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <View style={[styles.bubble, isUser ? [styles.userBubble, accentColor ? { backgroundColor: accentColor + '25' } : null] : styles.assistantBubble]}>
        {renderFormattedText(message.content, isUser)}
      </View>
    </View>
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
    <View style={[styles.container, styles.assistantContainer]}>
      <View style={styles.assistantBubble}>{renderFormattedText(content, false)}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.xs },
  userContainer: { alignItems: 'flex-end' },
  assistantContainer: { alignItems: 'flex-start' },
  systemContainer: { alignItems: 'center', paddingHorizontal: spacing.xxl, paddingVertical: spacing.sm },
  bubble: { maxWidth: '85%', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg },
  userBubble: { backgroundColor: colors.accent.primary + '30', borderTopRightRadius: borderRadius.sm, borderWidth: 1, borderColor: colors.accent.primary + '20' },
  assistantBubble: { backgroundColor: colors.bg.tertiary, borderTopLeftRadius: borderRadius.sm, maxWidth: '90%', borderWidth: 1, borderColor: colors.border.subtle },
  userText: { ...typography.body, color: colors.text.primary },
  assistantText: { ...typography.body, color: colors.text.primary, lineHeight: 24 },
  systemText: { ...typography.caption, color: colors.text.muted, textAlign: 'center', fontStyle: 'italic' },
  actionText: { fontStyle: 'italic', color: colors.text.secondary },
});

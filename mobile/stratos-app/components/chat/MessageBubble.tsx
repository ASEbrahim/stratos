import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import * as Clipboard from 'expo-clipboard';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, FadeIn } from 'react-native-reanimated';
import { ChatMessage } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import { fonts } from '../../constants/fonts';

interface MessageBubbleProps { message: ChatMessage; accentColor?: string; }

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    const h = d.getHours();
    const m = d.getMinutes().toString().padStart(2, '0');
    const time = `${h % 12 || 12}:${m} ${h >= 12 ? 'PM' : 'AM'}`;

    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (d.toDateString() === now.toDateString()) return time;
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return `Yesterday ${time}`;
    return `${d.getMonth() + 1}/${d.getDate()} ${time}`;
  } catch { return ''; }
}

export const MessageBubble = React.memo(function MessageBubble({ message, accentColor }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const [showTime, setShowTime] = useState(false);
  const tc = useThemeStore(s => s.colors);

  const formattedContent = useMemo(() => renderFormattedText(message.content, isUser, tc), [message.content, isUser, tc]);

  if (isSystem) return <View style={styles.systemContainer}><Text style={[styles.systemText, { color: tc.text.muted }]}>{message.content}</Text></View>;

  const handleLongPress = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Message', undefined, [
      { text: 'Copy Text', onPress: async () => {
        await Clipboard.setStringAsync(message.content);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }},
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
            : [styles.assistantBubble, { backgroundColor: tc.bg.tertiary, borderColor: accent + '10', borderLeftColor: accent + '30', borderLeftWidth: 2 }]
        ]}
        onLongPress={handleLongPress}
        onPress={handleTap}
        activeOpacity={0.9}
        delayLongPress={400}
      >
        {formattedContent}
      </TouchableOpacity>
      {showTime && message.timestamp && (
        <Text style={[styles.timestamp, { color: tc.text.muted }, isUser ? styles.timestampRight : styles.timestampLeft]}>{formatTime(message.timestamp)}</Text>
      )}
    </Animated.View>
  );
});

function renderFormattedText(text: string, isUser: boolean, tc: any) {
  const textColor = tc.text.primary;
  const italicColor = tc.text.secondary;
  const quoteColor = tc.accent.primary;
  return text.split('\n\n').map((para, pi) => (
    <View key={pi} style={pi > 0 ? { marginTop: spacing.md } : undefined}>
      {para.split('\n').map((line, li) => {
        const isQuote = line.startsWith('> ');
        const displayLine = isQuote ? line.slice(2) : line;
        if (isQuote) {
          return (
            <View key={li} style={{ flexDirection: 'row', marginVertical: 2 }}>
              <View style={{ width: 3, backgroundColor: quoteColor + '60', borderRadius: 2, marginRight: spacing.sm }} />
              <Text style={[isUser ? styles.userText : styles.assistantText, { color: italicColor, fontStyle: 'italic', flex: 1 }]}>{renderLine(displayLine, italicColor)}</Text>
            </View>
          );
        }
        return <Text key={li} style={[isUser ? styles.userText : styles.assistantText, { color: textColor }]}>{renderLine(displayLine, italicColor)}</Text>;
      })}
    </View>
  ));
}

function renderLine(line: string, italicColor: string) {
  // Process markup: ***bold italic***, **bold**, *italic*
  const parts: React.ReactNode[] = [];
  const regex = /(\*{3})([^*]+)\1|(\*{2})([^*]+)\3|(\*)([^*]+)\5/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(line)) !== null) {
    if (match.index > lastIndex) parts.push(<Text key={`t-${lastIndex}`}>{line.slice(lastIndex, match.index)}</Text>);
    if (match[1] === '***') {
      // bold italic
      parts.push(<Text key={`bi-${match.index}`} style={{ fontWeight: '700', fontStyle: 'italic', color: italicColor }}>{match[2]}</Text>);
    } else if (match[3] === '**') {
      // bold
      parts.push(<Text key={`b-${match.index}`} style={{ fontWeight: '700' }}>{match[4]}</Text>);
    } else {
      // italic
      parts.push(<Text key={`i-${match.index}`} style={{ fontStyle: 'italic', color: italicColor }}>{match[6]}</Text>);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < line.length) parts.push(<Text key={`r-${lastIndex}`}>{line.slice(lastIndex)}</Text>);
  return parts.length === 0 ? line : parts;
}

export const StreamingBubble = React.memo(function StreamingBubble({ content, accentColor }: { content: string; accentColor?: string }) {
  const tc = useThemeStore(s => s.colors);
  return (
    <Animated.View entering={FadeIn.duration(150)} style={[styles.container, styles.assistantContainer]}>
      <View style={[styles.assistantBubble, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]}>{renderFormattedText(content, false, tc)}</View>
    </Animated.View>
  );
});

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.xs },
  userContainer: { alignItems: 'flex-end' },
  assistantContainer: { alignItems: 'flex-start' },
  systemContainer: { alignItems: 'center', paddingHorizontal: spacing.xxl, paddingVertical: spacing.sm },
  bubble: { maxWidth: '85%', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg },
  userBubble: { borderTopRightRadius: borderRadius.sm, borderWidth: 1 },
  assistantBubble: { borderTopLeftRadius: borderRadius.sm, maxWidth: '90%', borderWidth: 1 },
  userText: { ...typography.body, fontFamily: fonts.body },
  assistantText: { ...typography.body, lineHeight: 24, fontFamily: fonts.body },
  systemText: { ...typography.caption, textAlign: 'center', fontStyle: 'italic', fontFamily: fonts.bodyLight },
  timestamp: { ...typography.small, fontSize: 9, marginTop: 2 },
  timestampLeft: { marginLeft: spacing.sm },
  timestampRight: { marginRight: spacing.sm },
});

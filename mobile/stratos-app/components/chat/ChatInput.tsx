import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Send, Square } from 'lucide-react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence, withRepeat, withTiming } from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius, typography } from '../../constants/theme';

interface ChatInputProps { onSend: (text: string) => void; disabled?: boolean; accentColor?: string; }

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export function ChatInput({ onSend, disabled = false, accentColor }: ChatInputProps) {
  const [text, setText] = useState('');
  const tc = useThemeStore(s => s.colors);
  const color = accentColor ?? tc.accent.primary;
  const btnScale = useSharedValue(1);
  const btnAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: btnScale.value }] }));

  const handleSend = () => {
    const t = text.trim();
    if (!t || disabled) return;
    btnScale.value = withSequence(withSpring(0.8, { damping: 15 }), withSpring(1, { damping: 8 }));
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    onSend(t);
    setText('');
  };

  const active = !!text.trim() && !disabled;
  const charCount = text.length;

  return (
    <View style={[styles.container, { backgroundColor: tc.bg.primary, borderTopColor: tc.border.subtle }]}>
      {charCount > 100 && (
        <Text style={[styles.charCount, { color: charCount > 3500 ? tc.status.error : tc.text.muted }]}>{charCount}/4000</Text>
      )}
      <View style={styles.inputRow}>
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary }]} value={text} onChangeText={setText} placeholder="Type your action..." placeholderTextColor={tc.text.muted} multiline maxLength={4000} editable={!disabled} onSubmitEditing={handleSend} blurOnSubmit={false} accessibilityLabel="Message input" accessibilityHint="Type your message to the character" />
        <AnimatedTouchable style={[styles.sendButton, { backgroundColor: disabled ? color + '60' : active ? color : tc.bg.elevated }, btnAnimStyle]} onPress={handleSend} disabled={!active && !disabled} activeOpacity={0.7} accessibilityLabel={disabled ? 'AI is responding' : 'Send message'} accessibilityRole="button">
          {disabled ? (
            <ActivityIndicator size={16} color="#fff" />
          ) : (
            <Send size={18} color={active ? '#fff' : tc.text.muted} />
          )}
        </AnimatedTouchable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { borderTopWidth: 1, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, paddingBottom: spacing.lg },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end', gap: spacing.sm },
  input: { flex: 1, borderRadius: borderRadius.xl, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, maxHeight: 120 },
  sendButton: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
  charCount: { ...typography.small, fontSize: 9, textAlign: 'right', marginBottom: 2 },
});

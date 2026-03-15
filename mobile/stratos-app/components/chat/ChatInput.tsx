import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Send } from 'lucide-react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence } from 'react-native-reanimated';
import { colors, spacing, borderRadius } from '../../constants/theme';

interface ChatInputProps { onSend: (text: string) => void; disabled?: boolean; accentColor?: string; }

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export function ChatInput({ onSend, disabled = false, accentColor }: ChatInputProps) {
  const [text, setText] = useState('');
  const color = accentColor ?? colors.accent.primary;
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

  return (
    <View style={styles.container}><View style={styles.inputRow}>
      <TextInput style={styles.input} value={text} onChangeText={setText} placeholder="Type your action..." placeholderTextColor={colors.text.muted} multiline maxLength={4000} editable={!disabled} onSubmitEditing={handleSend} blurOnSubmit={false} />
      <AnimatedTouchable style={[styles.sendButton, { backgroundColor: active ? color : colors.bg.elevated }, btnAnimStyle]} onPress={handleSend} disabled={!active} activeOpacity={0.7}>
        <Send size={18} color={active ? '#fff' : colors.text.muted} />
      </AnimatedTouchable>
    </View></View>
  );
}

const styles = StyleSheet.create({
  container: { backgroundColor: colors.bg.primary, borderTopWidth: 1, borderTopColor: colors.border.subtle, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, paddingBottom: spacing.lg },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end', gap: spacing.sm },
  input: { flex: 1, backgroundColor: colors.bg.tertiary, borderRadius: borderRadius.xl, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, color: colors.text.primary, fontSize: 15, maxHeight: 120 },
  sendButton: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
});

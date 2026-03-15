import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Send } from 'lucide-react-native';
import { colors, spacing, borderRadius } from '../../constants/theme';

interface ChatInputProps { onSend: (text: string) => void; disabled?: boolean; accentColor?: string; }

export function ChatInput({ onSend, disabled = false, accentColor }: ChatInputProps) {
  const [text, setText] = useState('');
  const color = accentColor ?? colors.accent.primary;
  const handleSend = () => { const t = text.trim(); if (!t || disabled) return; Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); onSend(t); setText(''); };
  return (
    <View style={styles.container}><View style={styles.inputRow}>
      <TextInput style={styles.input} value={text} onChangeText={setText} placeholder="Type your action..." placeholderTextColor={colors.text.muted} multiline maxLength={4000} editable={!disabled} onSubmitEditing={handleSend} blurOnSubmit={false} />
      <TouchableOpacity style={[styles.sendButton, { backgroundColor: text.trim() && !disabled ? color : colors.bg.elevated }]} onPress={handleSend} disabled={!text.trim() || disabled} activeOpacity={0.7}>
        <Send size={18} color={text.trim() && !disabled ? '#fff' : colors.text.muted} />
      </TouchableOpacity>
    </View></View>
  );
}

const styles = StyleSheet.create({
  container: { backgroundColor: colors.bg.primary, borderTopWidth: 1, borderTopColor: colors.border.subtle, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, paddingBottom: spacing.lg },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end', gap: spacing.sm },
  input: { flex: 1, backgroundColor: colors.bg.tertiary, borderRadius: borderRadius.xl, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, color: colors.text.primary, fontSize: 15, maxHeight: 120 },
  sendButton: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' },
});

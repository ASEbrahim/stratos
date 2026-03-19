import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Modal, KeyboardAvoidingView, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { X, Check } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { editMessage } from '../../lib/rp';
import { reportError } from '../../lib/utils';

const EDIT_REASONS = [
  { id: 'voice', label: 'Voice' },
  { id: 'length', label: 'Length' },
  { id: 'accuracy', label: 'Accuracy' },
  { id: 'tone', label: 'Tone' },
  { id: 'agency', label: 'Agency' },
  { id: 'other', label: 'Other' },
] as const;

interface EditSheetProps {
  visible: boolean;
  messageId: string;
  dbId?: number;  // Backend rp_messages.id
  originalContent: string;
  onClose: () => void;
  onSaved: (newContent: string) => void;
  accentColor?: string;
}

export function EditSheet({ visible, messageId, dbId, originalContent, onClose, onSaved, accentColor }: EditSheetProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;
  const [content, setContent] = useState(originalContent);
  const [reason, setReason] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (content.trim() === originalContent.trim()) { onClose(); return; }
    setSaving(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    try {
      // Only call backend if we have a real DB message ID
      if (dbId) {
        await editMessage(dbId, content.trim(), reason || undefined);
      }
      // Always update the UI locally regardless
      onSaved(content.trim());
      onClose();
    } catch (err) {
      reportError('EditSheet:handleSave', err);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView style={styles.overlay} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <View style={[styles.sheet, { backgroundColor: tc.bg.primary, borderColor: tc.border.subtle }]}>
          <View style={styles.header}>
            <Text style={[styles.title, { color: tc.text.primary }]}>Edit Response</Text>
            <TouchableOpacity onPress={onClose} hitSlop={8}>
              <X size={20} color={tc.text.muted} />
            </TouchableOpacity>
          </View>

          <TextInput
            style={[styles.editor, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]}
            value={content}
            onChangeText={setContent}
            multiline
            textAlignVertical="top"
            autoFocus
          />

          <Text style={[styles.reasonLabel, { color: tc.text.muted }]}>Reason (optional)</Text>
          <View style={styles.reasonRow}>
            {EDIT_REASONS.map(r => (
              <TouchableOpacity
                key={r.id}
                style={[styles.reasonChip, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle },
                  reason === r.id && { backgroundColor: accent + '20', borderColor: accent }]}
                onPress={() => { setReason(reason === r.id ? null : r.id); Haptics.selectionAsync(); }}
              >
                <Text style={[styles.reasonText, { color: tc.text.secondary },
                  reason === r.id && { color: accent }]}>{r.label}</Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity
            style={[styles.saveBtn, { backgroundColor: accent }, saving && { opacity: 0.6 }]}
            onPress={handleSave}
            disabled={saving}
          >
            <Check size={16} color="#fff" />
            <Text style={styles.saveBtnText}>{saving ? 'Saving...' : 'Save Edit'}</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: { flex: 1, justifyContent: 'flex-end', backgroundColor: 'rgba(0,0,0,0.5)' },
  sheet: { borderTopLeftRadius: 20, borderTopRightRadius: 20, borderWidth: 1, borderBottomWidth: 0, padding: spacing.lg, maxHeight: '80%' },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.md },
  title: { fontSize: 16, fontFamily: fonts.heading },
  editor: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 14, fontFamily: fonts.body, minHeight: 150, maxHeight: 300, borderWidth: 1 },
  reasonLabel: { fontSize: 11, fontFamily: fonts.body, marginTop: spacing.md, marginBottom: spacing.sm },
  reasonRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs },
  reasonChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.xs, borderRadius: borderRadius.full, borderWidth: 1 },
  reasonText: { fontSize: 11, fontFamily: fonts.button },
  saveBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.lg, marginTop: spacing.lg },
  saveBtnText: { fontSize: 14, fontFamily: fonts.heading, color: '#fff' },
});

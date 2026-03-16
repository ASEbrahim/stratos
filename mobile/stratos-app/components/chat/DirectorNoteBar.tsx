import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Sparkles, X, RotateCcw } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

interface DirectorNoteBarProps {
  note: string;
  lastUsedNote: string;
  onNoteChange: (note: string) => void;
  onReuse: () => void;
  onClear: () => void;
  accentColor?: string;
}

export function DirectorNoteBar({ note, lastUsedNote, onNoteChange, onReuse, onClear, accentColor }: DirectorNoteBarProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;
  const [expanded, setExpanded] = useState(false);

  const toggle = () => { setExpanded(!expanded); Haptics.selectionAsync(); };

  return (
    <View style={[styles.container, { borderTopColor: tc.border.subtle }]}>
      <TouchableOpacity onPress={toggle} style={styles.header} activeOpacity={0.7}>
        <Sparkles size={14} color={note ? accent : tc.text.muted} />
        <Text style={[styles.headerText, { color: note ? accent : tc.text.muted }]}>
          {note ? 'Director\'s Note active' : 'Steer AI'}
        </Text>
        {note && (
          <TouchableOpacity onPress={() => { onClear(); Haptics.selectionAsync(); }} hitSlop={8}>
            <X size={14} color={tc.text.muted} />
          </TouchableOpacity>
        )}
      </TouchableOpacity>

      {expanded && (
        <View style={styles.body}>
          <TextInput
            style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: note ? accent + '40' : tc.border.subtle }]}
            value={note}
            onChangeText={onNoteChange}
            placeholder="Guide the AI's next response (e.g., 'keep it short', 'make her hesitant')"
            placeholderTextColor={tc.text.muted}
            multiline
            maxLength={500}
          />
          {note.trim() ? (
            <Text style={[styles.activeHint, { color: accent }]}>Will apply to your next message</Text>
          ) : null}
          {!note && lastUsedNote ? (
            <TouchableOpacity
              onPress={() => { onReuse(); Haptics.selectionAsync(); }}
              style={[styles.reuseBtn, { borderColor: accent + '30' }]}
            >
              <RotateCcw size={12} color={accent} />
              <Text style={[styles.reuseText, { color: accent }]} numberOfLines={1}>
                Reuse: {lastUsedNote}
              </Text>
            </TouchableOpacity>
          ) : null}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { borderTopWidth: 1 },
  header: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  headerText: { fontSize: 12, fontFamily: fonts.button, flex: 1 },
  body: { paddingHorizontal: spacing.lg, paddingBottom: spacing.sm, gap: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, fontSize: 13, fontFamily: fonts.body, minHeight: 40, maxHeight: 80, borderWidth: 1 },
  reuseBtn: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, borderRadius: borderRadius.full, borderWidth: 1, alignSelf: 'flex-start' },
  reuseText: { fontSize: 11, fontFamily: fonts.body, maxWidth: 200 },
  activeHint: { fontSize: 10, fontFamily: fonts.body, marginTop: 2 },
});

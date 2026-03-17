import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Upload, Wand2 } from 'lucide-react-native';
import { CharacterCardCreate } from '../../lib/types';
import { FEATURES } from '../../constants/config';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { useThemeStore } from '../../stores/themeStore';
import Animated, { AnimatedStyle } from 'react-native-reanimated';

interface EditorActionsProps {
  card: CharacterCardCreate;
  isEditing: boolean;
  saving: boolean;
  importing: boolean;
  saveBtnAnimStyle: AnimatedStyle;
  onSave: () => void;
  onImportTavern: () => void;
  onGenerateImage: () => void;
  onUpdateContentRating: (rating: string) => void;
}

export const EditorActions = React.memo(function EditorActions({
  card,
  isEditing,
  saving,
  importing,
  saveBtnAnimStyle,
  onSave,
  onImportTavern,
  onGenerateImage,
  onUpdateContentRating,
}: EditorActionsProps) {
  const tc = useThemeStore(s => s.colors);

  return (
    <>
      {/* TavernCard import */}
      {FEATURES.enableTavernImport && (
        <TouchableOpacity style={[styles.importBtn, { borderColor: tc.accent.primary + '40', backgroundColor: tc.accent.primary + '08' }]} onPress={onImportTavern} disabled={importing} activeOpacity={0.7} accessibilityLabel={importing ? 'Importing TavernCard' : 'Import TavernCard V2'} accessibilityRole="button">
          <Upload size={16} color={tc.accent.primary} />
          <Text style={[styles.importText, { color: tc.accent.primary }]}>{importing ? 'Importing...' : 'Import TavernCard V2'}</Text>
        </TouchableOpacity>
      )}

      {/* Generate Image — accessible from top of editor */}
      <TouchableOpacity
        style={[styles.importBtn, { borderColor: tc.accent.secondary + '40', backgroundColor: tc.accent.secondary + '08' }]}
        onPress={onGenerateImage}
        activeOpacity={0.7}
      >
        <Wand2 size={16} color={tc.accent.secondary} />
        <Text style={[styles.importText, { color: tc.accent.secondary }]}>Generate Image</Text>
      </TouchableOpacity>

      {/* Content Rating */}
      <View style={styles.ratingRow}>
        <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Content Rating</Text>
        <View style={styles.contentRatingRow}>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'sfw' && { backgroundColor: tc.status.success + '20', borderColor: tc.status.success }]} onPress={() => onUpdateContentRating('sfw')} accessibilityLabel={`SFW content rating${card.content_rating === 'sfw' ? ', selected' : ''}`} accessibilityRole="button">
            <Text style={{ color: card.content_rating === 'sfw' ? tc.status.success : tc.text.muted }}>SFW</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'nsfw' && { backgroundColor: tc.nsfw + '20', borderColor: tc.nsfw }]} onPress={() => onUpdateContentRating('nsfw')} accessibilityLabel={`NSFW content rating${card.content_rating === 'nsfw' ? ', selected' : ''}`} accessibilityRole="button">
            <Text style={{ color: card.content_rating === 'nsfw' ? tc.nsfw : tc.text.muted }}>NSFW</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Save button */}
      <Animated.View style={saveBtnAnimStyle}>
        <TouchableOpacity style={[styles.saveBtn, { backgroundColor: tc.accent.primary }, saving && { opacity: 0.6 }]} onPress={onSave} disabled={saving} accessibilityLabel={saving ? (isEditing ? 'Saving changes' : 'Creating character') : (isEditing ? 'Save changes' : 'Create character')} accessibilityRole="button">
          <Text style={styles.saveBtnText}>{saving ? (isEditing ? 'Saving...' : 'Creating...') : (isEditing ? 'Save Changes' : 'Create Character')}</Text>
        </TouchableOpacity>
      </Animated.View>
    </>
  );
});

const styles = StyleSheet.create({
  importBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1, marginBottom: spacing.lg },
  importText: { ...typography.caption, fontWeight: '600' },
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  ratingRow: { marginTop: spacing.lg },
  contentRatingRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  ratingOpt: { paddingHorizontal: spacing.xl, paddingVertical: spacing.sm, borderRadius: borderRadius.md, borderWidth: 1 },
  genImageBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderWidth: 1, marginTop: spacing.lg },
  genImageText: { fontSize: 13, fontFamily: fonts.button },
  saveBtn: { paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginTop: spacing.md },
  saveBtnText: { ...typography.subheading, color: '#fff' },
});

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { Camera, Image as ImageIcon, X } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface AvatarPickerProps {
  avatarUri: string;
  onPick: (uri: string) => void;
  onClear: () => void;
  accentColor?: string;
  compact?: boolean;
}

export function AvatarPicker({ avatarUri, onPick, onClear, accentColor, compact }: AvatarPickerProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow photo access to pick a character avatar.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      onPick(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow camera access to take a character photo.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled && result.assets[0]) {
      onPick(result.assets[0].uri);
    }
  };

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <TouchableOpacity onPress={pickImage} activeOpacity={0.7} accessibilityLabel={avatarUri ? 'Change avatar' : 'Pick avatar'} accessibilityRole="button">
          {avatarUri ? (
            <View style={styles.compactPreviewWrap}>
              <Image source={{ uri: avatarUri }} style={styles.compactPreview} />
              <TouchableOpacity style={styles.compactClearBtn} onPress={onClear} accessibilityLabel="Remove avatar" accessibilityRole="button">
                <X size={12} color="#fff" />
              </TouchableOpacity>
            </View>
          ) : (
            <View style={[styles.compactPlaceholder, { borderColor: accent + '30', backgroundColor: tc.bg.tertiary }]}>
              <Camera size={20} color={tc.text.muted} />
            </View>
          )}
        </TouchableOpacity>
      </View>
    );
  }

  if (avatarUri) {
    return (
      <View style={styles.previewContainer}>
        <Image source={{ uri: avatarUri }} style={styles.preview} />
        <TouchableOpacity style={styles.clearBtn} onPress={onClear} accessibilityLabel="Remove avatar" accessibilityRole="button">
          <X size={16} color={tc.text.primary} />
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={[styles.placeholder, { borderColor: accent + '30', backgroundColor: tc.bg.tertiary }]}>
        <ImageIcon size={32} color={tc.text.muted} />
        <Text style={[styles.label, { color: tc.text.muted }]}>Character Avatar</Text>
      </View>
      <View style={styles.buttonRow}>
        <TouchableOpacity style={[styles.pickBtn, { borderColor: accent + '40', backgroundColor: tc.bg.tertiary }]} onPress={pickImage} accessibilityLabel="Pick avatar from gallery" accessibilityRole="button">
          <ImageIcon size={16} color={accent} />
          <Text style={[styles.pickText, { color: accent }]}>Gallery</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.pickBtn, { borderColor: accent + '40', backgroundColor: tc.bg.tertiary }]} onPress={takePhoto} accessibilityLabel="Take avatar photo with camera" accessibilityRole="button">
          <Camera size={16} color={accent} />
          <Text style={[styles.pickText, { color: accent }]}>Camera</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginBottom: spacing.lg },
  placeholder: {
    width: '100%', aspectRatio: 1, maxHeight: 200,
    borderRadius: borderRadius.xl, borderWidth: 1, borderStyle: 'dashed',
    justifyContent: 'center', alignItems: 'center', gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  label: { ...typography.caption },
  buttonRow: { flexDirection: 'row', gap: spacing.sm },
  pickBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm,
    paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1,
  },
  pickText: { ...typography.caption, fontWeight: '600' },
  previewContainer: { position: 'relative', marginBottom: spacing.lg },
  preview: { width: '100%', aspectRatio: 1, maxHeight: 200, borderRadius: borderRadius.xl },
  clearBtn: {
    position: 'absolute', top: spacing.sm, right: spacing.sm,
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center',
  },
  compactContainer: {},
  compactPlaceholder: {
    width: 80, height: 80, borderRadius: borderRadius.lg,
    borderWidth: 1, borderStyle: 'dashed',
    justifyContent: 'center', alignItems: 'center',
  },
  compactPreviewWrap: { position: 'relative' },
  compactPreview: { width: 80, height: 80, borderRadius: borderRadius.lg },
  compactClearBtn: {
    position: 'absolute', top: -4, right: -4,
    width: 20, height: 20, borderRadius: 10,
    backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center',
  },
});

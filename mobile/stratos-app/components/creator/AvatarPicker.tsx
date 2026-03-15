import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { Camera, Image as ImageIcon, X } from 'lucide-react-native';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface AvatarPickerProps {
  avatarUri: string;
  onPick: (uri: string) => void;
  onClear: () => void;
  accentColor?: string;
}

export function AvatarPicker({ avatarUri, onPick, onClear, accentColor = colors.accent.primary }: AvatarPickerProps) {
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

  if (avatarUri) {
    return (
      <View style={styles.previewContainer}>
        <Image source={{ uri: avatarUri }} style={styles.preview} />
        <TouchableOpacity style={styles.clearBtn} onPress={onClear}>
          <X size={16} color={colors.text.primary} />
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={[styles.placeholder, { borderColor: accentColor + '30' }]}>
        <ImageIcon size={32} color={colors.text.muted} />
        <Text style={styles.label}>Character Avatar</Text>
      </View>
      <View style={styles.buttonRow}>
        <TouchableOpacity style={[styles.pickBtn, { borderColor: accentColor + '40' }]} onPress={pickImage}>
          <ImageIcon size={16} color={accentColor} />
          <Text style={[styles.pickText, { color: accentColor }]}>Gallery</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.pickBtn, { borderColor: accentColor + '40' }]} onPress={takePhoto}>
          <Camera size={16} color={accentColor} />
          <Text style={[styles.pickText, { color: accentColor }]}>Camera</Text>
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
    backgroundColor: colors.bg.tertiary,
    justifyContent: 'center', alignItems: 'center', gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  label: { ...typography.caption, color: colors.text.muted },
  buttonRow: { flexDirection: 'row', gap: spacing.sm },
  pickBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm,
    paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1,
    backgroundColor: colors.bg.tertiary,
  },
  pickText: { ...typography.caption, fontWeight: '600' },
  previewContainer: { position: 'relative', marginBottom: spacing.lg },
  preview: { width: '100%', aspectRatio: 1, maxHeight: 200, borderRadius: borderRadius.xl },
  clearBtn: {
    position: 'absolute', top: spacing.sm, right: spacing.sm,
    width: 28, height: 28, borderRadius: 14,
    backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center',
  },
});

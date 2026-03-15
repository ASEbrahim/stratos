import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { ChevronLeft, MoreVertical } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';

interface SessionHeaderProps {
  characterName: string;
  accentColor?: string;
  characterId?: string;
  onNewSession?: () => void;
  onClearHistory?: () => void;
}

export function SessionHeader({ characterName, accentColor, characterId, onNewSession, onClearHistory }: SessionHeaderProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const color = accentColor ?? tc.accent.primary;

  const handleMenu = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const actions: any[] = [];
    if (characterId) actions.push({ text: 'Character Info', onPress: () => router.push(`/character/${characterId}`) });
    if (onNewSession) actions.push({ text: 'New Session', onPress: onNewSession });
    if (onClearHistory) actions.push({ text: 'Clear History', style: 'destructive', onPress: onClearHistory });
    actions.push({ text: 'Cancel', style: 'cancel' });
    Alert.alert(characterName, undefined, actions);
  };

  return (
    <View style={[styles.container, { backgroundColor: tc.bg.primary, borderBottomColor: tc.border.subtle }]}>
      <TouchableOpacity onPress={() => router.back()} style={styles.backButton}><ChevronLeft size={24} color={tc.text.primary} /></TouchableOpacity>
      <View style={styles.center}>
        <View style={[styles.avatar, { backgroundColor: color + '20' }]}><Text style={[styles.avatarText, { color }]}>{characterName[0]}</Text></View>
        <Text style={[styles.name, { color: tc.text.primary }]} numberOfLines={1}>{characterName}</Text>
      </View>
      <TouchableOpacity style={styles.menuButton} onPress={handleMenu}><MoreVertical size={20} color={tc.text.secondary} /></TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.md, borderBottomWidth: 1 },
  backButton: { padding: spacing.xs },
  center: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingHorizontal: spacing.sm },
  avatar: { width: 32, height: 32, borderRadius: 16, justifyContent: 'center', alignItems: 'center' },
  avatarText: { fontSize: 14, fontWeight: '700' },
  name: { ...typography.subheading, flex: 1 },
  menuButton: { padding: spacing.xs },
});

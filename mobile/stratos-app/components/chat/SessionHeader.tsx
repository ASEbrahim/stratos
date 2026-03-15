import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { ChevronLeft, MoreVertical } from 'lucide-react-native';
import { colors, typography, spacing } from '../../constants/theme';

interface SessionHeaderProps { characterName: string; accentColor?: string; }

export function SessionHeader({ characterName, accentColor }: SessionHeaderProps) {
  const router = useRouter();
  const color = accentColor ?? colors.accent.primary;
  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={() => router.back()} style={styles.backButton}><ChevronLeft size={24} color={colors.text.primary} /></TouchableOpacity>
      <View style={styles.center}>
        <View style={[styles.avatar, { backgroundColor: color + '20' }]}><Text style={[styles.avatarText, { color }]}>{characterName[0]}</Text></View>
        <Text style={styles.name} numberOfLines={1}>{characterName}</Text>
      </View>
      <TouchableOpacity style={styles.menuButton}><MoreVertical size={20} color={colors.text.secondary} /></TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.md, backgroundColor: colors.bg.primary, borderBottomWidth: 1, borderBottomColor: colors.border.subtle },
  backButton: { padding: spacing.xs },
  center: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingHorizontal: spacing.sm },
  avatar: { width: 32, height: 32, borderRadius: 16, justifyContent: 'center', alignItems: 'center' },
  avatarText: { fontSize: 14, fontWeight: '700' },
  name: { ...typography.subheading, color: colors.text.primary, flex: 1 },
  menuButton: { padding: spacing.xs },
});

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { ChevronLeft } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';

interface HeaderProps { title?: string; subtitle?: string; showBack?: boolean; right?: React.ReactNode; }

export function Header({ title, subtitle, showBack = false, right }: HeaderProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={[styles.container, { backgroundColor: tc.bg.primary, borderBottomColor: tc.border.subtle }]}>
      <View style={styles.left}>
        {showBack && (
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <ChevronLeft size={24} color={tc.text.primary} />
          </TouchableOpacity>
        )}
      </View>
      <View style={{ flex: 1, alignItems: 'center' }}>
        {title && <Text style={[styles.title, { color: tc.text.primary }]} numberOfLines={1}>{title}</Text>}
        {subtitle && <Text style={[styles.subtitle, { color: tc.text.muted }]}>{subtitle}</Text>}
      </View>
      <View style={styles.right}>{right}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderBottomWidth: 1 },
  left: { width: 40 },
  title: { ...typography.subheading, textAlign: 'center' },
  subtitle: { fontSize: 10, textAlign: 'center', marginTop: 1 },
  right: { width: 40, alignItems: 'flex-end' },
  backButton: { padding: spacing.xs },
});

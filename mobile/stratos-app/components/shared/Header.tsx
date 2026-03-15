import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { ChevronLeft } from 'lucide-react-native';
import { colors, typography, spacing } from '../../constants/theme';

interface HeaderProps { title?: string; showBack?: boolean; right?: React.ReactNode; }

export function Header({ title, showBack = false, right }: HeaderProps) {
  const router = useRouter();
  return (
    <View style={styles.container}>
      <View style={styles.left}>
        {showBack && (
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <ChevronLeft size={24} color={colors.text.primary} />
          </TouchableOpacity>
        )}
      </View>
      {title && <Text style={styles.title} numberOfLines={1}>{title}</Text>}
      <View style={styles.right}>{right}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, backgroundColor: colors.bg.primary, borderBottomWidth: 1, borderBottomColor: colors.border.subtle },
  left: { width: 40 },
  title: { ...typography.subheading, color: colors.text.primary, flex: 1, textAlign: 'center' },
  right: { width: 40, alignItems: 'flex-end' },
  backButton: { padding: spacing.xs },
});

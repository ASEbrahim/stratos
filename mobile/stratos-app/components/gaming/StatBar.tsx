import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Heart, Sword, Brain, Users } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface StatBarProps { stats: Record<string, number>; accentColor?: string; }

const STAT_ICONS: Record<string, any> = {
  hp: Heart, strength: Sword, wisdom: Brain, charisma: Users,
  technical: Brain, perception: Brain, composure: Heart,
  sanity: Brain, courage: Heart,
};

const STAT_COLORS: Record<string, string> = {
  hp: '#ef4444', sanity: '#a78bfa',
  strength: '#f97316', technical: '#38bdf8',
  wisdom: '#34d399', perception: '#f0cc55',
  charisma: '#f0a0b8', composure: '#38bdf8',
  courage: '#e8b931',
};

export const StatBar = React.memo(function StatBar({ stats, accentColor }: StatBarProps) {
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={[styles.container, { backgroundColor: tc.bg.secondary, borderBottomColor: tc.border.subtle }]}>
      {Object.entries(stats).map(([key, value]) => {
        const color = STAT_COLORS[key.toLowerCase()] ?? accentColor ?? tc.accent.primary;
        const Icon = STAT_ICONS[key.toLowerCase()];
        const isHp = key.toLowerCase() === 'hp';
        const maxVal = isHp ? 100 : 20;
        const pct = Math.min(value / maxVal, 1);
        return (
          <View key={key} style={styles.stat}>
            <View style={styles.statHeader}>
              {Icon && <Icon size={10} color={color} />}
              <Text style={[styles.label, { color: tc.text.muted }]}>{key.toUpperCase()}</Text>
            </View>
            <Text style={[styles.value, { color }]}>{value}</Text>
            {isHp && (
              <View style={[styles.hpBarBg, { backgroundColor: tc.bg.tertiary }]}>
                <View style={[styles.hpBarFill, { backgroundColor: color, width: `${pct * 100}%` as any }]} />
              </View>
            )}
          </View>
        );
      })}
    </View>
  );
});

const styles = StyleSheet.create({
  container: { flexDirection: 'row', borderBottomWidth: 1, paddingVertical: spacing.sm, paddingHorizontal: spacing.lg, gap: spacing.lg },
  stat: { alignItems: 'center', minWidth: 44 },
  statHeader: { flexDirection: 'row', alignItems: 'center', gap: 2, marginBottom: 2 },
  label: { ...typography.small, letterSpacing: 1, fontSize: 8 },
  value: { ...typography.subheading, fontSize: 16 },
  hpBarBg: { width: 40, height: 3, borderRadius: 2, marginTop: 3, overflow: 'hidden' },
  hpBarFill: { height: '100%', borderRadius: 2 },
});

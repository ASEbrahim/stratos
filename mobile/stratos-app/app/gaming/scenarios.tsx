import React, { useEffect, useState } from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { ScenarioCard } from '../../components/gaming/ScenarioCard';
import { GamingScenario } from '../../lib/types';
import { getScenarios } from '../../lib/gaming';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';

export default function ScenariosScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  useEffect(() => { getScenarios().then(setScenarios); }, []);
  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title="Gaming Scenarios" showBack />
      <ScrollView contentContainerStyle={styles.content}>{scenarios.map(s => <ScenarioCard key={s.id} scenario={s} />)}</ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg, gap: spacing.lg },
});

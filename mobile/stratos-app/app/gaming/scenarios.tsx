import React, { useEffect, useState } from 'react';
import { View, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Plus } from 'lucide-react-native';
import { Header } from '../../components/shared/Header';
import { ScenarioCard } from '../../components/gaming/ScenarioCard';
import { GamingScenario } from '../../lib/types';
import { getScenarios } from '../../lib/gaming';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';

export default function ScenariosScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const router = useRouter();
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  useEffect(() => { getScenarios().then(setScenarios); }, []);
  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header
        title="Gaming Scenarios"
        showBack
        right={
          <TouchableOpacity onPress={() => router.push('/gaming/import')} activeOpacity={0.7}>
            <Plus size={22} color={tc.accent.primary} />
          </TouchableOpacity>
        }
      />
      <ScrollView contentContainerStyle={styles.content}>{scenarios.map(s => <ScenarioCard key={s.id} scenario={s} />)}</ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg, gap: spacing.lg },
});

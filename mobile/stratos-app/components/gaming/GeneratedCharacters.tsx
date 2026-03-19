import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';
import { fonts } from '../../constants/fonts';
import { spacing, borderRadius } from '../../constants/theme';
import { ScenarioNpc, mapNpcToCard } from '../../lib/mappers';
import { CharacterCardCreate } from '../../lib/types';

interface GeneratedCharactersProps {
  characters: ScenarioNpc[];
  worldScenario: string;
  onStartChat: (card: CharacterCardCreate) => void;
  onImport: (card: CharacterCardCreate) => void;
}

export default function GeneratedCharacters({
  characters,
  worldScenario,
  onStartChat,
  onImport,
}: GeneratedCharactersProps) {
  const tc = useThemeStore(s => s.colors);

  if (characters.length === 0) return null;

  return (
    <View style={styles.container}>
      <Text style={[styles.heading, { color: tc.text.primary, fontFamily: fonts.heading }]}>
        Generated Characters
      </Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {characters.map((npc, index) => {
          const card = mapNpcToCard(npc, worldScenario);
          return (
            <View
              key={`${npc.name}-${index}`}
              style={[styles.card, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}
            >
              <Text
                style={[styles.name, { color: tc.accent.primary, fontFamily: fonts.heading }]}
                numberOfLines={1}
              >
                {npc.name}
              </Text>
              <Text
                style={[styles.personality, { color: tc.text.muted, fontFamily: fonts.body }]}
                numberOfLines={2}
              >
                {npc.personality || npc.description || 'No description'}
              </Text>
              <View style={styles.buttons}>
                <TouchableOpacity
                  style={[styles.button, { backgroundColor: tc.accent.primary }]}
                  onPress={() => onStartChat(card)}
                  activeOpacity={0.7}
                >
                  <Text style={[styles.buttonText, { color: tc.text.inverse, fontFamily: fonts.button }]}>
                    Start Chat
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.button, { backgroundColor: tc.accent.secondary }]}
                  onPress={() => onImport(card)}
                  activeOpacity={0.7}
                >
                  <Text style={[styles.buttonText, { color: tc.text.inverse, fontFamily: fonts.button }]}>
                    Import
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginTop: spacing.lg },
  heading: { fontSize: 18, marginBottom: spacing.md, paddingHorizontal: spacing.lg },
  scroll: { paddingHorizontal: spacing.lg, gap: spacing.md },
  card: {
    width: 220,
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
  },
  name: { fontSize: 16, marginBottom: spacing.xs },
  personality: { fontSize: 13, lineHeight: 18, marginBottom: spacing.md, minHeight: 36 },
  buttons: { flexDirection: 'row', gap: spacing.sm },
  button: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  buttonText: { fontSize: 12 },
});

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { CardEditor } from '../../components/creator/CardEditor';
import { colors } from '../../constants/theme';

export default function CreateScreen() {
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <Header title="Create Character" />
      <CardEditor />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1, backgroundColor: colors.bg.primary } });

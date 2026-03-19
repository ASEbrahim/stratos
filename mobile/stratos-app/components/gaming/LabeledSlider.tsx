import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Slider from '@react-native-community/slider';
import { useThemeStore } from '../../stores/themeStore';
import { fonts } from '../../constants/fonts';

interface LabeledSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}

export default function LabeledSlider({ label, value, min, max, onChange }: LabeledSliderProps) {
  const tc = useThemeStore(s => s.colors);

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <Text style={[styles.label, { color: tc.text.secondary, fontFamily: fonts.bodyMedium }]}>{label}</Text>
        <Text style={[styles.value, { color: tc.accent.primary, fontFamily: fonts.bodySemiBold }]}>{value}</Text>
      </View>
      <Slider
        style={styles.slider}
        minimumValue={min}
        maximumValue={max}
        step={1}
        value={value}
        onValueChange={onChange}
        minimumTrackTintColor={tc.accent.primary}
        maximumTrackTintColor={tc.bg.tertiary}
        thumbTintColor={tc.accent.primary}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginBottom: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  label: { fontSize: 13 },
  value: { fontSize: 13 },
  slider: { width: '100%', height: 32 },
});

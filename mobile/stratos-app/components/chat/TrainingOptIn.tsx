import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Haptics from 'expo-haptics';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { setTrainingOptIn } from '../../lib/rp';
import { reportError } from '../../lib/utils';

const OPT_IN_KEY = 'training_opt_in_prompted';

interface TrainingOptInProps {
  onDismiss: () => void;
}

export function useTrainingOptInCheck() {
  const [showOptIn, setShowOptIn] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(OPT_IN_KEY).then(val => {
      if (!val) setShowOptIn(true);
    });
  }, []);

  const dismiss = () => {
    setShowOptIn(false);
    AsyncStorage.setItem(OPT_IN_KEY, 'true');
  };

  return { showOptIn, dismiss };
}

export function TrainingOptInPopup({ onDismiss }: TrainingOptInProps) {
  const tc = useThemeStore(s => s.colors);

  const handleAccept = async () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    try { await setTrainingOptIn(true); } catch (err) { reportError('TrainingOptIn:accept', err); }
    await AsyncStorage.setItem(OPT_IN_KEY, 'accepted');
    onDismiss();
  };

  const handleDecline = async () => {
    Haptics.selectionAsync();
    try { await setTrainingOptIn(false); } catch (err) { reportError('TrainingOptIn:decline', err); }
    await AsyncStorage.setItem(OPT_IN_KEY, 'declined');
    onDismiss();
  };

  return (
    <Modal visible transparent animationType="fade" onRequestClose={handleDecline}>
      <View style={styles.overlay}>
        <View style={[styles.card, { backgroundColor: tc.bg.primary, borderColor: tc.border.medium }]}>
          <Text style={[styles.title, { color: tc.text.primary }]}>Help Improve StratOS</Text>
          <Text style={[styles.body, { color: tc.text.secondary }]}>
            Your conversations help make StratOS better. Allow anonymous conversation data to be used for AI training?{'\n\n'}
            Your messages are stripped of personal details and only used to improve response quality. You can change this anytime in Settings.
          </Text>
          <View style={styles.btnRow}>
            <TouchableOpacity style={[styles.btn, { backgroundColor: tc.bg.tertiary }]} onPress={handleDecline} activeOpacity={0.7}>
              <Text style={[styles.btnText, { color: tc.text.secondary }]}>Decline</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.btn, { backgroundColor: tc.accent.primary }]} onPress={handleAccept} activeOpacity={0.7}>
              <Text style={[styles.btnText, { color: '#fff' }]}>Accept</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.6)', padding: spacing.xl },
  card: { width: '100%', borderRadius: borderRadius.lg, borderWidth: 1, padding: spacing.xl },
  title: { fontSize: 18, fontFamily: fonts.heading, marginBottom: spacing.md, textAlign: 'center' },
  body: { fontSize: 13, fontFamily: fonts.body, lineHeight: 20, marginBottom: spacing.xl },
  btnRow: { flexDirection: 'row', gap: spacing.sm },
  btn: { flex: 1, paddingVertical: spacing.md, borderRadius: borderRadius.lg, alignItems: 'center' },
  btnText: { fontSize: 14, fontFamily: fonts.heading },
});

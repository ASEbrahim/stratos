import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { ChevronLeft, MoreVertical } from 'lucide-react-native';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay } from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

export interface MenuAction {
  text: string;
  style?: 'cancel' | 'default' | 'destructive';
  onPress?: () => void;
}

interface SessionHeaderProps {
  characterName: string;
  accentColor?: string;
  characterId?: string;
  isTyping?: boolean;
  onNewSession?: () => void;
  onClearHistory?: () => void;
  onExportChat?: () => void;
  onMenuOpen?: (actions: MenuAction[]) => void;
}

export function SessionHeader({ characterName, accentColor, characterId, isTyping, onNewSession, onClearHistory, onExportChat, onMenuOpen }: SessionHeaderProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const color = accentColor ?? tc.accent.primary;
  const pulseOpacity = useSharedValue(1);
  useEffect(() => { pulseOpacity.value = withRepeat(withDelay(1000, withTiming(0.3, { duration: 1000 })), -1, true); }, []);
  const pulseStyle = useAnimatedStyle(() => ({ opacity: pulseOpacity.value }));

  const handleMenu = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const actions: MenuAction[] = [];
    if (characterId) actions.push({ text: 'Character Info', onPress: () => router.push(`/character/${characterId}`) });
    if (onNewSession) actions.push({ text: 'New Session', onPress: onNewSession });
    if (onExportChat) actions.push({ text: 'Export Chat', onPress: onExportChat });
    if (onClearHistory) actions.push({ text: 'Clear History', style: 'destructive', onPress: onClearHistory });
    actions.push({ text: 'Cancel', style: 'cancel' });
    if (onMenuOpen) {
      onMenuOpen(actions);
    } else {
      Alert.alert(characterName, undefined, actions);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: tc.bg.primary, borderBottomColor: tc.border.subtle }]}>
      <TouchableOpacity onPress={() => router.back()} style={styles.backButton} accessibilityLabel="Go back" accessibilityRole="button"><ChevronLeft size={24} color={tc.text.primary} /></TouchableOpacity>
      <View style={styles.center}>
        <View>
          <View style={[styles.avatar, { backgroundColor: color + '20' }]}><Text style={[styles.avatarText, { color }]}>{characterName[0]}</Text></View>
          <Animated.View style={[styles.onlineDot, { backgroundColor: tc.status.success }, pulseStyle]} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={[styles.name, { color: tc.text.primary }]} numberOfLines={1}>{characterName}</Text>
          {isTyping && <Text style={[styles.typingText, { color: tc.status.success }]}>typing...</Text>}
        </View>
      </View>
      <TouchableOpacity style={styles.menuButton} onPress={handleMenu} accessibilityLabel="Chat options menu" accessibilityRole="button"><MoreVertical size={20} color={tc.text.secondary} /></TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.md, borderBottomWidth: 1 },
  backButton: { padding: spacing.xs },
  center: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingHorizontal: spacing.sm },
  avatar: { width: 32, height: 32, borderRadius: 16, justifyContent: 'center', alignItems: 'center' },
  avatarText: { fontSize: 14, fontFamily: fonts.heading },
  name: { ...typography.subheading, fontFamily: fonts.heading },
  typingText: { fontSize: 10, fontFamily: fonts.bodyMedium, marginTop: 1 },
  menuButton: { padding: spacing.xs },
  onlineDot: { position: 'absolute', bottom: 0, right: 0, width: 10, height: 10, borderRadius: 5, borderWidth: 2, borderColor: '#0a0a0f' },
});

import React, { useState } from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { ThumbsUp, ThumbsDown } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing } from '../../constants/theme';
import { sendFeedback } from '../../lib/rp';
import { reportError } from '../../lib/utils';

interface FeedbackButtonsProps {
  messageId: string;
  dbId?: number;  // Backend rp_messages.id
  accentColor?: string;
}

export const FeedbackButtons = React.memo(function FeedbackButtons({ messageId, dbId, accentColor }: FeedbackButtonsProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;
  const [selected, setSelected] = useState<'up' | 'down' | null>(null);

  const handleFeedback = async (type: 'up' | 'down') => {
    if (selected === type) return;
    setSelected(type);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    try {
      await sendFeedback(dbId || 0, type === 'up' ? 'thumbs_up' : 'thumbs_down');
    } catch (err) { reportError('FeedbackButtons:sendFeedback', err); }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPress={() => handleFeedback('up')}
        style={[styles.btn, selected === 'up' && { backgroundColor: tc.status.success + '20' }]}
        activeOpacity={0.6}
        hitSlop={8}
      >
        <ThumbsUp size={12} color={selected === 'up' ? tc.status.success : tc.text.faint} fill={selected === 'up' ? tc.status.success : 'transparent'} />
      </TouchableOpacity>
      <TouchableOpacity
        onPress={() => handleFeedback('down')}
        style={[styles.btn, selected === 'down' && { backgroundColor: tc.status.error + '20' }]}
        activeOpacity={0.6}
        hitSlop={8}
      >
        <ThumbsDown size={12} color={selected === 'down' ? tc.status.error : tc.text.faint} fill={selected === 'down' ? tc.status.error : 'transparent'} />
      </TouchableOpacity>
    </View>
  );
});

const styles = StyleSheet.create({
  container: { flexDirection: 'row', gap: 4, paddingLeft: spacing.lg, marginTop: 2 },
  btn: { padding: 4, borderRadius: 8 },
});

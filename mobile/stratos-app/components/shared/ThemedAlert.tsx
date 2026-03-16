import React from 'react';
import { View, Text, TouchableOpacity, Modal, StyleSheet, TouchableWithoutFeedback } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

export interface AlertButton {
  text: string;
  style?: 'default' | 'cancel' | 'destructive';
  onPress?: () => void;
}

interface ThemedAlertProps {
  visible: boolean;
  title: string;
  message?: string;
  buttons?: AlertButton[];
  onDismiss: () => void;
}

export function ThemedAlert({ visible, title, message, buttons, onDismiss }: ThemedAlertProps) {
  const tc = useThemeStore(s => s.colors);

  const alertButtons = buttons || [{ text: 'OK', onPress: onDismiss }];

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onDismiss}>
      <TouchableWithoutFeedback onPress={onDismiss}>
        <View style={styles.overlay}>
          <TouchableWithoutFeedback>
            <View style={[styles.container, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]}>
              <Text style={[styles.title, { color: tc.text.primary }]}>{title}</Text>
              {message ? <Text style={[styles.message, { color: tc.text.secondary }]}>{message}</Text> : null}
              <View style={[styles.buttonRow, alertButtons.length > 3 && styles.buttonColumn, alertButtons.length === 1 && { justifyContent: 'center' }]}>
                {alertButtons.filter(b => b.style !== 'cancel').map((btn, i) => {
                  const isDestructive = btn.style === 'destructive';
                  return (
                    <TouchableOpacity
                      key={i}
                      style={[
                        styles.button,
                        alertButtons.length <= 3 && alertButtons.length > 1 && { flex: 1 },
                        isDestructive && { backgroundColor: tc.status.error + '15' },
                        !isDestructive && { backgroundColor: tc.accent.primary + '10' },
                      ]}
                      onPress={() => { btn.onPress?.(); onDismiss(); }}
                      activeOpacity={0.7}
                    >
                      <Text style={[
                        styles.buttonText,
                        isDestructive && { color: tc.status.error },
                        !isDestructive && { color: tc.accent.primary },
                      ]}>
                        {btn.text}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
              {/* Cancel button always at bottom, full width */}
              {alertButtons.some(b => b.style === 'cancel') && (
                <TouchableOpacity
                  style={[styles.cancelButton, { backgroundColor: tc.bg.tertiary }]}
                  onPress={onDismiss}
                  activeOpacity={0.7}
                >
                  <Text style={[styles.buttonText, { color: tc.text.muted }]}>
                    {alertButtons.find(b => b.style === 'cancel')?.text ?? 'Cancel'}
                  </Text>
                </TouchableOpacity>
              )}
            </View>
          </TouchableWithoutFeedback>
        </View>
      </TouchableWithoutFeedback>
    </Modal>
  );
}

/**
 * Hook for managing themed alerts.
 * Usage:
 *   const { alert, AlertComponent } = useThemedAlert();
 *   alert('Title', 'Message', [{ text: 'OK' }]);
 *   return <>{AlertComponent}</>
 */
export function useThemedAlert() {
  const [state, setState] = React.useState<{ visible: boolean; title: string; message?: string; buttons?: AlertButton[] }>({
    visible: false, title: '',
  });

  const alert = (title: string, message?: string, buttons?: AlertButton[]) => {
    setState({ visible: true, title, message, buttons });
  };

  const dismiss = () => setState(prev => ({ ...prev, visible: false }));

  const AlertComponent = (
    <ThemedAlert
      visible={state.visible}
      title={state.title}
      message={state.message}
      buttons={state.buttons}
      onDismiss={dismiss}
    />
  );

  return { alert, AlertComponent };
}

const styles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center', padding: spacing.xl },
  container: { width: '100%', maxWidth: 340, borderRadius: borderRadius.xl, padding: spacing.xl, borderWidth: 1, gap: spacing.md },
  title: { fontSize: 18, fontFamily: fonts.heading, textAlign: 'center' },
  message: { fontSize: 14, fontFamily: fonts.body, textAlign: 'center', lineHeight: 20 },
  buttonRow: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm },
  buttonColumn: { flexDirection: 'column' },
  button: { paddingVertical: spacing.md, paddingHorizontal: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center' },
  cancelButton: { paddingVertical: spacing.md, borderRadius: borderRadius.lg, alignItems: 'center' },
  buttonText: { fontSize: 15, fontFamily: fonts.heading },
});

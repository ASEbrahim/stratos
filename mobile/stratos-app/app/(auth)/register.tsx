import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { useAuthStore } from '../../stores/authStore';
import { StarParallax } from '../../components/shared/StarParallax';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function RegisterScreen() {
  const router = useRouter();
  const { register, isLoading } = useAuthStore();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const handleRegister = async () => {
    if (!name.trim() || !email.trim() || !password.trim()) { Alert.alert('Error', 'Fill in all fields.'); return; }
    if (password.length < 6) { Alert.alert('Error', 'Password must be at least 6 characters.'); return; }
    try { await register(name.trim(), email.trim(), password); router.replace('/(tabs)/discover'); }
    catch { Alert.alert('Registration Failed', 'Please try again.'); }
  };
  return (
    <StarParallax>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <View style={styles.content}>
          <View style={styles.header}><Text style={styles.logo}>StratOS</Text><Text style={styles.subtitle}>Create your account</Text></View>
          <View style={styles.form}>
            <TextInput style={styles.input} value={name} onChangeText={setName} placeholder="Display Name" placeholderTextColor={colors.text.muted} autoCapitalize="words" />
            <TextInput style={styles.input} value={email} onChangeText={setEmail} placeholder="Email" placeholderTextColor={colors.text.muted} keyboardType="email-address" autoCapitalize="none" />
            <TextInput style={styles.input} value={password} onChangeText={setPassword} placeholder="Password" placeholderTextColor={colors.text.muted} secureTextEntry />
            <TouchableOpacity style={[styles.button, isLoading && { opacity: 0.6 }]} onPress={handleRegister} disabled={isLoading} activeOpacity={0.8}>
              <Text style={styles.buttonText}>{isLoading ? 'Creating...' : 'Create Account'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.linkButton} onPress={() => router.back()}>
              <Text style={styles.linkText}>Already have an account? <Text style={styles.linkHL}>Sign In</Text></Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </StarParallax>
  );
}

const styles = StyleSheet.create({
  content: { flex: 1, justifyContent: 'center', padding: spacing.xxl },
  header: { alignItems: 'center', marginBottom: spacing.xxl * 2 },
  logo: { fontSize: 36, fontWeight: '800', color: colors.text.primary, letterSpacing: -1, textShadowColor: 'rgba(79, 168, 212, 0.3)', textShadowOffset: { width: 0, height: 0 }, textShadowRadius: 20 },
  subtitle: { ...typography.body, color: colors.text.secondary, marginTop: spacing.xs },
  form: { gap: spacing.md },
  input: { backgroundColor: 'rgba(21, 24, 40, 0.7)', borderRadius: borderRadius.lg, paddingHorizontal: spacing.lg, paddingVertical: spacing.lg, color: colors.text.primary, fontSize: 16, borderWidth: 1, borderColor: 'rgba(79, 168, 212, 0.12)' },
  button: { backgroundColor: colors.accent.primary, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginTop: spacing.md, shadowColor: colors.accent.primary, shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8 },
  buttonText: { ...typography.subheading, color: '#fff', fontSize: 16 },
  linkButton: { alignItems: 'center', paddingVertical: spacing.md },
  linkText: { ...typography.body, color: colors.text.secondary },
  linkHL: { color: colors.accent.primary, fontWeight: '600' },
});

import React, { useState, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  KeyboardAvoidingView, Platform, Alert, ScrollView,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuthStore } from '../../stores/authStore';
import { StarParallax } from '../../components/shared/StarParallax';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function RegisterScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { register, isLoading } = useAuthStore();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const emailRef = useRef<TextInput>(null);
  const passwordRef = useRef<TextInput>(null);

  const handleRegister = async () => {
    if (!name.trim() || !email.trim() || !password.trim()) {
      Alert.alert('Error', 'Fill in all fields.');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters.');
      return;
    }
    try {
      await register(name.trim(), email.trim(), password);
      router.replace('/(tabs)/discover');
    } catch {
      Alert.alert('Registration Failed', 'Please try again.');
    }
  };

  return (
    <StarParallax>
      <KeyboardAvoidingView
        style={[styles.container, { paddingTop: insets.top, paddingBottom: insets.bottom }]}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <Text style={styles.logo}>StratOS</Text>
            <Text style={styles.subtitle}>Create your account</Text>
          </View>

          <View style={styles.form}>
            <TextInput style={styles.input} value={name} onChangeText={setName} placeholder="Display Name" placeholderTextColor={colors.text.muted} autoCapitalize="words" returnKeyType="next" onSubmitEditing={() => emailRef.current?.focus()} />
            <TextInput ref={emailRef} style={styles.input} value={email} onChangeText={setEmail} placeholder="Email" placeholderTextColor={colors.text.muted} keyboardType="email-address" autoCapitalize="none" autoCorrect={false} returnKeyType="next" onSubmitEditing={() => passwordRef.current?.focus()} />
            <TextInput ref={passwordRef} style={styles.input} value={password} onChangeText={setPassword} placeholder="Password (6+ characters)" placeholderTextColor={colors.text.muted} secureTextEntry returnKeyType="go" onSubmitEditing={handleRegister} />

            <TouchableOpacity style={[styles.button, isLoading && styles.buttonDisabled]} onPress={handleRegister} disabled={isLoading} activeOpacity={0.8}>
              <Text style={styles.buttonText}>{isLoading ? 'Creating...' : 'Create Account'}</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.linkButton} onPress={() => router.back()}>
              <Text style={styles.linkText}>Already have an account? <Text style={styles.linkHL}>Sign In</Text></Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </StarParallax>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollContent: { flexGrow: 1, justifyContent: 'center', padding: spacing.xxl },
  header: { alignItems: 'center', marginBottom: 48 },
  logo: { fontSize: 40, fontWeight: '800', color: colors.text.primary, letterSpacing: -1.5, textShadowColor: 'rgba(79, 168, 212, 0.35)', textShadowOffset: { width: 0, height: 0 }, textShadowRadius: 24 },
  subtitle: { ...typography.body, color: colors.text.secondary, marginTop: spacing.sm },
  form: { gap: spacing.lg },
  input: { backgroundColor: 'rgba(21, 24, 40, 0.75)', borderRadius: borderRadius.lg, paddingHorizontal: spacing.xl, paddingVertical: 18, color: colors.text.primary, fontSize: 16, borderWidth: 1, borderColor: 'rgba(79, 168, 212, 0.15)' },
  button: { backgroundColor: colors.accent.primary, paddingVertical: 18, borderRadius: borderRadius.lg, alignItems: 'center', marginTop: spacing.sm, shadowColor: colors.accent.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8 },
  buttonDisabled: { opacity: 0.5 },
  buttonText: { fontSize: 17, fontWeight: '700', color: '#fff', letterSpacing: 0.3 },
  linkButton: { alignItems: 'center', paddingVertical: spacing.lg },
  linkText: { ...typography.body, color: colors.text.secondary },
  linkHL: { color: colors.accent.primary, fontWeight: '600' },
});

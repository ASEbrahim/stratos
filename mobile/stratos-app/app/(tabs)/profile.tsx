import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { LogOut, Moon, Type, Bell, Server, Shield } from 'lucide-react-native';
import { useAuthStore } from '../../stores/authStore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function ProfileScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const handleLogout = () => Alert.alert('Sign Out', 'Are you sure?', [{ text: 'Cancel', style: 'cancel' }, { text: 'Sign Out', style: 'destructive', onPress: async () => { await logout(); router.replace('/(auth)/login'); } }]);
  return (
    <ScrollView style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <View style={styles.avatar}><Text style={styles.avatarText}>{user?.name?.[0] ?? '?'}</Text></View>
        <Text style={styles.name}>{user?.name ?? 'User'}</Text>
        <Text style={styles.email}>{user?.email ?? ''}</Text>
      </View>
      <View style={styles.statsRow}>
        <View style={styles.stat}><Text style={styles.statVal}>0</Text><Text style={styles.statLbl}>Sessions</Text></View>
        <View style={styles.statDiv} />
        <View style={styles.stat}><Text style={styles.statVal}>0</Text><Text style={styles.statLbl}>Characters</Text></View>
        <View style={styles.statDiv} />
        <View style={styles.stat}><Text style={styles.statVal}>0</Text><Text style={styles.statLbl}>Messages</Text></View>
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Settings</Text>
        <SettingsItem icon={<Shield size={20} color={colors.text.secondary} />} label="Content Filter" value="SFW Only" />
        <SettingsItem icon={<Moon size={20} color={colors.text.secondary} />} label="Theme" value="Arcane" />
        <SettingsItem icon={<Type size={20} color={colors.text.secondary} />} label="Text Size" value="Medium" />
        <SettingsItem icon={<Bell size={20} color={colors.text.secondary} />} label="Notifications" value="On" />
        <SettingsItem icon={<Server size={20} color={colors.text.secondary} />} label="API Server" value="Default" />
      </View>
      <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
        <LogOut size={18} color={colors.status.error} /><Text style={styles.logoutText}>Sign Out</Text>
      </TouchableOpacity>
      <View style={{ height: spacing.xxl * 2 }} />
    </ScrollView>
  );
}

function SettingsItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <TouchableOpacity style={styles2.item} activeOpacity={0.6}>
      {icon}<Text style={styles2.label}>{label}</Text><Text style={styles2.value}>{value}</Text>
    </TouchableOpacity>
  );
}
const styles2 = StyleSheet.create({
  item: { flexDirection: 'row', alignItems: 'center', paddingVertical: spacing.lg, borderBottomWidth: 1, borderBottomColor: colors.border.subtle, gap: spacing.md },
  label: { ...typography.body, color: colors.text.primary, flex: 1 },
  value: { ...typography.body, color: colors.text.muted },
});

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  header: { alignItems: 'center', paddingVertical: spacing.xxl },
  avatar: { width: 80, height: 80, borderRadius: 40, backgroundColor: colors.accent.primary + '20', justifyContent: 'center', alignItems: 'center', marginBottom: spacing.md },
  avatarText: { fontSize: 32, fontWeight: '700', color: colors.accent.primary },
  name: { ...typography.heading, color: colors.text.primary },
  email: { ...typography.body, color: colors.text.secondary, marginTop: spacing.xs },
  statsRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', backgroundColor: colors.bg.secondary, marginHorizontal: spacing.lg, borderRadius: borderRadius.lg, paddingVertical: spacing.lg, marginBottom: spacing.xxl },
  stat: { flex: 1, alignItems: 'center' },
  statVal: { ...typography.heading, color: colors.text.primary },
  statLbl: { ...typography.small, color: colors.text.muted, marginTop: 2 },
  statDiv: { width: 1, height: 30, backgroundColor: colors.border.subtle },
  section: { paddingHorizontal: spacing.lg, marginBottom: spacing.lg },
  sectionTitle: { ...typography.subheading, color: colors.text.muted, marginBottom: spacing.md, textTransform: 'uppercase', letterSpacing: 1, fontSize: 12 },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, marginHorizontal: spacing.lg, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1, borderColor: colors.status.error + '40', marginTop: spacing.lg },
  logoutText: { ...typography.subheading, color: colors.status.error },
});

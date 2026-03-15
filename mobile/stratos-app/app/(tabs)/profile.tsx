import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { LogOut, Type, Bell, Server, Shield } from 'lucide-react-native';
import { useAuthStore } from '../../stores/authStore';
import { useThemeStore } from '../../stores/themeStore';
import { useChatStore } from '../../stores/chatStore';
import { getGenreColor } from '../../constants/genres';
import { getDetailedStats, DetailedStats } from '../../lib/storage';
import { formatCount } from '../../lib/types';
import { THEMES } from '../../constants/themes';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function ProfileScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const { themeId, setTheme, colors: tc, nsfwFilter, setNsfwFilter } = useThemeStore();
  const { recentSessions, loadRecentSessions, resumeSession } = useChatStore();
  const [stats, setStats] = useState<DetailedStats>({ totalSessions: 0, totalMessages: 0, totalWords: 0, avgSessionLength: 0, favoriteGenre: 'None yet', longestSession: 0, totalCharacters: 0 });

  useEffect(() => { getDetailedStats().then(setStats); loadRecentSessions(); }, []);

  const handleLogout = () => Alert.alert('Sign Out', 'Are you sure?', [{ text: 'Cancel', style: 'cancel' }, { text: 'Sign Out', style: 'destructive', onPress: async () => { await logout(); router.replace('/(auth)/login'); } }]);

  const handleThemeChange = (id: string) => {
    setTheme(id);
    Haptics.selectionAsync();
  };

  return (
    <ScrollView style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <View style={styles.header}>
        <View style={[styles.avatar, { backgroundColor: tc.accent.primary + '20' }]}><Text style={[styles.avatarText, { color: tc.accent.primary }]}>{user?.name?.[0] ?? '?'}</Text></View>
        <Text style={[styles.name, { color: tc.text.primary }]}>{user?.name ?? 'User'}</Text>
        <Text style={[styles.email, { color: tc.text.secondary }]}>{user?.email ?? ''}</Text>
      </View>
      <View style={[styles.statsRow, { backgroundColor: tc.bg.secondary }]}>
        <View style={styles.stat}><Text style={[styles.statVal, { color: tc.text.primary }]}>{formatCount(stats.totalSessions)}</Text><Text style={[styles.statLbl, { color: tc.text.muted }]}>Sessions</Text></View>
        <View style={[styles.statDiv, { backgroundColor: tc.border.subtle }]} />
        <View style={styles.stat}><Text style={[styles.statVal, { color: tc.text.primary }]}>{formatCount(stats.totalCharacters)}</Text><Text style={[styles.statLbl, { color: tc.text.muted }]}>Characters</Text></View>
        <View style={[styles.statDiv, { backgroundColor: tc.border.subtle }]} />
        <View style={styles.stat}><Text style={[styles.statVal, { color: tc.text.primary }]}>{formatCount(stats.totalMessages)}</Text><Text style={[styles.statLbl, { color: tc.text.muted }]}>Messages</Text></View>
      </View>

      {/* Detailed Stats */}
      {(stats.totalWords > 0 || stats.totalSessions > 0) && (
        <View style={[styles.section, { marginBottom: spacing.md }]}>
          <Text style={[styles.sectionTitle, { color: tc.text.muted }]}>Your Journey</Text>
          <View style={[styles.detailStatsGrid, { backgroundColor: tc.bg.secondary, borderRadius: borderRadius.lg }]}>
            <View style={styles.detailStat}>
              <Text style={[styles.detailVal, { color: tc.accent.primary }]}>{formatCount(stats.totalWords)}</Text>
              <Text style={[styles.detailLbl, { color: tc.text.muted }]}>Words Written</Text>
            </View>
            <View style={styles.detailStat}>
              <Text style={[styles.detailVal, { color: tc.accent.primary }]}>{stats.avgSessionLength}</Text>
              <Text style={[styles.detailLbl, { color: tc.text.muted }]}>Avg Messages</Text>
            </View>
            <View style={styles.detailStat}>
              <Text style={[styles.detailVal, { color: tc.accent.primary }]}>{stats.longestSession}</Text>
              <Text style={[styles.detailLbl, { color: tc.text.muted }]}>Longest Chat</Text>
            </View>
            <View style={styles.detailStat}>
              <Text style={[styles.detailVal, { color: tc.accent.primary }]}>{stats.favoriteGenre}</Text>
              <Text style={[styles.detailLbl, { color: tc.text.muted }]}>Top Genre</Text>
            </View>
          </View>
        </View>
      )}

      {/* Top Characters */}
      {recentSessions.length > 0 && (
        <View style={[styles.section, { marginBottom: spacing.md }]}>
          <Text style={[styles.sectionTitle, { color: tc.text.muted }]}>Top Characters</Text>
          <View style={styles.topCharsRow}>
            {recentSessions
              .sort((a, b) => b.messages.length - a.messages.length)
              .slice(0, 3)
              .map(s => (
                <TouchableOpacity key={s.id} style={styles.topChar} onPress={() => { resumeSession(s); router.push(`/chat/${s.character_id}`); }} activeOpacity={0.7} accessibilityLabel={`Resume chat with ${s.character_name}, ${s.messages.length} messages`} accessibilityRole="button">
                  <View style={[styles.topCharAvatar, { backgroundColor: tc.accent.primary + '15' }]}>
                    <Text style={[styles.topCharLetter, { color: tc.accent.primary }]}>{s.character_name[0]}</Text>
                  </View>
                  <Text style={[styles.topCharName, { color: tc.text.primary }]} numberOfLines={1}>{s.character_name.split(' ')[0]}</Text>
                  <Text style={[styles.topCharMsgs, { color: tc.text.muted }]}>{s.messages.length} msgs</Text>
                </TouchableOpacity>
              ))}
          </View>
        </View>
      )}

      {/* Theme selector */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: tc.text.muted }]}>Theme</Text>
        <View style={styles.themeGrid}>
          {THEMES.map(theme => {
            const active = theme.id === themeId;
            return (
              <TouchableOpacity
                key={theme.id}
                style={[
                  styles.themeCard,
                  { backgroundColor: theme.colors.bg.secondary, borderColor: active ? theme.colors.accent.primary : theme.colors.border.subtle },
                  active && { borderWidth: 2, shadowColor: theme.colors.accent.primary, shadowOpacity: 0.3, shadowRadius: 12, elevation: 4 },
                ]}
                onPress={() => handleThemeChange(theme.id)}
                activeOpacity={0.7}
                accessibilityLabel={`${theme.label} theme${active ? ', selected' : ''}`}
                accessibilityRole="button"
              >
                <View style={[styles.themePreview, { backgroundColor: theme.colors.bg.primary }]}>
                  <View style={[styles.themeAccentDot, { backgroundColor: theme.colors.accent.primary }]} />
                  <View style={[styles.themeAccentDot2, { backgroundColor: theme.colors.star.color2 ? `rgb(${theme.colors.star.color2.r},${theme.colors.star.color2.g},${theme.colors.star.color2.b})` : theme.colors.accent.light }]} />
                  <View style={[styles.themeAccentDot3, { backgroundColor: theme.colors.star.color1 ? `rgb(${theme.colors.star.color1.r},${theme.colors.star.color1.g},${theme.colors.star.color1.b})` : theme.colors.accent.primary }]} />
                </View>
                <Text style={[styles.themeIcon]}>{theme.icon}</Text>
                <Text style={[styles.themeName, { color: active ? theme.colors.accent.primary : tc.text.secondary }]}>{theme.label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: tc.text.muted }]}>Settings</Text>
        <TouchableOpacity style={[styles2.item, { borderBottomColor: tc.border.subtle }]} activeOpacity={0.6} onPress={() => { setNsfwFilter(!nsfwFilter); Haptics.selectionAsync(); }} accessibilityLabel={`Content filter: ${nsfwFilter ? 'SFW only' : 'all content'}`} accessibilityRole="button">
          <Shield size={20} color={tc.text.secondary} />
          <Text style={[styles2.label, { color: tc.text.primary }]}>Content Filter</Text>
          <Text style={[styles2.value, { color: nsfwFilter ? tc.status.success : tc.status.error }]}>{nsfwFilter ? 'SFW Only' : 'All Content'}</Text>
        </TouchableOpacity>
        <SettingsItem icon={<Type size={20} color={tc.text.secondary} />} label="Text Size" value="Medium" tc={tc} />
        <SettingsItem icon={<Bell size={20} color={tc.text.secondary} />} label="Notifications" value="On" tc={tc} />
        <SettingsItem icon={<Server size={20} color={tc.text.secondary} />} label="API Server" value="Default" tc={tc} />
      </View>
      <TouchableOpacity style={[styles.logoutBtn, { borderColor: tc.status.error + '40' }]} onPress={handleLogout} accessibilityLabel="Sign out" accessibilityRole="button">
        <LogOut size={18} color={tc.status.error} /><Text style={[styles.logoutText, { color: tc.status.error }]}>Sign Out</Text>
      </TouchableOpacity>
      <View style={styles.aboutSection}>
        <Text style={[styles.aboutText, { color: tc.text.muted }]}>StratOS Mobile v1.0.0</Text>
        <Text style={[styles.aboutText, { color: tc.text.faint }]}>Strategic Intelligence Platform</Text>
      </View>
      <View style={{ height: spacing.xxl * 2 }} />
    </ScrollView>
  );
}

function SettingsItem({ icon, label, value, tc }: { icon: React.ReactNode; label: string; value: string; tc: any }) {
  return (
    <TouchableOpacity style={[styles2.item, { borderBottomColor: tc.border.subtle }]} activeOpacity={0.6} accessibilityLabel={`${label}: ${value}`} accessibilityRole="button">
      {icon}<Text style={[styles2.label, { color: tc.text.primary }]}>{label}</Text><Text style={[styles2.value, { color: tc.text.muted }]}>{value}</Text>
    </TouchableOpacity>
  );
}
const styles2 = StyleSheet.create({
  item: { flexDirection: 'row', alignItems: 'center', paddingVertical: spacing.lg, borderBottomWidth: 1, gap: spacing.md },
  label: { ...typography.body, flex: 1 },
  value: { ...typography.body },
});

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { alignItems: 'center', paddingVertical: spacing.xxl },
  avatar: { width: 80, height: 80, borderRadius: 40, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.md },
  avatarText: { fontSize: 32, fontWeight: '700' },
  name: { ...typography.heading },
  email: { ...typography.body, marginTop: spacing.xs },
  statsRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginHorizontal: spacing.lg, borderRadius: borderRadius.lg, paddingVertical: spacing.lg, marginBottom: spacing.xxl },
  stat: { flex: 1, alignItems: 'center' },
  statVal: { ...typography.heading },
  statLbl: { ...typography.small, marginTop: 2 },
  statDiv: { width: 1, height: 30 },
  section: { paddingHorizontal: spacing.lg, marginBottom: spacing.lg },
  sectionTitle: { ...typography.subheading, marginBottom: spacing.md, textTransform: 'uppercase', letterSpacing: 1, fontSize: 12 },
  themeGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  themeCard: { width: '18.5%', aspectRatio: 0.75, borderRadius: borderRadius.md, borderWidth: 1, alignItems: 'center', justifyContent: 'flex-end', paddingBottom: spacing.xs, overflow: 'hidden' },
  themePreview: { ...StyleSheet.absoluteFillObject, borderRadius: borderRadius.md },
  themeAccentDot: { position: 'absolute', top: 8, left: 8, width: 6, height: 6, borderRadius: 3 },
  themeAccentDot2: { position: 'absolute', top: 16, right: 10, width: 4, height: 4, borderRadius: 2, opacity: 0.6 },
  themeAccentDot3: { position: 'absolute', bottom: 20, left: 12, width: 3, height: 3, borderRadius: 1.5, opacity: 0.4 },
  themeIcon: { fontSize: 16, marginBottom: 2, zIndex: 1 },
  themeName: { fontSize: 9, fontWeight: '600', zIndex: 1 },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, marginHorizontal: spacing.lg, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1, marginTop: spacing.lg },
  logoutText: { ...typography.subheading },
  aboutSection: { alignItems: 'center', paddingVertical: spacing.xxl, gap: 4 },
  aboutText: { fontSize: 11 },
  detailStatsGrid: { flexDirection: 'row', flexWrap: 'wrap', padding: spacing.md },
  detailStat: { width: '50%', alignItems: 'center', paddingVertical: spacing.md },
  detailVal: { fontSize: 18, fontWeight: '700', marginBottom: 2 },
  detailLbl: { fontSize: 10, fontWeight: '500' },
  topCharsRow: { flexDirection: 'row', justifyContent: 'space-around', paddingVertical: spacing.md },
  topChar: { alignItems: 'center', width: 80 },
  topCharAvatar: { width: 50, height: 50, borderRadius: 25, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.xs },
  topCharLetter: { fontSize: 20, fontWeight: '700' },
  topCharName: { fontSize: 12, fontWeight: '600', textAlign: 'center' },
  topCharMsgs: { fontSize: 9, marginTop: 2 },
});

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, FlatList } from 'react-native';
import * as Haptics from 'expo-haptics';
import { GitBranch, ChevronDown, X } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import type { Branch } from '../../lib/rp';

interface BranchSelectorProps {
  branches: Branch[];
  currentBranch: string;
  onSelect: (branchId: string) => void;
  accentColor?: string;
}

export function BranchSelector({ branches, currentBranch, onSelect, accentColor }: BranchSelectorProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;
  const [open, setOpen] = useState(false);

  if (branches.length <= 1) return null;

  const current = branches.find(b => b.id === currentBranch);

  return (
    <>
      <TouchableOpacity
        style={[styles.trigger, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}
        onPress={() => { setOpen(true); Haptics.selectionAsync(); }}
        activeOpacity={0.7}
      >
        <GitBranch size={12} color={accent} />
        <Text style={[styles.triggerText, { color: tc.text.secondary }]}>
          {currentBranch === 'main' ? 'Main' : currentBranch.slice(0, 12)}
        </Text>
        <Text style={[styles.branchCount, { color: tc.text.muted }]}>{branches.length}</Text>
        <ChevronDown size={12} color={tc.text.muted} />
      </TouchableOpacity>

      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <TouchableOpacity style={styles.overlay} activeOpacity={1} onPress={() => setOpen(false)}>
          <View style={[styles.dropdown, { backgroundColor: tc.bg.primary, borderColor: tc.border.subtle }]}>
            <View style={styles.dropdownHeader}>
              <Text style={[styles.dropdownTitle, { color: tc.text.primary }]}>Branches</Text>
              <TouchableOpacity onPress={() => setOpen(false)} hitSlop={8}>
                <X size={18} color={tc.text.muted} />
              </TouchableOpacity>
            </View>
            <FlatList
              data={branches}
              keyExtractor={b => b.id}
              renderItem={({ item: b }) => (
                <TouchableOpacity
                  style={[styles.branchItem, { borderBottomColor: tc.border.subtle },
                    b.id === currentBranch && { backgroundColor: accent + '10' }]}
                  onPress={() => { onSelect(b.id); setOpen(false); Haptics.selectionAsync(); }}
                >
                  <View style={styles.branchInfo}>
                    <Text style={[styles.branchName, { color: b.id === currentBranch ? accent : tc.text.primary }]}>
                      {b.id === 'main' ? 'Main timeline' : b.id.slice(0, 16)}
                    </Text>
                    <Text style={[styles.branchMeta, { color: tc.text.muted }]}>
                      {b.turn_count} turns{b.branch_point_turn != null ? ` · from turn ${b.branch_point_turn}` : ''}
                    </Text>
                  </View>
                  {b.id === currentBranch && (
                    <View style={[styles.activeDot, { backgroundColor: accent }]} />
                  )}
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  trigger: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, paddingHorizontal: spacing.md, paddingVertical: spacing.xs, borderRadius: borderRadius.full, borderWidth: 1, alignSelf: 'center', marginVertical: spacing.xs },
  triggerText: { fontSize: 11, fontFamily: fonts.button },
  branchCount: { fontSize: 9, fontFamily: fonts.body },
  overlay: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.5)' },
  dropdown: { width: '80%', maxHeight: '60%', borderRadius: borderRadius.lg, borderWidth: 1, overflow: 'hidden' },
  dropdownHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.05)' },
  dropdownTitle: { fontSize: 16, fontFamily: fonts.heading },
  branchItem: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderBottomWidth: 1 },
  branchInfo: { flex: 1, gap: 2 },
  branchName: { fontSize: 13, fontFamily: fonts.button },
  branchMeta: { fontSize: 10, fontFamily: fonts.body },
  activeDot: { width: 8, height: 8, borderRadius: 4 },
});

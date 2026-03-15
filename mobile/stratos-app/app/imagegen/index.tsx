import React, { useState } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Wand2, Download, ImageIcon } from 'lucide-react-native';
import { Header } from '../../components/shared/Header';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { generateImage, generateCharacterPortrait, getImageUrl } from '../../lib/rp';

type Style = 'anime' | 'realistic' | 'illustration';
type Size = 'portrait' | 'square' | 'landscape';

const SIZES: Record<Size, { w: number; h: number; label: string }> = {
  portrait: { w: 768, h: 1024, label: 'Portrait' },
  square: { w: 1024, h: 1024, label: 'Square' },
  landscape: { w: 1024, h: 768, label: 'Landscape' },
};

export default function ImageGenScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const params = useLocalSearchParams<{ name?: string; description?: string; card_id?: string }>();

  const [prompt, setPrompt] = useState(params.description || '');
  const [style, setStyle] = useState<Style>('anime');
  const [size, setSize] = useState<Size>('portrait');
  const [nsfw, setNsfw] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [imageId, setImageId] = useState<string | null>(null);
  const [error, setError] = useState('');

  const isCharacterMode = !!(params.name && params.description);

  const handleGenerate = async () => {
    if (!prompt.trim() && !isCharacterMode) return;
    setGenerating(true);
    setError('');
    setImageId(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      let result: { success: boolean; image_id?: string; error?: string };
      if (isCharacterMode) {
        result = await generateCharacterPortrait({
          character_name: params.name!,
          physical_description: params.description!,
          style,
          nsfw,
          character_card_id: params.card_id,
        });
      } else {
        const s = SIZES[size];
        result = await generateImage({
          prompt: nsfw ? `score_9, score_8_up, score_7_up, ${prompt.trim()}, rating_explicit` : prompt.trim(),
          model: nsfw ? 'pony' : 'flux',
          width: s.w,
          height: s.h,
        });
      }

      if (result.success && result.image_id) {
        setImageId(result.image_id);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else {
        setError(result.error || 'Generation failed');
      }
    } catch (e: any) {
      setError(e.message || 'Connection failed — is the server running?');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title={isCharacterMode ? 'Generate Portrait' : 'Image Generation'} showBack />
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">

        {/* Prompt input */}
        {isCharacterMode ? (
          <View style={[styles.charInfo, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}>
            <Text style={[styles.charName, { color: tc.accent.primary }]}>{params.name}</Text>
            <Text style={[styles.charDesc, { color: tc.text.secondary }]} numberOfLines={3}>{params.description}</Text>
          </View>
        ) : (
          <>
            <Text style={[styles.label, { color: tc.text.primary }]}>Prompt</Text>
            <TextInput
              style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]}
              value={prompt}
              onChangeText={setPrompt}
              placeholder="Describe what you want to generate..."
              placeholderTextColor={tc.text.muted}
              multiline
              textAlignVertical="top"
            />
          </>
        )}

        {/* Style selector */}
        <Text style={[styles.label, { color: tc.text.primary }]}>Style</Text>
        <View style={styles.chipRow}>
          {(['anime', 'realistic', 'illustration'] as Style[]).map(s => (
            <TouchableOpacity
              key={s}
              style={[styles.chip, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle },
                style === s && { backgroundColor: tc.accent.primary + '20', borderColor: tc.accent.primary }]}
              onPress={() => { setStyle(s); Haptics.selectionAsync(); }}
            >
              <Text style={[styles.chipText, { color: tc.text.secondary },
                style === s && { color: tc.accent.primary }]}>
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Size selector (free-form only) */}
        {!isCharacterMode && (
          <>
            <Text style={[styles.label, { color: tc.text.primary }]}>Size</Text>
            <View style={styles.chipRow}>
              {(Object.entries(SIZES) as [Size, typeof SIZES[Size]][]).map(([key, val]) => (
                <TouchableOpacity
                  key={key}
                  style={[styles.chip, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle },
                    size === key && { backgroundColor: tc.accent.primary + '20', borderColor: tc.accent.primary }]}
                  onPress={() => { setSize(key); Haptics.selectionAsync(); }}
                >
                  <Text style={[styles.chipText, { color: tc.text.secondary },
                    size === key && { color: tc.accent.primary }]}>
                    {val.label} ({val.w}x{val.h})
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </>
        )}

        {/* Model toggle */}
        <View style={[styles.modelToggle, { borderColor: tc.border.subtle }]}>
          <TouchableOpacity
            style={[styles.modelBtn, !nsfw && { backgroundColor: tc.accent.primary + '20' }]}
            onPress={() => { setNsfw(false); Haptics.selectionAsync(); }}
          >
            <Text style={[styles.chipText, { color: !nsfw ? tc.accent.primary : tc.text.muted }]}>FLUX (SFW)</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.modelBtn, nsfw && { backgroundColor: tc.status.error + '20' }]}
            onPress={() => { setNsfw(true); Haptics.selectionAsync(); }}
          >
            <Text style={[styles.chipText, { color: nsfw ? tc.status.error : tc.text.muted }]}>Pony V7 (NSFW)</Text>
          </TouchableOpacity>
        </View>
        {nsfw && <Text style={[styles.nsfwNote, { color: tc.text.muted }]}>Pony V7 understands booru tags, explicit content terms, and anime-specific anatomy. Quality tags (score_9, etc.) are auto-prepended.</Text>}

        {/* Generate button */}
        <TouchableOpacity
          style={[styles.generateBtn, { backgroundColor: tc.accent.primary }, generating && { opacity: 0.6 }]}
          onPress={handleGenerate}
          disabled={generating || (!prompt.trim() && !isCharacterMode)}
          activeOpacity={0.7}
        >
          {generating ? (
            <>
              <ActivityIndicator size={18} color="#fff" />
              <Text style={styles.generateText}>Generating... ~5-30s</Text>
            </>
          ) : (
            <>
              <Wand2 size={18} color="#fff" />
              <Text style={styles.generateText}>Generate</Text>
            </>
          )}
        </TouchableOpacity>

        {/* Error */}
        {error ? (
          <Text style={[styles.error, { color: tc.status.error }]}>{error}</Text>
        ) : null}

        {/* Result */}
        {imageId && (
          <View style={styles.resultContainer}>
            <Image
              source={{ uri: getImageUrl(imageId) }}
              style={[styles.resultImage, { borderColor: tc.border.subtle }]}
              contentFit="contain"
            />
            <Text style={[styles.imageIdText, { color: tc.text.muted }]}>ID: {imageId}</Text>
          </View>
        )}

        {/* Placeholder when no image */}
        {!imageId && !generating && (
          <View style={[styles.placeholder, { borderColor: tc.border.subtle }]}>
            <ImageIcon size={40} color={tc.text.faint} />
            <Text style={[styles.placeholderText, { color: tc.text.muted }]}>
              {isCharacterMode ? 'Generate a portrait for your character' : 'Your generated image will appear here'}
            </Text>
          </View>
        )}

        <View style={{ height: spacing.xxl * 2 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg },
  label: { fontFamily: fonts.heading, fontSize: 14, marginTop: spacing.lg, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, fontFamily: fonts.body, minHeight: 100, borderWidth: 1, textAlignVertical: 'top' },
  chipRow: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
  chip: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  chipText: { fontSize: 13, fontFamily: fonts.button },
  modelToggle: { flexDirection: 'row', borderRadius: borderRadius.lg, borderWidth: 1, overflow: 'hidden', marginTop: spacing.lg },
  modelBtn: { flex: 1, paddingVertical: spacing.md, alignItems: 'center' },
  nsfwNote: { fontSize: 10, fontFamily: fonts.body, lineHeight: 15, marginTop: spacing.xs },
  generateBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, marginTop: spacing.lg },
  generateText: { fontSize: 16, fontFamily: fonts.heading, color: '#fff' },
  error: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', marginTop: spacing.md },
  resultContainer: { marginTop: spacing.lg, alignItems: 'center' },
  resultImage: { width: '100%', aspectRatio: 3 / 4, borderRadius: borderRadius.lg, borderWidth: 1 },
  imageIdText: { fontSize: 10, fontFamily: fonts.body, marginTop: spacing.xs },
  placeholder: { marginTop: spacing.xl, alignItems: 'center', justifyContent: 'center', paddingVertical: spacing.xxl * 2, borderWidth: 1, borderStyle: 'dashed', borderRadius: borderRadius.lg, gap: spacing.md },
  placeholderText: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', paddingHorizontal: spacing.xl },
  charInfo: { borderRadius: borderRadius.lg, padding: spacing.lg, borderWidth: 1, gap: spacing.xs },
  charName: { fontSize: 18, fontFamily: fonts.heading },
  charDesc: { fontSize: 13, fontFamily: fonts.body, lineHeight: 20 },
});

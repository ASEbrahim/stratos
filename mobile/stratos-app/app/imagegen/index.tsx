import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, StyleSheet, ActivityIndicator, Alert, Dimensions, Platform } from 'react-native';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { cacheDirectory, downloadAsync } from 'expo-file-system/legacy';
import * as MediaLibrary from 'expo-media-library';
import { Wand2, Download, ImageIcon, Trash2, ChevronDown, ChevronUp, Shuffle, Grid3X3 } from 'lucide-react-native';
import { Header } from '../../components/shared/Header';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { generateImage, generateCharacterPortrait, getImageUrl, getImageGallery, deleteImage } from '../../lib/rp';

type Style = 'anime' | 'realistic' | 'illustration';
type Size = 'portrait' | 'square' | 'landscape';

const SIZES: Record<Size, { w: number; h: number; label: string }> = {
  portrait: { w: 768, h: 1024, label: 'Portrait' },
  square: { w: 1024, h: 1024, label: 'Square' },
  landscape: { w: 1024, h: 768, label: 'Landscape' },
};

const SCREEN_WIDTH = Dimensions.get('window').width;
const GALLERY_COLS = 3;
const GALLERY_GAP = 4;
const GALLERY_THUMB = (SCREEN_WIDTH - spacing.lg * 2 - GALLERY_GAP * (GALLERY_COLS - 1)) / GALLERY_COLS;

interface GalleryImage {
  id: string;
  prompt: string;
  model: string;
  width: number;
  height: number;
  created_at: string;
}

export default function ImageGenScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const router = useRouter();
  const params = useLocalSearchParams<{ name?: string; description?: string; card_id?: string }>();

  const [prompt, setPrompt] = useState(params.description || '');
  const [style, setStyle] = useState<Style>('anime');
  const [size, setSize] = useState<Size>('portrait');
  const [seed, setSeed] = useState('');
  const [generating, setGenerating] = useState(false);
  const [imageId, setImageId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showGallery, setShowGallery] = useState(false);
  const [gallery, setGallery] = useState<GalleryImage[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const isCharacterMode = !!(params.name && params.description);

  const loadGallery = useCallback(async () => {
    setGalleryLoading(true);
    try {
      const res = await getImageGallery();
      setGallery(res.images || []);
    } catch { /* ignore */ }
    setGalleryLoading(false);
  }, []);

  useEffect(() => {
    if (showGallery && gallery.length === 0) loadGallery();
  }, [showGallery]);

  const randomSeed = () => {
    const s = Math.floor(Math.random() * 2147483647).toString();
    setSeed(s);
    Haptics.selectionAsync();
  };

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
          character_card_id: params.card_id,
        });
      } else {
        const s = SIZES[size];
        const stylePrefix = style === 'anime' ? 'masterpiece, highly detailed anime illustration, '
          : style === 'realistic' ? 'masterpiece, photorealistic, highly detailed photograph, '
          : 'masterpiece, best quality, detailed illustration, ';
        result = await generateImage({
          prompt: stylePrefix + prompt.trim(),
          width: s.w,
          height: s.h,
          ...(seed ? { seed: parseInt(seed, 10) } : {}),
        });
      }

      if (result.success && result.image_id) {
        setImageId(result.image_id);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        // Refresh gallery if it's open
        if (showGallery) loadGallery();
      } else {
        setError(result.error || 'Generation failed');
      }
    } catch (e: any) {
      setError(e.message || 'Connection failed — is the server running?');
    } finally {
      setGenerating(false);
    }
  };

  const handleSaveToDevice = async () => {
    if (!imageId) return;
    setSaving(true);
    try {
      const uri = getImageUrl(imageId);
      if (Platform.OS === 'web') {
        // Browser: open image in new tab for right-click save
        window.open(uri, '_blank');
        setSaving(false);
        return;
      }
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Allow photo library access to save images.');
        setSaving(false);
        return;
      }
      const fileUri = cacheDirectory + `stratos_${imageId}.png`;
      const download = await downloadAsync(uri, fileUri);
      await MediaLibrary.saveToLibraryAsync(download.uri);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Saved', 'Image saved to your photo library.');
    } catch (e: any) {
      Alert.alert('Error', 'Failed to save image.');
    }
    setSaving(false);
  };

  const handleDeleteImage = async (id: string) => {
    Alert.alert('Delete Image', 'This will permanently delete this image.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive', onPress: async () => {
          try {
            await deleteImage(id);
            if (imageId === id) setImageId(null);
            setGallery(prev => prev.filter(img => img.id !== id));
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          } catch { Alert.alert('Error', 'Failed to delete image.'); }
        },
      },
    ]);
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title={isCharacterMode ? 'Generate Portrait' : 'Image Generation'} showBack />
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">

        {/* CHROMA model badge */}
        <View style={[styles.modelBadge, { backgroundColor: tc.accent.primary + '12', borderColor: tc.accent.primary + '30' }]}>
          <Wand2 size={12} color={tc.accent.primary} />
          <Text style={[styles.modelText, { color: tc.accent.primary }]}>CHROMA · 8-step · ~30s</Text>
        </View>

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

        {/* Advanced options toggle */}
        {!isCharacterMode && (
          <TouchableOpacity
            style={styles.advancedToggle}
            onPress={() => { setShowAdvanced(!showAdvanced); Haptics.selectionAsync(); }}
            activeOpacity={0.7}
          >
            <Text style={[styles.advancedText, { color: tc.text.muted }]}>Advanced</Text>
            {showAdvanced ? <ChevronUp size={14} color={tc.text.muted} /> : <ChevronDown size={14} color={tc.text.muted} />}
          </TouchableOpacity>
        )}

        {/* Seed control */}
        {showAdvanced && !isCharacterMode && (
          <View style={styles.seedRow}>
            <Text style={[styles.seedLabel, { color: tc.text.secondary }]}>Seed</Text>
            <TextInput
              style={[styles.seedInput, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]}
              value={seed}
              onChangeText={t => setSeed(t.replace(/[^0-9]/g, ''))}
              placeholder="Random"
              placeholderTextColor={tc.text.muted}
              keyboardType="numeric"
            />
            <TouchableOpacity style={[styles.seedBtn, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]} onPress={randomSeed} activeOpacity={0.7}>
              <Shuffle size={16} color={tc.text.secondary} />
            </TouchableOpacity>
          </View>
        )}

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
              <Text style={styles.generateText}>Generating...</Text>
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
            {/* Action buttons under result */}
            <View style={styles.resultActions}>
              <TouchableOpacity
                style={[styles.resultBtn, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]}
                onPress={handleSaveToDevice}
                disabled={saving}
                activeOpacity={0.7}
              >
                {saving ? <ActivityIndicator size={14} color={tc.text.secondary} /> : <Download size={16} color={tc.text.secondary} />}
                <Text style={[styles.resultBtnText, { color: tc.text.secondary }]}>{saving ? 'Saving...' : 'Save'}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.resultBtn, { backgroundColor: tc.status.error + '10', borderColor: tc.status.error + '30' }]}
                onPress={() => handleDeleteImage(imageId)}
                activeOpacity={0.7}
              >
                <Trash2 size={16} color={tc.status.error} />
                <Text style={[styles.resultBtnText, { color: tc.status.error }]}>Delete</Text>
              </TouchableOpacity>
            </View>
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

        {/* Gallery toggle */}
        <TouchableOpacity
          style={[styles.galleryToggle, { borderColor: tc.border.subtle }]}
          onPress={() => { setShowGallery(!showGallery); Haptics.selectionAsync(); }}
          activeOpacity={0.7}
        >
          <Grid3X3 size={16} color={tc.text.secondary} />
          <Text style={[styles.galleryToggleText, { color: tc.text.secondary }]}>
            {showGallery ? 'Hide Gallery' : 'Recent Generations'}
          </Text>
          <Text style={[styles.galleryCount, { color: tc.text.muted }]}>
            {gallery.length > 0 ? `(${gallery.length})` : ''}
          </Text>
          {showGallery ? <ChevronUp size={14} color={tc.text.muted} /> : <ChevronDown size={14} color={tc.text.muted} />}
        </TouchableOpacity>

        {/* Gallery grid */}
        {showGallery && (
          <View style={styles.galleryContainer}>
            {galleryLoading ? (
              <ActivityIndicator size="small" color={tc.accent.primary} style={{ paddingVertical: spacing.xl }} />
            ) : gallery.length === 0 ? (
              <Text style={[styles.galleryEmpty, { color: tc.text.muted }]}>No images generated yet</Text>
            ) : (
              <View style={styles.galleryGrid}>
                {gallery.map(item => (
                  <TouchableOpacity
                    key={item.id}
                    style={[styles.galleryThumb, { backgroundColor: tc.bg.tertiary }]}
                    onPress={() => setImageId(item.id)}
                    onLongPress={() => handleDeleteImage(item.id)}
                    activeOpacity={0.7}
                  >
                    <Image
                      source={{ uri: getImageUrl(item.id) }}
                      style={styles.galleryImage}
                      contentFit="cover"
                    />
                    {imageId === item.id && (
                      <View style={[styles.gallerySelected, { borderColor: tc.accent.primary }]} />
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            )}
            <Text style={[styles.galleryHint, { color: tc.text.faint }]}>Tap to preview · Long press to delete</Text>
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
  modelBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', paddingHorizontal: spacing.md, paddingVertical: 4, borderRadius: borderRadius.full, borderWidth: 1, marginBottom: spacing.sm },
  modelText: { fontSize: 11, fontFamily: fonts.button },
  label: { fontFamily: fonts.heading, fontSize: 14, marginTop: spacing.lg, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, fontFamily: fonts.body, minHeight: 100, borderWidth: 1, textAlignVertical: 'top' },
  chipRow: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
  chip: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  chipText: { fontSize: 13, fontFamily: fonts.button },
  advancedToggle: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: spacing.md, alignSelf: 'flex-start' },
  advancedText: { fontSize: 12, fontFamily: fonts.button },
  seedRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginTop: spacing.sm },
  seedLabel: { fontSize: 13, fontFamily: fonts.body, width: 36 },
  seedInput: { flex: 1, borderRadius: borderRadius.md, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, fontSize: 14, fontFamily: fonts.body, borderWidth: 1 },
  seedBtn: { width: 40, height: 40, borderRadius: borderRadius.md, borderWidth: 1, justifyContent: 'center', alignItems: 'center' },
  generateBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, marginTop: spacing.lg },
  generateText: { fontSize: 16, fontFamily: fonts.heading, color: '#fff' },
  error: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', marginTop: spacing.md },
  resultContainer: { marginTop: spacing.lg, alignItems: 'center' },
  resultImage: { width: '100%', aspectRatio: 3 / 4, borderRadius: borderRadius.lg, borderWidth: 1 },
  resultActions: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.md, width: '100%' },
  resultBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1 },
  resultBtnText: { fontSize: 13, fontFamily: fonts.button },
  imageIdText: { fontSize: 10, fontFamily: fonts.body, marginTop: spacing.xs },
  placeholder: { marginTop: spacing.xl, alignItems: 'center', justifyContent: 'center', paddingVertical: spacing.xxl * 2, borderWidth: 1, borderStyle: 'dashed', borderRadius: borderRadius.lg, gap: spacing.md },
  placeholderText: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', paddingHorizontal: spacing.xl },
  charInfo: { borderRadius: borderRadius.lg, padding: spacing.lg, borderWidth: 1, gap: spacing.xs },
  charName: { fontSize: 18, fontFamily: fonts.heading },
  charDesc: { fontSize: 13, fontFamily: fonts.body, lineHeight: 20 },
  galleryToggle: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginTop: spacing.xl, paddingVertical: spacing.md, borderTopWidth: 1 },
  galleryToggleText: { fontSize: 14, fontFamily: fonts.heading, flex: 1 },
  galleryCount: { fontSize: 12, fontFamily: fonts.body },
  galleryContainer: { marginTop: spacing.sm },
  galleryGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: GALLERY_GAP },
  galleryThumb: { width: GALLERY_THUMB, height: GALLERY_THUMB, borderRadius: borderRadius.sm, overflow: 'hidden' },
  galleryImage: { width: '100%', height: '100%' },
  gallerySelected: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderWidth: 2, borderRadius: borderRadius.sm },
  galleryEmpty: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', paddingVertical: spacing.xl },
  galleryHint: { fontSize: 10, fontFamily: fonts.body, textAlign: 'center', marginTop: spacing.sm },
});

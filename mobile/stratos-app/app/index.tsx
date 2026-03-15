import React from 'react';
import { Redirect } from 'expo-router';
import { useAuthStore } from '../stores/authStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';

export default function Index() {
  const { isLoading } = useAuthStore();
  if (isLoading) return <LoadingScreen />;
  // Auth is optional — app works without login (anonymous/local mode)
  // Login is available in Settings for sync/premium features
  return <Redirect href="/(tabs)/discover" />;
}

import React from 'react';
import { Redirect } from 'expo-router';
import { useAuthStore } from '../stores/authStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';

export default function Index() {
  const { isAuthenticated, isLoading } = useAuthStore();
  if (isLoading) return <LoadingScreen />;
  if (!isAuthenticated) return <Redirect href="/(auth)/login" />;
  return <Redirect href="/(tabs)/discover" />;
}

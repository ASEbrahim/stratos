// Font families loaded in _layout.tsx
// Comfortaa: round, geometric — used for headings, logos, buttons
// Poppins: clean geometric — used for body text, labels

export const fonts = {
  // Comfortaa — round/geometric headings
  heading: 'Comfortaa_700Bold',
  headingBold: 'Comfortaa_700Bold',
  logo: 'Comfortaa_700Bold',
  button: 'Comfortaa_600SemiBold',

  // Poppins — clean/modern body
  body: 'Poppins_400Regular',
  bodyMedium: 'Poppins_500Medium',
  bodySemiBold: 'Poppins_600SemiBold',
  bodyBold: 'Poppins_700Bold',
  bodyLight: 'Poppins_300Light',

  // Fallback for before fonts load
  fallback: undefined as string | undefined,
};

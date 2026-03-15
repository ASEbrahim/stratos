// Font families loaded in _layout.tsx
// Nunito: bubbly, friendly — used for headings, logos, buttons
// Poppins: clean geometric — used for body text, labels

export const fonts = {
  // Nunito — bubbly/friendly
  heading: 'Nunito_700Bold',
  headingBold: 'Nunito_800ExtraBold',
  logo: 'Nunito_900Black',
  button: 'Nunito_600SemiBold',

  // Poppins — clean/modern
  body: 'Poppins_400Regular',
  bodyMedium: 'Poppins_500Medium',
  bodySemiBold: 'Poppins_600SemiBold',
  bodyBold: 'Poppins_700Bold',
  bodyLight: 'Poppins_300Light',

  // Fallback for before fonts load
  fallback: undefined as string | undefined,
};

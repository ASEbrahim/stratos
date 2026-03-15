// Font families loaded in _layout.tsx
// Quicksand: round, modern — used for headings, logos, buttons
// Poppins: clean geometric — used for body text, labels
// Nunito: bubbly friendly — available as alternate

export const fonts = {
  // Quicksand — round/modern headings
  heading: 'Quicksand_700Bold',
  headingBold: 'Quicksand_700Bold',
  logo: 'Quicksand_700Bold',
  button: 'Quicksand_600SemiBold',

  // Poppins — clean/modern body
  body: 'Poppins_400Regular',
  bodyMedium: 'Poppins_500Medium',
  bodySemiBold: 'Poppins_600SemiBold',
  bodyBold: 'Poppins_700Bold',
  bodyLight: 'Poppins_300Light',

  // Fallback for before fonts load
  fallback: undefined as string | undefined,
};

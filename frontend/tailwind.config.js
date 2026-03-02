/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.html', './*.js'],
  safelist: [
    // Dynamic opacity modifier classes from JS template literals
    'active:bg-emerald-500/50',
    'focus:border-emerald-500/50',
    'group-hover/cat:opacity-100',
    'hover:bg-amber-800/50', 'hover:bg-amber-900/20', 'hover:bg-amber-900/30',
    'hover:bg-blue-600/50', 'hover:bg-blue-900/30',
    'hover:bg-emerald-500/30', 'hover:bg-emerald-900/30',
    'hover:bg-purple-800/50',
    'hover:bg-red-500/10', 'hover:bg-red-900/20', 'hover:bg-red-900/30', 'hover:bg-red-900/50',
    'hover:bg-slate-600/50', 'hover:bg-slate-800/30', 'hover:bg-slate-800/40', 'hover:bg-slate-800/50',
    'hover:border-blue-500/50', 'hover:border-emerald-500/30',
    'hover:border-emerald-700/50',
    'hover:border-red-500/30', 'hover:border-red-500/40', 'hover:border-red-500/50',
    'hover:border-white/50',
    'hover:shadow-black/10',
    // Dynamic accent colors from feed.js
    'text-emerald-400', 'text-purple-400', 'text-blue-400',
    'hover:text-emerald-400', 'hover:text-purple-400', 'hover:text-blue-400',
  ],
  theme: { extend: {} },
  plugins: [],
}

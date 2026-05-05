// useTheme.ts — theme state hook with localStorage persistence
//
// Manages light/dark theme by toggling the `dark` class on <html> and
// updating the <meta name="theme-color"> tag. Defaults to dark mode
// (preserves existing behaviour).

import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'hs-theme';

export type Theme = 'light' | 'dark';

const META_COLORS: Record<Theme, string> = {
  dark: '#000000',
  light: '#fafafa',
};

function applyTheme(theme: Theme): void {
  const html = document.documentElement;
  if (theme === 'dark') {
    html.classList.add('dark');
  } else {
    html.classList.remove('dark');
  }

  // Update theme-color meta tag
  const meta = document.getElementById('theme-color-meta') as HTMLMetaElement | null;
  if (meta) {
    meta.content = META_COLORS[theme];
  }
}

export function useTheme(): { theme: Theme; toggleTheme: () => void } {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'light' || stored === 'dark') return stored;
    } catch { /* ignore */ }
    return 'dark';
  });

  // Apply theme to DOM on mount + when theme changes
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark';
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch { /* ignore */ }
      return next;
    });
  }, []);

  return { theme, toggleTheme };
}

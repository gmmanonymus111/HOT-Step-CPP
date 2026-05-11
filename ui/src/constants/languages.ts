// languages.ts — All 51 vocal languages supported by ACE-Step 1.5
//
// Source: upstream ACE-Step argparse choices list
// https://github.com/scragnog/HOT-Step-CPP/issues/14#issuecomment (iChristGit)
//
// The engine passes vocal_language as a free-form string, so any code the
// model was trained on will work. This list is for the UI dropdowns.

export interface VocalLanguage {
  value: string;
  label: string;
}

export const VOCAL_LANGUAGES: VocalLanguage[] = [
  // ── Tier 1: Best prompt fidelity and pronunciation ──
  { value: 'en', label: 'English' },
  { value: 'zh', label: '中文 (Chinese)' },
  { value: 'ja', label: '日本語 (Japanese)' },
  { value: 'ko', label: '한국어 (Korean)' },
  { value: 'es', label: 'Español (Spanish)' },
  { value: 'fr', label: 'Français (French)' },
  { value: 'de', label: 'Deutsch (German)' },
  { value: 'it', label: 'Italiano (Italian)' },
  { value: 'pt', label: 'Português (Portuguese)' },
  { value: 'ru', label: 'Русский (Russian)' },

  // ── Tier 2: Well-supported languages ──
  { value: 'ar', label: 'العربية (Arabic)' },
  { value: 'hi', label: 'हिन्दी (Hindi)' },
  { value: 'tr', label: 'Türkçe (Turkish)' },
  { value: 'vi', label: 'Tiếng Việt (Vietnamese)' },
  { value: 'th', label: 'ไทย (Thai)' },
  { value: 'sv', label: 'Svenska (Swedish)' },
  { value: 'pl', label: 'Polski (Polish)' },
  { value: 'nl', label: 'Nederlands (Dutch)' },
  { value: 'yue', label: '粵語 (Cantonese)' },

  // ── Remaining supported languages (alphabetical by code) ──
  { value: 'az', label: 'Azərbaycan (Azerbaijani)' },
  { value: 'bg', label: 'Български (Bulgarian)' },
  { value: 'bn', label: 'বাংলা (Bengali)' },
  { value: 'ca', label: 'Català (Catalan)' },
  { value: 'cs', label: 'Čeština (Czech)' },
  { value: 'da', label: 'Dansk (Danish)' },
  { value: 'el', label: 'Ελληνικά (Greek)' },
  { value: 'fa', label: 'فارسی (Persian)' },
  { value: 'fi', label: 'Suomi (Finnish)' },
  { value: 'he', label: 'עברית (Hebrew)' },
  { value: 'hr', label: 'Hrvatski (Croatian)' },
  { value: 'ht', label: 'Kreyòl (Haitian Creole)' },
  { value: 'hu', label: 'Magyar (Hungarian)' },
  { value: 'id', label: 'Bahasa Indonesia (Indonesian)' },
  { value: 'is', label: 'Íslenska (Icelandic)' },
  { value: 'kr', label: 'Kanuri' },
  { value: 'la', label: 'Latina (Latin)' },
  { value: 'lt', label: 'Lietuvių (Lithuanian)' },
  { value: 'ms', label: 'Bahasa Melayu (Malay)' },
  { value: 'ne', label: 'नेपाली (Nepali)' },
  { value: 'no', label: 'Norsk (Norwegian)' },
  { value: 'pa', label: 'ਪੰਜਾਬੀ (Punjabi)' },
  { value: 'ro', label: 'Română (Romanian)' },
  { value: 'sa', label: 'संस्कृत (Sanskrit)' },
  { value: 'sk', label: 'Slovenčina (Slovak)' },
  { value: 'sr', label: 'Српски (Serbian)' },
  { value: 'sw', label: 'Kiswahili (Swahili)' },
  { value: 'ta', label: 'தமிழ் (Tamil)' },
  { value: 'te', label: 'తెలుగు (Telugu)' },
  { value: 'tl', label: 'Tagalog' },
  { value: 'uk', label: 'Українська (Ukrainian)' },
  { value: 'ur', label: 'اردو (Urdu)' },
];

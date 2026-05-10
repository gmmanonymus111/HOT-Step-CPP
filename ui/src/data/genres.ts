// genres.ts — Curated hierarchical genre taxonomy for Insta-Gen
//
// ACE-Step 1.5 uses free-text caption prompts rather than a rigid genre
// database.  This taxonomy provides a user-friendly selection UI that
// maps to well-known musical terms the LM understands well.
//
// Categories and genres are ordered roughly by popularity / breadth.

export interface GenreCategory {
  name: string;
  icon: string;      // emoji for visual grouping
  genres: string[];
}

export const GENRE_TAXONOMY: GenreCategory[] = [
  {
    name: 'Pop',
    icon: '🎤',
    genres: [
      'Pop', 'Synth-Pop', 'Electropop', 'Dance Pop', 'Dream Pop',
      'Indie Pop', 'Art Pop', 'Bubblegum Pop', 'Teen Pop', 'Power Pop',
      'Chamber Pop', 'Baroque Pop', 'K-Pop', 'J-Pop', 'C-Pop',
      'Sophisti-Pop', 'Hyperpop', 'City Pop',
    ],
  },
  {
    name: 'Rock',
    icon: '🎸',
    genres: [
      'Rock', 'Alternative Rock', 'Indie Rock', 'Classic Rock', 'Hard Rock',
      'Soft Rock', 'Progressive Rock', 'Psychedelic Rock', 'Post-Rock',
      'Punk Rock', 'Pop Punk', 'Garage Rock', 'Surf Rock', 'Grunge',
      'Shoegaze', 'Math Rock', 'Stoner Rock', 'Blues Rock', 'Folk Rock',
      'Post-Punk', 'Emo', 'Noise Rock',
    ],
  },
  {
    name: 'Electronic',
    icon: '🎧',
    genres: [
      'Electronic', 'EDM', 'House', 'Deep House', 'Tech House',
      'Progressive House', 'Techno', 'Minimal Techno', 'Trance',
      'Psytrance', 'Drum and Bass', 'Dubstep', 'Future Bass',
      'Ambient', 'Downtempo', 'Chillwave', 'Synthwave', 'Retrowave',
      'IDM', 'Breakbeat', 'Garage', 'Hardstyle', 'Electro',
      'Vaporwave', 'Glitch',
    ],
  },
  {
    name: 'Hip-Hop',
    icon: '🎙️',
    genres: [
      'Hip-Hop', 'Rap', 'Trap', 'Lo-Fi Hip-Hop', 'Boom Bap',
      'Drill', 'Grime', 'Cloud Rap', 'Conscious Hip-Hop',
      'Gangsta Rap', 'Mumble Rap', 'Old School Hip-Hop',
      'Phonk', 'Crunk', 'Chopped and Screwed',
    ],
  },
  {
    name: 'R&B / Soul',
    icon: '💜',
    genres: [
      'R&B', 'Soul', 'Neo-Soul', 'Contemporary R&B', 'Funk',
      'Disco', 'Motown', 'Quiet Storm', 'New Jack Swing',
      'P-Funk', 'Afrobeats', 'Gospel',
    ],
  },
  {
    name: 'Metal',
    icon: '🤘',
    genres: [
      'Heavy Metal', 'Thrash Metal', 'Death Metal', 'Black Metal',
      'Doom Metal', 'Power Metal', 'Progressive Metal', 'Symphonic Metal',
      'Nu Metal', 'Metalcore', 'Deathcore', 'Gothic Metal',
      'Sludge Metal', 'Speed Metal', 'Folk Metal', 'Djent',
      'Industrial Metal',
    ],
  },
  {
    name: 'Jazz',
    icon: '🎷',
    genres: [
      'Jazz', 'Smooth Jazz', 'Bebop', 'Cool Jazz', 'Swing',
      'Jazz Fusion', 'Acid Jazz', 'Free Jazz', 'Latin Jazz',
      'Bossa Nova Jazz', 'Modal Jazz', 'Gypsy Jazz',
    ],
  },
  {
    name: 'Classical',
    icon: '🎻',
    genres: [
      'Classical', 'Orchestral', 'Chamber Music', 'Opera', 'Baroque',
      'Romantic', 'Minimalist', 'Contemporary Classical',
      'Choral', 'Neoclassical',
    ],
  },
  {
    name: 'Country',
    icon: '🤠',
    genres: [
      'Country', 'Country Pop', 'Country Rock', 'Bluegrass',
      'Americana', 'Honky-Tonk', 'Outlaw Country', 'Alt-Country',
      'Country Blues',
    ],
  },
  {
    name: 'Folk',
    icon: '🪕',
    genres: [
      'Folk', 'Indie Folk', 'Contemporary Folk', 'Celtic',
      'World Music', 'Flamenco', 'Acoustic', 'Singer-Songwriter',
      'Neofolk', 'Freak Folk',
    ],
  },
  {
    name: 'Blues',
    icon: '🎵',
    genres: [
      'Blues', 'Delta Blues', 'Chicago Blues', 'Electric Blues',
      'Blues Rock', 'Jump Blues', 'Rhythm and Blues', 'Boogie-Woogie',
    ],
  },
  {
    name: 'Reggae / Caribbean',
    icon: '🌴',
    genres: [
      'Reggae', 'Dancehall', 'Ska', 'Dub', 'Roots Reggae',
      'Reggaeton', 'Soca', 'Calypso',
    ],
  },
  {
    name: 'Latin',
    icon: '💃',
    genres: [
      'Latin', 'Salsa', 'Bossa Nova', 'Bachata', 'Cumbia',
      'Merengue', 'Tango', 'Latin Pop', 'Mariachi', 'Norteño',
    ],
  },
  {
    name: 'Soundtrack / Cinematic',
    icon: '🎬',
    genres: [
      'Film Score', 'Epic', 'Cinematic', 'Video Game Music',
      'Orchestral Soundtrack', 'Trailer Music', 'Dark Ambient',
      'Fantasy', 'Sci-Fi',
    ],
  },
  {
    name: 'Experimental / Other',
    icon: '🔮',
    genres: [
      'Experimental', 'Avant-Garde', 'Noise', 'Industrial',
      'New Age', 'Meditation', 'Lo-Fi', 'Post-Industrial',
      'Art Rock', 'Drone', 'Musique Concrète',
    ],
  },
];

/** Flat list of all genre names (for validation / autocomplete) */
export const ALL_GENRES: string[] = GENRE_TAXONOMY.flatMap(cat => cat.genres);

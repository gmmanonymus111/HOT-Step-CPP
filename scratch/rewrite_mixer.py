import re

path = r'D:\Ace-Step-Latest\hot-step-cpp\ui\src\components\cover-studio\StemMixer.tsx'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Props
code = re.sub(
    r'onRecombine\?: \(audioBlob: Blob\) => void;\n\}',
    'onRecombine?: (audioBlob: Blob) => void;\n  onClose?: () => void;\n}',
    code
)

code = code.replace(
    'export const StemMixer: React.FC<StemMixerProps> = ({ jobId, stems, controls, onControlsChange, onRecombine }) => {',
    'export const StemMixer: React.FC<StemMixerProps> = ({ jobId, stems, controls, onControlsChange, onRecombine, onClose }) => {'
)

# 2. State & Refs
state_add = """  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const playbackOffsetRef = useRef(0);
  const playbackStartTimeRef = useRef(0);
  const animationFrameRef = useRef<number>();

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };
"""
code = code.replace(
    '  const [loadingStems, setLoadingStems] = useState(false);',
    '  const [loadingStems, setLoadingStems] = useState(false);\n' + state_add
)

# 3. loadBuffers (duration)
load_add = """
        if (stem.index === stems[0].index) {
          setDuration(audioBuf.duration);
        }
"""
code = code.replace(
    'buffersRef.current.set(stem.index, audioBuf);',
    'buffersRef.current.set(stem.index, audioBuf);' + load_add
)

# 4. updateProgress loop
progress_loop = """
  const updateProgress = useCallback(() => {
    if (audioContextRef.current && isPlaying) {
      const elapsed = audioContextRef.current.currentTime - playbackStartTimeRef.current;
      let newTime = playbackOffsetRef.current + elapsed;
      if (duration > 0 && newTime >= duration) {
         newTime = 0;
         setIsPlaying(false);
         playbackOffsetRef.current = 0;
      } else {
         setCurrentTime(newTime);
         animationFrameRef.current = requestAnimationFrame(updateProgress);
      }
    }
  }, [isPlaying, duration]);

  useEffect(() => {
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(updateProgress);
    } else {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
    return () => { if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current); };
  }, [isPlaying, updateProgress]);
"""
code = code.replace(
    '  // Start/stop preview playback',
    progress_loop + '\n  // Start/stop preview playback'
)

# 5. togglePlayback
toggle_playback_orig = """  // Start/stop preview playback
  const togglePlayback = useCallback(async () => {
    if (isPlaying) {
      // Stop
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();
      setIsPlaying(false);
      return;
    }"""

toggle_playback_new = """  // Start/stop preview playback
  const togglePlayback = useCallback(async () => {
    if (isPlaying) {
      // Pause
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();
      setIsPlaying(false);
      if (audioContextRef.current) {
         playbackOffsetRef.current += (audioContextRef.current.currentTime - playbackStartTimeRef.current);
      }
      return;
    }"""
code = code.replace(toggle_playback_orig, toggle_playback_new)

# 6. Source starting offset
source_start_orig = """      source.connect(gain);
      gain.connect(ctx.destination);
      source.start(startTime);"""

source_start_new = """      source.connect(gain);
      gain.connect(ctx.destination);
      source.start(startTime, playbackOffsetRef.current);"""
code = code.replace(source_start_orig, source_start_new)

# 7. playbackStartTimeRef update
source_end_loop_orig = """      source.onended = () => {
        sourceNodesRef.current.delete(stem.index);
        if (sourceNodesRef.current.size === 0) {
          setIsPlaying(false);
        }
      };
    }
    setIsPlaying(true);"""

source_end_loop_new = """      source.onended = () => {
        sourceNodesRef.current.delete(stem.index);
        if (sourceNodesRef.current.size === 0) {
          setIsPlaying(false);
          playbackOffsetRef.current = 0;
          setCurrentTime(0);
        }
      };
    }
    playbackStartTimeRef.current = ctx.currentTime;
    setIsPlaying(true);"""
code = code.replace(source_end_loop_orig, source_end_loop_new)

# 8. Scrubber seek
scrub_handler = """
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    playbackOffsetRef.current = time;
    
    // If playing, restart from new time
    if (isPlaying) {
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();
      setIsPlaying(false);
      setTimeout(togglePlayback, 50); // micro-delay to let stop propagate
    }
  };
"""
code = code.replace('  // Update gain nodes in real-time when controls change', scrub_handler + '\n  // Update gain nodes in real-time when controls change')

# 9. UI / Modal wrap
ui_orig = """  return (
    <div style={styles.container}>"""

ui_new = """  const content = (
    <div style={{ ...styles.container, width: '100%', maxWidth: '600px', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }}>"""

code = code.replace(ui_orig, ui_new)

close_btn = """        <h3 style={styles.title}>🎛️ Stem Mixer</h3>
        {onClose && (
           <button onClick={onClose} className="text-zinc-400 hover:text-white transition-colors" style={{ marginLeft: 'auto', marginRight: 16 }}>\u2715</button>
        )}"""
code = code.replace('<h3 style={styles.title}>🎛️ Stem Mixer</h3>', close_btn)

# add scrubber UI
scrubber_ui = """
      {duration > 0 && (
        <div style={{ padding: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 12, color: '#a3a3a3', fontFamily: 'monospace', width: 40, textAlign: 'right' }}>{formatTime(currentTime)}</span>
          <input 
            type="range" 
            min="0" max={duration} step="0.1" 
            value={currentTime} 
            onChange={handleSeek}
            style={{ flex: 1, accentColor: '#ec4899', height: 4, cursor: 'pointer' }} 
          />
          <span style={{ fontSize: 12, color: '#a3a3a3', fontFamily: 'monospace', width: 40 }}>{formatTime(duration)}</span>
        </div>
      )}
"""
code = code.replace('      <div style={styles.stemList}>', scrubber_ui + '\n      <div style={{ ...styles.stemList, overflowY: \'auto\' }}>')

# close modal wrapper
code = code.replace('    </div>\n  );\n};', """    </div>
  );

  if (!onClose) return content;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }} onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
       {content}
    </div>
  );
};""")

# Remove dependency from togglePlayback hook list
code = code.replace('  }, [isPlaying, stems, controls, soloIndex, loadedStems, loadBuffers]);', '  }, [isPlaying, stems, controls, soloIndex, loadedStems, loadBuffers, duration]);')


with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

print("StemMixer rewritten successfully.")

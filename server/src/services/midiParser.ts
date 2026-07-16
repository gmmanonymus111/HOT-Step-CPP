// midiParser.ts — minimal Standard MIDI File reader for piano-roll previews
//
// Parses note on/off + tempo map from format 0/1 SMF and returns notes with
// absolute times in seconds. Deliberately small: this only powers the MIDI
// Studio piano-roll preview; the authoritative artifact is the .mid file
// itself, which the user downloads for their DAW.

export interface MidiNote {
  pitch: number;      // 0-127
  velocity: number;   // 1-127
  channel: number;    // 0-15 (9 = GM drums)
  start: number;      // seconds
  duration: number;   // seconds
}

export interface MidiChannelInfo {
  channel: number;
  program: number;    // GM program number (first program change seen, else 0)
  isDrums: boolean;
  noteCount: number;
}

export interface ParsedMidi {
  durationSec: number;
  noteCount: number;
  channels: MidiChannelInfo[];
  notes: MidiNote[];
}

interface RawNoteEvent { tick: number; on: boolean; channel: number; pitch: number; velocity: number; order: number; }
interface TempoEvent { tick: number; usPerQuarter: number; }

class Reader {
  pos = 0;
  constructor(private buf: Buffer) {}
  get eof() { return this.pos >= this.buf.length; }
  u8(): number { return this.buf[this.pos++]; }
  peek(): number { return this.buf[this.pos]; }
  u16(): number { const v = this.buf.readUInt16BE(this.pos); this.pos += 2; return v; }
  u32(): number { const v = this.buf.readUInt32BE(this.pos); this.pos += 4; return v; }
  skip(n: number) { this.pos += n; }
  varLen(): number {
    let v = 0;
    for (let i = 0; i < 4; i++) {
      const b = this.u8();
      v = (v << 7) | (b & 0x7f);
      if ((b & 0x80) === 0) break;
    }
    return v;
  }
  ascii(n: number): string { const s = this.buf.toString('latin1', this.pos, this.pos + n); this.pos += n; return s; }
}

export function parseMidiFile(buf: Buffer): ParsedMidi {
  const r = new Reader(buf);
  if (r.ascii(4) !== 'MThd') throw new Error('Not a MIDI file (missing MThd)');
  const headerLen = r.u32();
  const format = r.u16();
  const ntrks = r.u16();
  const division = r.u16();
  r.skip(headerLen - 6);
  if (format > 2) throw new Error(`Unsupported MIDI format ${format}`);

  const smpte = (division & 0x8000) !== 0;
  // SMPTE: ticks map to seconds directly; PPQ: via the tempo map
  let secPerTick = 0;
  if (smpte) {
    const fps = 256 - (division >> 8);          // two's-complement negative byte
    const ticksPerFrame = division & 0xff;
    secPerTick = 1 / (fps * ticksPerFrame);
  }
  const ppq = division & 0x7fff;

  const noteEvents: RawNoteEvent[] = [];
  const tempoEvents: TempoEvent[] = [];
  const channelProgram = new Map<number, number>();
  let order = 0;

  for (let t = 0; t < ntrks && !r.eof; t++) {
    if (r.ascii(4) !== 'MTrk') throw new Error(`Track ${t}: missing MTrk`);
    const len = r.u32();
    const end = r.pos + len;
    let tick = 0;
    let runningStatus = 0;

    while (r.pos < end) {
      tick += r.varLen();
      let status = r.peek();
      if (status & 0x80) { r.skip(1); if (status < 0xf0) runningStatus = status; }
      else { status = runningStatus; if (!status) throw new Error(`Track ${t}: data byte with no running status`); }

      if (status === 0xff) {                    // meta event
        const type = r.u8();
        const mlen = r.varLen();
        if (type === 0x51 && mlen === 3) {
          tempoEvents.push({ tick, usPerQuarter: (r.u8() << 16) | (r.u8() << 8) | r.u8() });
        } else {
          r.skip(mlen);
          if (type === 0x2f) break;             // end of track
        }
      } else if (status === 0xf0 || status === 0xf7) {   // sysex
        r.skip(r.varLen());
      } else {
        const kind = status & 0xf0;
        const channel = status & 0x0f;
        if (kind === 0x90 || kind === 0x80) {
          const pitch = r.u8();
          const velocity = r.u8();
          const on = kind === 0x90 && velocity > 0;
          noteEvents.push({ tick, on, channel, pitch, velocity, order: order++ });
        } else if (kind === 0xc0) {
          const program = r.u8();
          if (!channelProgram.has(channel)) channelProgram.set(channel, program);
        } else if (kind === 0xd0) {
          r.skip(1);                            // channel aftertouch
        } else {
          r.skip(2);                            // poly AT, CC, pitch bend
        }
      }
    }
    r.pos = end;                                // realign in case of sloppy track data
  }

  // Tick → seconds via the tempo map (default 120 bpm = 500000 us/quarter)
  tempoEvents.sort((a, b) => a.tick - b.tick);
  const segments: Array<{ tick: number; sec: number; secPerTick: number }> = [];
  {
    let curTick = 0, curSec = 0;
    let curSpt = smpte ? secPerTick : 500_000 / 1_000_000 / ppq;
    segments.push({ tick: 0, sec: 0, secPerTick: curSpt });
    if (!smpte) {
      for (const te of tempoEvents) {
        curSec += (te.tick - curTick) * curSpt;
        curTick = te.tick;
        curSpt = te.usPerQuarter / 1_000_000 / ppq;
        segments.push({ tick: curTick, sec: curSec, secPerTick: curSpt });
      }
    }
  }
  const tickToSec = (tick: number): number => {
    let seg = segments[0];
    for (let i = segments.length - 1; i >= 0; i--) {
      if (segments[i].tick <= tick) { seg = segments[i]; break; }
    }
    return seg.sec + (tick - seg.tick) * seg.secPerTick;
  };

  // Pair note-ons with note-offs (FIFO per channel+pitch; merged track order)
  noteEvents.sort((a, b) => a.tick - b.tick || a.order - b.order);
  const open = new Map<number, RawNoteEvent[]>();
  const notes: MidiNote[] = [];
  for (const ev of noteEvents) {
    const key = ev.channel * 128 + ev.pitch;
    if (ev.on) {
      let q = open.get(key);
      if (!q) { q = []; open.set(key, q); }
      q.push(ev);
    } else {
      const q = open.get(key);
      const start = q?.shift();
      if (start) {
        notes.push({
          pitch: start.pitch,
          velocity: start.velocity,
          channel: start.channel,
          start: round3(tickToSec(start.tick)),
          duration: round3(Math.max(0.01, tickToSec(ev.tick) - tickToSec(start.tick))),
        });
      }
    }
  }
  notes.sort((a, b) => a.start - b.start);

  const channelCounts = new Map<number, number>();
  let durationSec = 0;
  for (const n of notes) {
    channelCounts.set(n.channel, (channelCounts.get(n.channel) || 0) + 1);
    durationSec = Math.max(durationSec, n.start + n.duration);
  }
  const channels: MidiChannelInfo[] = [...channelCounts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([channel, noteCount]) => ({
      channel,
      program: channelProgram.get(channel) ?? 0,
      isDrums: channel === 9,
      noteCount,
    }));

  return { durationSec: round3(durationSec), noteCount: notes.length, channels, notes };
}

function round3(v: number): number { return Math.round(v * 1000) / 1000; }

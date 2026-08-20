// migrateMidi.ts — One-time migration for stale MIDI job data
//
// Migrates flat MIDI job directories (data/midi/<jobId>/) to user-scoped
// (data/midi/<userId>/<jobId>/) by assigning them to the first admin user
// (or first user if no admin exists).
//
// Safe to run repeatedly — it's a no-op if no stale MIDI jobs exist.
//
// Usage:
//   npx tsx server/src/services/migrateMidi.ts
//
// Or add to database.ts init:
//   import { migrateMidi } from './services/migrateMidi.js';
//   migrateMidi(getDb());

import path from 'path';
import fs from 'fs';
import { config } from '../config.js';
import type Database from 'better-sqlite3';

export function migrateMidi(db: Database.Database): void {
  const midiBaseDir = path.join(config.data.dir, 'midi');
  if (!fs.existsSync(midiBaseDir)) {
    return; // Nothing to migrate
  }

  // Get target user (first admin, or first user)
  const adminUser = db.prepare("SELECT id FROM users WHERE role='admin' LIMIT 1").get() as any;
  const firstUser = db.prepare("SELECT id FROM users LIMIT 1").get() as any;
  const targetUserId = adminUser?.id || firstUser?.id;

  if (!targetUserId) {
    console.log('[MIDI Migration] No users found — skipping migration');
    return;
  }

  // Check for stale MIDI jobs (flat directories directly under midi/)
  const entries = fs.readdirSync(midiBaseDir, { withFileTypes: true });
  const staleDirs = entries.filter(e => e.isDirectory());

  if (staleDirs.length === 0) {
    return; // No stale MIDI jobs
  }

  // Check if target user directory already exists
  const targetUserDir = path.join(midiBaseDir, targetUserId);
  if (!fs.existsSync(targetUserDir)) {
    fs.mkdirSync(targetUserDir, { recursive: true });
  }

  let migrated = 0;
  for (const entry of staleDirs) {
    // Skip if this looks like a user ID directory (has _meta.json with userId)
    const metaPath = path.join(midiBaseDir, entry.name, '_meta.json');
    if (fs.existsSync(metaPath)) {
      try {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
        // If it has userId, it's already scoped — skip
        if (meta.userId) {
          continue;
        }
      } catch {
        // Corrupted meta — still migrate it
      }
    }

    // Migrate this directory to target user
    const srcPath = path.join(midiBaseDir, entry.name);
    const destPath = path.join(targetUserDir, entry.name);

    try {
      fs.renameSync(srcPath, destPath);
      migrated++;
      console.log(`[MIDI Migration] Migrated ${entry.name} → ${targetUserId}/${entry.name}`);
    } catch (err: any) {
      console.error(`[MIDI Migration] Failed to migrate ${entry.name}: ${err.message}`);
    }
  }

  if (migrated > 0) {
    console.log(`[MIDI Migration] Complete: migrated ${migrated} stale MIDI jobs to user ${targetUserId}`);
  }
}

// Run if called directly
if (require.main === module) {
  // Initialize db for standalone execution
  const Database = require('better-sqlite3');
  const dbPath = path.join(config.data.dir, 'hotstep.db');
  const db = new Database(dbPath);
  console.log('Running MIDI migration...');
  migrateMidi(db);
  db.close();
  console.log('Done');
}
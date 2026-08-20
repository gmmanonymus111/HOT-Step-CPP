// migrateStems.ts — One-time migration for stale stem data
//
// Migrates flat stem directories (data/stems/<jobId>/) to user-scoped
// (data/stems/<userId>/<jobId>/) by assigning them to the first admin user
// (or first user if no admin exists).
//
// Safe to run repeatedly — it's a no-op if no stale stems exist.
//
// Usage:
//   npx tsx server/src/services/migrateStems.ts
//
// Or add to database.ts init:
//   import { migrateStems } from './services/migrateStems.js';
//   migrateStems(getDb());

import path from 'path';
import fs from 'fs';
import { config } from '../config.js';
import type Database from 'better-sqlite3';

export function migrateStems(db: Database.Database): void {
  const stemsBaseDir = path.join(config.data.dir, 'stems');
  if (!fs.existsSync(stemsBaseDir)) {
    return; // Nothing to migrate
  }

  // Get target user (first admin, or first user)
  const adminUser = db.prepare("SELECT id FROM users WHERE role='admin' LIMIT 1").get() as any;
  const firstUser = db.prepare("SELECT id FROM users LIMIT 1").get() as any;
  const targetUserId = adminUser?.id || firstUser?.id;

  if (!targetUserId) {
    console.log('[Stems Migration] No users found — skipping migration');
    return;
  }

  // Check for stale stems (flat directories directly under stems/)
  const entries = fs.readdirSync(stemsBaseDir, { withFileTypes: true });
  const staleDirs = entries.filter(e => e.isDirectory());

  if (staleDirs.length === 0) {
    return; // No stale stems
  }

  // Check if target user directory already exists
  const targetUserDir = path.join(stemsBaseDir, targetUserId);
  if (!fs.existsSync(targetUserDir)) {
    fs.mkdirSync(targetUserDir, { recursive: true });
  }

  let migrated = 0;
  for (const entry of staleDirs) {
    // Skip if this looks like a user ID directory (has _meta.json with userId)
    const metaPath = path.join(stemsBaseDir, entry.name, '_meta.json');
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
    const srcPath = path.join(stemsBaseDir, entry.name);
    const destPath = path.join(targetUserDir, entry.name);

    try {
      fs.renameSync(srcPath, destPath);
      migrated++;
      console.log(`[Stems Migration] Migrated ${entry.name} → ${targetUserId}/${entry.name}`);
    } catch (err: any) {
      console.error(`[Stems Migration] Failed to migrate ${entry.name}: ${err.message}`);
    }
  }

  if (migrated > 0) {
    console.log(`[Stems Migration] Complete: migrated ${migrated} stale stems to user ${targetUserId}`);
  }
}

// Run if called directly
if (require.main === module) {
  // Initialize db for standalone execution
  const Database = require('better-sqlite3');
  const dbPath = path.join(config.data.dir, 'hotstep.db');
  const db = new Database(dbPath);
  console.log('Running stem migration...');
  migrateStems(db);
  db.close();
  console.log('Done');
}
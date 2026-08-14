// auth.ts — Authentication service: password hashing, JWT, admin bootstrap
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import { getDb } from '../db/database.js';
import { config } from '../config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── JWT Secret Management ──────────────────────────────────────────────────
// Priority: env JWT_SECRET → server/data/jwt-secret → generate & persist

const JWT_SECRET_FILE = path.join(config.data.dir, 'jwt-secret');

function getJwtSecret(): string {
  // 1. Check env
  if (process.env.JWT_SECRET) return process.env.JWT_SECRET;

  // 2. Check persisted file
  if (fs.existsSync(JWT_SECRET_FILE)) {
    return fs.readFileSync(JWT_SECRET_FILE, 'utf-8').trim();
  }

  // 3. Generate and persist
  const secret = crypto.randomBytes(32).toString('hex');
  fs.mkdirSync(path.dirname(JWT_SECRET_FILE), { recursive: true });
  fs.writeFileSync(JWT_SECRET_FILE, secret, 'utf-8');
  console.log('[Auth] Generated new JWT secret');
  return secret;
}

const JWT_SECRET = getJwtSecret();
const JWT_EXPIRES_IN = '7d'; // 7 days, refreshable

// ── Password Hashing ───────────────────────────────────────────────────────

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 10);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// ── JWT Operations ─────────────────────────────────────────────────────────

export interface JwtPayload {
  userId: string;
  role: 'admin' | 'user';
  username: string;
}

export function signJwt(payload: JwtPayload): string {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

export function verifyJwt(token: string): JwtPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as JwtPayload;
  } catch {
    return null;
  }
}

// ── Admin Bootstrap ────────────────────────────────────────────────────────

/** Generate a random password (12 chars, alphanumeric + symbols) */
function generatePassword(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789!@#$%&*';
  const arr = new Uint8Array(12);
  crypto.getRandomValues(arr);
  return Array.from(arr, b => chars[b % chars.length]).join('');
}

/**
 * Ensure an admin user exists. On first launch, creates "admin" with a
 * generated password and logs it. On subsequent runs, does nothing.
 * Also migrates legacy "Producer" user to admin if found.
 */
export async function ensureAdmin(): Promise<{ created: boolean; username: string; password?: string }> {
  const db = getDb();

  // Check for existing admin
  const existingAdmin = db.prepare(
    "SELECT * FROM users WHERE role = 'admin'"
  ).get() as any;

  if (existingAdmin) {
    console.log(`[Auth] Admin user exists: ${existingAdmin.username}`);
    return { created: false, username: existingAdmin.username };
  }

  // Check for legacy "Producer" user (from auto-auth era) and upgrade
  const legacyUser = db.prepare(
    "SELECT * FROM users WHERE username = 'Producer' LIMIT 1"
  ).get() as any;

  if (legacyUser) {
    const password = generatePassword();
    const hash = await hashPassword(password);

    db.prepare(
      `UPDATE users SET role = 'admin', password_hash = ? WHERE id = ?`
    ).run(hash, legacyUser.id);

    console.log('');
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║          HOT-Step CPP — First Launch Setup               ║');
    console.log('╠══════════════════════════════════════════════════════════╣');
    console.log(`║  Your admin account has been created:                    ║`);
    console.log(`║                                                          ║`);
    console.log(`║  Username: Producer                                      ║`);
    console.log(`║  Password: ${password.padEnd(42)}║`);
    console.log(`║                                                          ║`);
    console.log(`║  ⚠  Save this password! It won't be shown again.         ║`);
    console.log(`║                                                          ║`);
    console.log(`║  To reset later, set ADMIN_RESET_PASSWORD in .env        ║`);
    console.log('╚══════════════════════════════════════════════════════════╝');
    console.log('');

    return { created: false, username: 'Producer', password };
  }

  // Create fresh admin
  const password = generatePassword();
  const hash = await hashPassword(password);
  const id = crypto.randomUUID();

  db.prepare(
    `INSERT INTO users (id, username, role, password_hash) VALUES (?, ?, 'admin', ?)`
  ).run(id, 'admin', hash);

  console.log('');
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║          HOT-Step CPP — First Launch Setup               ║');
  console.log('╠══════════════════════════════════════════════════════════╣');
  console.log(`║  Your admin account has been created:                    ║`);
  console.log(`║                                                          ║`);
  console.log(`║  Username: admin                                         ║`);
  console.log(`║  Password: ${password.padEnd(42)}║`);
  console.log(`║                                                          ║`);
  console.log(`║  ⚠  Save this password! It won't be shown again.         ║`);
  console.log(`║                                                          ║`);
  console.log(`║  To reset later, set ADMIN_RESET_PASSWORD in .env        ║`);
  console.log('╚══════════════════════════════════════════════════════════╝');
  console.log('');

  return { created: true, username: 'admin', password };
}

/**
 * Check for ADMIN_RESET_PASSWORD in .env and apply it if present.
 * Consumes the value (clears from .env) after use.
 */
export async function applyAdminPasswordReset(): Promise<boolean> {
  const resetPassword = process.env.ADMIN_RESET_PASSWORD;
  if (!resetPassword || resetPassword.trim() === '') {
    return false;
  }

  const db = getDb();
  const admin = db.prepare(
    "SELECT * FROM users WHERE role = 'admin' LIMIT 1"
  ).get() as any;

  if (!admin) {
    console.warn('[Auth] ADMIN_RESET_PASSWORD set but no admin user found');
    return false;
  }

  const hash = await hashPassword(resetPassword);
  db.prepare("UPDATE users SET password_hash = ? WHERE id = ?").run(hash, admin.id);

  console.log(`[Auth] Admin password reset for user: ${admin.username}`);

  // Clear ADMIN_RESET_PASSWORD from .env
  try {
    const { ENV_FILE_PATH } = await import('../config.js');
    if (fs.existsSync(ENV_FILE_PATH)) {
      const content = fs.readFileSync(ENV_FILE_PATH, 'utf-8');
      const lines = content.split(/\r?\n/);
      const newLines = lines.filter(line => !line.trim().startsWith('ADMIN_RESET_PASSWORD='));
      fs.writeFileSync(ENV_FILE_PATH, newLines.join('\n'), 'utf-8');
      console.log('[Auth] Cleared ADMIN_RESET_PASSWORD from .env');
    }
  } catch (err: any) {
    console.warn(`[Auth] Could not clear ADMIN_RESET_PASSWORD from .env: ${err.message}`);
  }

  return true;
}
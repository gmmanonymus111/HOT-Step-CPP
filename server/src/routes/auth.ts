// auth.ts — Authentication routes: login, logout, register, user management
import { Router, Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { getDb } from '../db/database.js';
import { hashPassword, verifyPassword, signJwt, verifyJwt } from '../services/auth.js';
import { requireAuth, requireAdmin } from '../middleware/authMiddleware.js';

const router = Router();

// ── Public Routes ──────────────────────────────────────────────────────────

/**
 * POST /api/auth/login
 * Body: { username, password }
 * Returns: { user, token } + sets httpOnly cookie
 */
router.post('/login', async (req: Request, res: Response) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      res.status(400).json({ error: 'Username and password required' });
      return;
    }

    const db = getDb();
    const user = db.prepare(
      "SELECT id, username, password_hash, role, settings FROM users WHERE username = ?"
    ).get(username) as any;

    if (!user || !user.password_hash) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    const valid = await verifyPassword(password, user.password_hash);
    if (!valid) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    const token = signJwt({
      userId: user.id,
      role: user.role,
      username: user.username,
    });

    // Sanitize user response (no password hash)
    const userResponse = {
      id: user.id,
      username: user.username,
      role: user.role,
      settings: user.settings ? JSON.parse(user.settings || '{}') : {},
      created_at: user.created_at,
    };

    // Set httpOnly cookie (7 days, matches JWT expiry)
    res.cookie('hs_token', token, {
      httpOnly: true,
      secure: false, // will be true when SSL is enabled
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 * 1000,
      path: '/',
    });

    res.json({ user: userResponse, token });
  } catch (err: any) {
    console.error('[Auth] Login error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/auth/me
 * Returns current user from cookie or Bearer token
 */
router.get('/me', (req: Request, res: Response) => {
  let token: string | null = null;

  // Check cookie
  if (req.cookies && req.cookies.hs_token) {
    token = req.cookies.hs_token;
  }

  // Check Authorization header
  if (!token) {
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      token = authHeader.slice(7);
    }
  }

  if (!token) {
    res.status(401).json({ error: 'Not authenticated' });
    return;
  }

  const payload = verifyJwt(token);
  if (!payload) {
    res.status(401).json({ error: 'Invalid or expired token' });
    return;
  }

  const db = getDb();
  const user = db.prepare(
    "SELECT id, username, role, settings, created_at FROM users WHERE id = ?"
  ).get(payload.userId) as any;

  if (!user) {
    res.status(404).json({ error: 'User not found' });
    return;
  }

  res.json({
    user: {
      id: user.id,
      username: user.username,
      role: user.role,
      settings: user.settings ? JSON.parse(user.settings || '{}') : {},
      created_at: user.created_at,
    },
  });
});

// ── Authenticated Routes ───────────────────────────────────────────────────

/**
 * POST /api/auth/logout
 * Clears the auth cookie
 */
router.post('/logout', requireAuth, (_req: Request, res: Response) => {
  res.clearCookie('hs_token', { path: '/' });
  res.json({ success: true });
});

/**
 * PATCH /api/auth/settings
 * Update current user's settings
 */
router.patch('/settings', requireAuth, (req: Request, res: Response) => {
  try {
    const settings = req.body.settings;
    if (settings === undefined) {
      res.status(400).json({ error: 'Missing "settings" object' });
      return;
    }

    const db = getDb();
    db.prepare(
      "UPDATE users SET settings = ? WHERE id = ?"
    ).run(JSON.stringify(settings), req.user!.userId);

    const updated = db.prepare(
      "SELECT settings FROM users WHERE id = ?"
    ).get(req.user!.userId) as any;

    res.json({
      settings: updated.settings ? JSON.parse(updated.settings) : {},
    });
  } catch (err: any) {
    console.error('[Auth] Settings update error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PATCH /api/auth/username
 * Update current user's username
 */
router.patch('/username', requireAuth, (req: Request, res: Response) => {
  try {
    const { username } = req.body;
    if (!username || typeof username !== 'string' || username.trim().length === 0) {
      res.status(400).json({ error: 'Username is required' });
      return;
    }

    const db = getDb();
    const trimmed = username.trim();

    // Check uniqueness
    const existing = db.prepare(
      "SELECT id FROM users WHERE username = ? AND id != ?"
    ).get(trimmed, req.user!.userId) as any;

    if (existing) {
      res.status(409).json({ error: 'Username already taken' });
      return;
    }

    db.prepare("UPDATE users SET username = ? WHERE id = ?").run(trimmed, req.user!.userId);

    const updated = db.prepare(
      "SELECT id, username, role, settings, created_at FROM users WHERE id = ?"
    ).get(req.user!.userId) as any;

    // Issue new token with updated username
    const token = signJwt({
      userId: updated.id,
      role: updated.role,
      username: updated.username,
    });

    res.cookie('hs_token', token, {
      httpOnly: true,
      secure: false,
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 * 1000,
      path: '/',
    });

    res.json({
      user: {
        id: updated.id,
        username: updated.username,
        role: updated.role,
        settings: updated.settings ? JSON.parse(updated.settings || '{}') : {},
        created_at: updated.created_at,
      },
      token,
    });
  } catch (err: any) {
    console.error('[Auth] Username update error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/auth/password
 * Change current user's password
 */
router.post('/password', requireAuth, async (req: Request, res: Response) => {
  try {
    const { currentPassword, newPassword } = req.body;

    if (!currentPassword || !newPassword) {
      res.status(400).json({ error: 'Current password and new password required' });
      return;
    }

    const db = getDb();
    const user = db.prepare(
      "SELECT password_hash FROM users WHERE id = ?"
    ).get(req.user!.userId) as any;

    const valid = await verifyPassword(currentPassword, user.password_hash);
    if (!valid) {
      res.status(401).json({ error: 'Current password is incorrect' });
      return;
    }

    const hash = await hashPassword(newPassword);
    db.prepare("UPDATE users SET password_hash = ? WHERE id = ?").run(hash, req.user!.userId);

    res.json({ success: true });
  } catch (err: any) {
    console.error('[Auth] Password change error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Admin-Only Routes ──────────────────────────────────────────────────────

/**
 * GET /api/auth/users
 * List all users (admin only)
 */
router.get('/users', requireAuth, requireAdmin, (_req: Request, res: Response) => {
  try {
    const db = getDb();
    const users = db.prepare(
      "SELECT id, username, role, created_at FROM users ORDER BY created_at ASC"
    ).all() as any[];

    res.json({ users });
  } catch (err: any) {
    console.error('[Auth] List users error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/auth/users
 * Create a new user (admin only)
 */
router.post('/users', requireAuth, requireAdmin, async (req: Request, res: Response) => {
  try {
    const { username, password, role } = req.body;

    if (!username || !password) {
      res.status(400).json({ error: 'Username and password required' });
      return;
    }

    const userRole = role === 'admin' ? 'admin' : 'user';
    const db = getDb();

    // Check uniqueness
    const existing = db.prepare(
      "SELECT id FROM users WHERE username = ?"
    ).get(username.trim()) as any;

    if (existing) {
      res.status(409).json({ error: 'Username already taken' });
      return;
    }

    const id = uuidv4();
    const hash = await hashPassword(password);

    db.prepare(
      "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)"
    ).run(id, username.trim(), hash, userRole);

    const user = db.prepare(
      "SELECT id, username, role, created_at FROM users WHERE id = ?"
    ).get(id) as any;

    res.status(201).json({ user });
  } catch (err: any) {
    console.error('[Auth] Create user error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PATCH /api/auth/users/:id
 * Update a user (admin only)
 */
router.patch('/users/:id', requireAuth, requireAdmin, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const { username, password, role } = req.body;

    const db = getDb();
    const user = db.prepare("SELECT * FROM users WHERE id = ?").get(id) as any;

    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    const updates: string[] = [];
    const params: any[] = [];

    if (username !== undefined && username.trim()) {
      const existing = db.prepare(
        "SELECT id FROM users WHERE username = ? AND id != ?"
      ).get(username.trim(), id) as any;
      if (existing) {
        res.status(409).json({ error: 'Username already taken' });
        return;
      }
      updates.push("username = ?");
      params.push(username.trim());
    }

    if (password !== undefined && password !== '') {
      const hash = await hashPassword(password);
      updates.push("password_hash = ?");
      params.push(hash);
    }

    if (role !== undefined && (role === 'admin' || role === 'user')) {
      updates.push("role = ?");
      params.push(role);
    }

    if (updates.length === 0) {
      res.json({ user });
      return;
    }

    params.push(id);
    db.prepare(`UPDATE users SET ${updates.join(', ')} WHERE id = ?`).run(...params);

    const updated = db.prepare(
      "SELECT id, username, role, created_at FROM users WHERE id = ?"
    ).get(id) as any;

    res.json({ user: updated });
  } catch (err: any) {
    console.error('[Auth] Update user error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * DELETE /api/auth/users/:id
 * Delete a user (admin only, cannot delete self)
 */
router.delete('/users/:id', requireAuth, requireAdmin, (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    if (id === req.user!.userId) {
      res.status(400).json({ error: 'Cannot delete yourself' });
      return;
    }

    const db = getDb();
    const user = db.prepare("SELECT * FROM users WHERE id = ?").get(id) as any;

    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    db.prepare("DELETE FROM users WHERE id = ?").run(id);
    res.json({ success: true });
  } catch (err: any) {
    console.error('[Auth] Delete user error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Helper: extract userId from request (for legacy compatibility)
export function getUserId(req: Request): string | null {
  return req.user?.userId || null;
}

export default router;
// authMiddleware.ts — JWT verification and role-based access control
import { Request, Response, NextFunction } from 'express';
import { verifyJwt } from '../services/auth.js';

// Extend Express Request to include user info
declare global {
  namespace Express {
    interface Request {
      user?: {
        userId: string;
        role: 'admin' | 'user';
        username: string;
      };
    }
  }
}

/**
 * Require valid JWT. Accepts:
 * - Cookie: hs_token
 * - Header: Authorization: Bearer <token>
 */
export function requireAuth(req: Request, res: Response, next: NextFunction): void {
  let token: string | null = null;

  // Check cookie first (preferred for security)
  if (req.cookies && req.cookies.hs_token) {
    token = req.cookies.hs_token;
  }

  // Fall back to Authorization header
  if (!token) {
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      token = authHeader.slice(7);
    }
  }

  if (!token) {
    res.status(401).json({ error: 'Authentication required' });
    return;
  }

  const payload = verifyJwt(token);
  if (!payload) {
    res.status(401).json({ error: 'Invalid or expired token' });
    return;
  }

  req.user = {
    userId: payload.userId,
    role: payload.role,
    username: payload.username,
  };

  next();
}

/**
 * Require admin role. Must be used after requireAuth.
 */
export function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  if (!req.user || req.user.role !== 'admin') {
    res.status(403).json({ error: 'Admin access required' });
    return;
  }
  next();
}
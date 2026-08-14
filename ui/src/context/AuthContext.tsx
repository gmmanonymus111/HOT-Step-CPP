// AuthContext.tsx — Authentication context with JWT support
import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { authApi } from '../services/api';
import { useGlobalParamsStore } from '../stores/globalParamsStore';
import { startSettingsSync, stopSettingsSync } from '../stores/settingsSync';
import type { User } from '../types';

interface AuthContextValue {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isAdmin: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  updateUsername: (name: string) => Promise<void>;
  updateSettings: (settings: Record<string, unknown>) => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
  isAdmin: false,
  login: async () => {},
  logout: async () => {},
  updateUsername: async () => {},
  updateSettings: async () => {},
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const isAuthenticated = !!user;
  const isAdmin = (user as any)?.role === 'admin';

  // Check if user is authenticated on mount (via cookie)
  useEffect(() => {
    let cancelled = false;

    const checkAuth = async () => {
      try {
        const { user } = await authApi.getMe();
        if (!cancelled && user) {
          setUser(user);
          setToken(localStorage.getItem('hs_token') || null);
        }
      } catch {
        // Not authenticated — that's fine, user will see login page
        if (!cancelled) {
          setUser(null);
          setToken(null);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    checkAuth();
    return () => { cancelled = true; };
  }, []);

  // Redirect to login if not authenticated and not already on login page
  useEffect(() => {
    if (!isLoading && !isAuthenticated && location.pathname !== '/login') {
      navigate('/login', { replace: true });
    }
  }, [isLoading, isAuthenticated, location.pathname, navigate]);

  const login = useCallback(async (username: string, password: string) => {
    const response = await authApi.login(username, password);
    setUser(response.user);
    setToken(response.token);
    localStorage.setItem('hs_token', response.token);

    // Load user's server-side settings and merge into Zustand store
    const userData = response.user as any;
    if (userData?.settings && typeof userData.settings === 'object') {
      const serverSettings = userData.settings as Record<string, unknown>;
      const store = useGlobalParamsStore.getState();
      const updates: Record<string, unknown> = {};

      // Merge only keys that exist in both the store and server settings
      for (const [key, value] of Object.entries(serverSettings)) {
        if (key in store && value !== null && value !== undefined) {
          updates[key] = value;
        }
      }

      if (Object.keys(updates).length > 0) {
        useGlobalParamsStore.setState(updates);
        console.log(`[Auth] Loaded ${Object.keys(updates).length} settings from server for ${username}`);
      }
    }

    // Start background sync of settings to server
    startSettingsSync(response.token);
  }, []);

  const logout = useCallback(async () => {
    // Stop background sync first
    stopSettingsSync();

    // Persist current store state to server before logging out
    if (token && user) {
      try {
        const store = useGlobalParamsStore.getState();
        const settingsToSave: Record<string, unknown> = {};

        for (const key of Object.keys(store)) {
          if (typeof store[key] !== 'function') {
            settingsToSave[key] = store[key];
          }
        }

        await authApi.updateSettings(settingsToSave, token);
      } catch {
        // Ignore — logout proceeds anyway
      }
    }

    try {
      await authApi.logout();
    } catch {
      // Ignore errors — still clear local state
    }
    setUser(null);
    setToken(null);
    localStorage.removeItem('hs_token');
    navigate('/login', { replace: true });
  }, [navigate, token, user]);

  const updateUsername = useCallback(async (name: string) => {
    if (!token) return;
    const { user: updated, token: newToken } = await authApi.updateUsername(name, token);
    if (updated) {
      setUser(updated);
      setToken(newToken);
      localStorage.setItem('hs_token', newToken);
    }
  }, [token]);

  const updateSettings = useCallback(async (settings: Record<string, unknown>) => {
    if (!token) return;
    const { settings: updated } = await authApi.updateSettings(settings, token);
    if (updated) {
      setUser(prev => prev ? { ...prev, settings: updated } : null);
    }
  }, [token]);

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated: !!user, isLoading, isAdmin, login, logout, updateUsername, updateSettings }}>
      {children}
    </AuthContext.Provider>
  );
};

// Protected route component
export const ProtectedRoute: React.FC<{ children: React.ReactNode; adminOnly?: boolean }> = ({ children, adminOnly }) => {
  const { isAuthenticated, isAdmin, isLoading } = useAuth();
  const navigate = useNavigate();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">⚡</div>
          <div className="text-lg text-zinc-400">Loading...</div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    navigate('/login', { replace: true });
    return null;
  }

  if (adminOnly && !isAdmin) {
    navigate('/', { replace: true });
    return null;
  }

  return <>{children}</>;
};
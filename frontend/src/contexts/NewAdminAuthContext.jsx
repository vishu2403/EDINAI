import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { newAdminAuthAPI } from '../utils/newAdminApi';

const STORAGE_TOKEN_KEY = 'new_admin_token';
const STORAGE_USER_KEY = 'new_admin_user';

const NewAdminAuthContext = createContext(null);

export const useNewAdminAuth = () => {
  const context = useContext(NewAdminAuthContext);
  if (!context) {
    throw new Error('useNewAdminAuth must be used within a NewAdminAuthProvider');
  }
  return context;
};

export const NewAdminAuthProvider = ({ children }) => {
  const [admin, setAdmin] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem(STORAGE_TOKEN_KEY);
    const storedUser = localStorage.getItem(STORAGE_USER_KEY);

    if (storedToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setToken(storedToken);
        setAdmin(parsedUser);
      } catch (error) {
        console.error('Failed to parse cached new admin user:', error);
        localStorage.removeItem(STORAGE_TOKEN_KEY);
        localStorage.removeItem(STORAGE_USER_KEY);
      }
    }

    setLoading(false);
  }, []);

  const login = async (email, password) => {
    try {
      setLoading(true);
      const response = await newAdminAuthAPI.login(email, password);
      const { status: ok, data: payload, message: responseMessage } = response || {};

      if (!ok || !payload) {
        throw new Error(responseMessage || 'Invalid response from admin login API');
      }

      const accessToken = payload?.access_token ?? payload?.token ?? null;
      const refreshToken = payload?.refresh_token ?? null;

      if (!accessToken) {
        throw new Error('Invalid response from admin login API');
      }

      const userType = payload?.user_type ?? payload?.role ?? null;
      const normalizedMessage = payload?.message ?? responseMessage ?? 'Login successful';

      if (userType === 'member') {
        const memberPayload = {
          admin_id: payload?.admin_id ?? null,
          member_id: payload?.member_id ?? payload?.id ?? null,
          work_type: payload?.work_type ?? null,
          login_status: payload?.login_status ?? null,
          message: normalizedMessage,
          email,
          token: accessToken,
          refresh_token: refreshToken,
        };

        return {
          success: true,
          userType: 'member',
          member: memberPayload,
          accessToken,
          refreshToken,
        };
      }

      const adminId = payload?.admin_id ?? payload?.id;

      if (!adminId) {
        throw new Error('Invalid response from admin login API');
      }

      const contactExists = payload?.contact_exists ?? true;

      const adminPayload = {
        ...payload,
        admin_id: adminId,
        login_status: payload?.login_status,
        message: normalizedMessage,
        contact_exists: contactExists,
        token: accessToken,
        refresh_token: refreshToken,
      };

      localStorage.setItem(STORAGE_TOKEN_KEY, accessToken);
      localStorage.setItem(STORAGE_USER_KEY, JSON.stringify(adminPayload));

      setToken(accessToken);
      setAdmin(adminPayload);

      toast.success(normalizedMessage || 'Welcome back!');
      return { success: true, userType: 'admin', admin: adminPayload, token: accessToken, refreshToken };
    } catch (error) {
      const message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Admin login failed';
      toast.error(message);
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  };

  const register = async (payload) => {
    try {
      setLoading(true);
      const response = await newAdminAuthAPI.register(payload);
      toast.success(response?.message || 'Registration successful. Please log in.');
      return { success: true, data: response };
    } catch (error) {
      const message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Admin registration failed';
      toast.error(message);
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  };

  const changePassword = async ({ oldPassword, newPassword }) => {
    try {
      const response = await newAdminAuthAPI.changePassword({ oldPassword, newPassword });
      const success = Boolean(response?.status ?? response?.success ?? false);
      const message = response?.message || 'Password updated successfully.';

      if (!success) {
        throw new Error(message);
      }

      toast.success(message);
      return { success: true, message };
    } catch (error) {
      const message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Unable to update password.';
      toast.error(message);
      return { success: false, error: message };
    }
  };

  const logout = async () => {
    try {
      await newAdminAuthAPI.logout();
    } catch (error) {
      // Ignore API errors during logout, simply clear local state
      console.warn('Admin logout API failed:', error);
    }

    localStorage.removeItem(STORAGE_TOKEN_KEY);
    localStorage.removeItem(STORAGE_USER_KEY);
    setToken(null);
    setAdmin(null);
    toast.success('Logged out');
  };

  const updateAdmin = (updates) => {
    setAdmin((prev) => {
      const next = { ...(prev || {}), ...updates };
      if (next && Object.keys(next).length > 0) {
        localStorage.setItem(STORAGE_USER_KEY, JSON.stringify(next));
      }
      return next;
    });
  };

  const value = useMemo(
    () => ({
      admin,
      token,
      loading,
      isAuthenticated: Boolean(admin && token),
      login,
      register,
      changePassword,
      logout,
      updateAdmin,
    }),
    [admin, token, loading]
  );

  return (
    <NewAdminAuthContext.Provider value={value}>
      {children}
    </NewAdminAuthContext.Provider>
  );
};
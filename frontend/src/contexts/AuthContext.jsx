/**
 * Authentication Context for INAI Education Management System
 * Manages user authentication state, JWT tokens, and role-based access
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../utils/api';
import toast from 'react-hot-toast';

const AuthContext = createContext({});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize auth state from localStorage
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const storedToken = localStorage.getItem('inai_token');
        const storedUser = localStorage.getItem('inai_user');

        if (storedToken && storedUser) {
          let parsedUser;
          try {
            parsedUser = JSON.parse(storedUser);
          } catch (parseError) {
            console.error('Failed to parse stored auth user:', parseError);
            logout();
            return;
          }

          setToken(storedToken);
          setUser(parsedUser);
          setIsAuthenticated(true);

          const shouldValidateToken = parsedUser?.role === 'admin';
          if (shouldValidateToken) {
            try {
              const response = await authAPI.getCurrentUser();
              if (response.status) {
                const userData = response.data;
                setUser(userData);
                localStorage.setItem('inai_user', JSON.stringify(userData));
              } else {
                throw new Error('Invalid token');
              }
            } catch (error) {
              logout();
            }
          }
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
        logout();
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (email, password) => {
    try {
      setLoading(true);
      const response = await authAPI.login(email, password);
      
      if (response.status && response.data) {
        const { token: newToken, ...userData } = response.data;

        const allowedRoles = ['admin', 'member'];
        if (!allowedRoles.includes(userData?.role)) {
          throw new Error('Unsupported account. Please contact support.');
        }

        // Store auth data
        localStorage.setItem('inai_token', newToken);
        localStorage.setItem('inai_user', JSON.stringify(userData));

        // Update state
        setToken(newToken);
        setUser(userData);
        setIsAuthenticated(true);

        toast.success(userData.role === 'member' ? 'Member login successful!' : 'Login successful!');
        return { success: true, data: userData };
      } else {
        throw new Error(response.message || 'Login failed');
      }
    } catch (error) {
      const message = error.response?.data?.message || error.message || 'Login failed';
      toast.error(message);
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    // Clear storage
    localStorage.removeItem('inai_token');
    localStorage.removeItem('inai_user');
    
    // Clear state
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
    
    toast.success('Logged out successfully');
  };

  const updateToken = (newToken) => {
    localStorage.setItem('inai_token', newToken);
    setToken(newToken);
  };

  const updateUser = (userData) => {
    localStorage.setItem('inai_user', JSON.stringify(userData));
    setUser(userData);
  };

  const changePassword = async (oldPassword, newPassword) => {
    try {
      const response = await authAPI.changePassword(oldPassword, newPassword);
      if (response.status) {
        toast.success('Password changed successfully!');
        return { success: true };
      } else {
        throw new Error(response.message || 'Password change failed');
      }
    } catch (error) {
      const message = error.response?.data?.message || error.message || 'Password change failed';
      toast.error(message);
      return { success: false, error: message };
    }
  };

  const bootstrapSession = ({ token: sessionToken, user: sessionUser }) => {
    if (!sessionToken || !sessionUser) {
      return;
    }

    localStorage.setItem('inai_token', sessionToken);
    localStorage.setItem('inai_user', JSON.stringify(sessionUser));

    setToken(sessionToken);
    setUser(sessionUser);
    setIsAuthenticated(true);
  };

  // Role-based access helpers
  const isAdmin = () => user?.role === 'admin';
  const isSuperAdmin = () => user?.role === 'admin' && user?.is_super_admin === true;
  const isMember = () => user?.role === 'member';
  const hasInaiCredentials = () => user?.has_inai_credentials === true;
  const needsOnboarding = () => isAdmin() && !hasInaiCredentials();

  const normalizeMemberWorkType = (workType) => {
    if (!workType) return null;

    switch (workType) {
      case 'chapter':
      case 'chapter_management':
        return 'chapter';
      case 'student':
      case 'student_management':
        return 'student';
      case 'lecture':
      case 'lecture_management':
        return 'lecture';
      default:
        return workType;
    }
  };

  const getMemberWorkType = () => {
    if (!isMember()) return null;
    return normalizeMemberWorkType(user?.work_type);
  };

  const isChapterMember = () => getMemberWorkType() === 'chapter';
  const isStudentMember = () => getMemberWorkType() === 'student';
  const isLectureMember = () => getMemberWorkType() === 'lecture';

  // Get appropriate dashboard route based on user role
  const getDashboardRoute = () => {
    if (!isAuthenticated || !user) return '/login';
    
    if (isAdmin()) {
      return needsOnboarding() ? '/complete-onboarding' : '/dev_admin';
    }
    
    if (isMember()) {
      const workType = getMemberWorkType();

      const routeMap = {
        chapter: '/chapter-dashboard',
        student: '/student-dashboard',
        lecture: '/lecture-dashboard',
      };

      return routeMap[workType] || '/login';
    }
    
    return '/login';
  };

  // Get next onboarding step
  const getNextOnboardingStep = () => {
    if (!isAdmin()) return '/login';
    return needsOnboarding() ? '/complete-onboarding' : '/dev_admin';
  };

  // Check if user has permission for a specific action
  const hasPermission = (permission) => {
    switch (permission) {
      case 'manage_admins':
        return isSuperAdmin();
      case 'manage_members':
        return isAdmin();
      case 'access_admin_dashboard':
        return isAdmin() && hasInaiCredentials();
      case 'access_onboarding':
        return isAdmin() && !hasInaiCredentials();
      case 'access_chapter_dashboard':
        return isChapterMember();
      case 'access_student_dashboard':
        return isStudentMember();
      case 'access_lecture_dashboard':
        return isLectureMember();
      default:
        return false;
    }
  };

  const value = {
    // State
    user,
    token,
    loading,
    isAuthenticated,
    
    // Actions
    login,
    logout,
    updateToken,
    updateUser,
    changePassword,
    bootstrapSession,
    
    // Role helpers
    isAdmin,
    isSuperAdmin,
    isMember,
    hasInaiCredentials,
    needsOnboarding,
    getMemberWorkType,
    isChapterMember,
    isStudentMember,
    isLectureMember,
    
    // Navigation helpers
    getDashboardRoute,
    getNextOnboardingStep,
    hasPermission,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

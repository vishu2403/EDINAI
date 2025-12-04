/**
 * Main App component for INAI Education Management System
 * Handles routing, protected routes, and layout
 */

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import { useNewAdminAuth } from './contexts/NewAdminAuthContext.jsx';
import LoadingSpinner from './components/ui/LoadingSpinner';

// Import page components
import LoginPage from './pages/auth/LoginPage';
import CompleteOnboardingPage from './pages/onboarding/CompleteOnboardingPage';
import AdminDashboard from './pages/super-administration/AdminDashboard';
import AdminManagement from './pages/super-administration/AdminManagement';
import AddAdmin from './pages/super-administration/AddAdmin';
import ChapterDashboard from './pages/member/ChapterDashboard';
import StudentDashboard from './pages/member/StudentDashboard';
import LectureDashboard from './pages/member/LectureDashboard';
import SelectedMaterials from './pages/member/SelectedMaterials';
import LectureEditor from './pages/member/LectureEditor';
import GenerateLecture from './pages/member/GenerateLecture.jsx';
import EditChapter from './pages/member/EditChapter.jsx';
import SchoolPortalPage from './pages/portal/SchoolPortalPage';
import StudentPortalDashboard from './pages/portal/StudentPortalDashboard';
import NotFoundPage from './pages/NotFoundPage';
import RoleSelectionPage from './pages/RoleSelectionPage.jsx';
import AdminLogin from './pages/new-admin/AdminLogin.jsx';
import AdminRegister from './pages/new-admin/AdminRegister.jsx';
import AdminMembersPage from './pages/new-admin/AdminMembersPage.jsx';
import AdminOnboarding from './pages/new-admin/AdminOnboarding.jsx';
import NewAdminDashboard from './pages/new-admin/AdminDashboard.jsx';
import AdminAllMembers from './pages/new-admin/AdminAllMembers.jsx';
import AdminResetPassword from './pages/new-admin/AdminResetPassword.jsx';

// Protected Route Component
const ProtectedRoute = ({ children, requireAuth = true, permission = null }) => {
  const { loading, isAuthenticated, hasPermission } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (requireAuth && !isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (permission && !hasPermission(permission)) {
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
};

const NewAdminProtectedRoute = ({ children, requireOnboardingComplete = false }) => {
  const { loading, isAuthenticated, admin } = useNewAdminAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/admin/login" replace />;
  }

  if (requireOnboardingComplete && !admin?.contact_exists) {
    return <Navigate to="/admin/onboarding" replace />;
  }

  return children;
};

// Onboarding Route - Only accessible to admins without credentials
const OnboardingRoute = ({ children }) => {
  const { loading, isAuthenticated, isAdmin, hasInaiCredentials } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (!isAdmin()) {
    return <Navigate to="/unauthorized" replace />;
  }

  if (hasInaiCredentials()) {
    return <Navigate to="/dev_admin" replace />;
  }

  return children;
};

// Dashboard Route - Redirects to appropriate dashboard
const DashboardRoute = () => {
  const { getDashboardRoute } = useAuth();
  const dashboardRoute = getDashboardRoute();
  return <Navigate to={dashboardRoute} replace />;
};

// Public Route - Only accessible when not authenticated
const PublicRoute = ({ children }) => {
  const { loading, isAuthenticated, getDashboardRoute } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (isAuthenticated) {
    const dashboardRoute = getDashboardRoute();
    return <Navigate to={dashboardRoute} replace />;
  }

  return children;
};

// Unauthorized Page Component
const UnauthorizedPage = () => {
  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4">
      <div className="card p-8 text-center max-w-md w-full">
        <div className="text-6xl mb-4">🚫</div>
        <h1 className="text-2xl font-bold text-dark-text-primary mb-2">
          Access Denied
        </h1>
        <p className="text-dark-text-secondary mb-6">
          You don't have permission to access this page.
        </p>
        <button
          onClick={() => window.history.back()}
          className="btn-primary"
        >
          Go Back
        </button>
      </div>
    </div>
  );
};

function App() {
  const { loading } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      <Routes>
        {/* Role Selection / Public Routes */}
        <Route path="/" element={<RoleSelectionPage />} />
        <Route 
          path="/login" 
          element={
            <PublicRoute>
              <LoginPage />
            </PublicRoute>
          } 
        />
        <Route 
          path="/school-portal" 
          element={<SchoolPortalPage />} 
        />
        <Route 
          path="/school-portal/dashboard" 
          element={<StudentPortalDashboard />} 
        />

        {/* Dashboard Redirect */}
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <DashboardRoute />
            </ProtectedRoute>
          } 
        />

        {/* Onboarding Route (Admin only, pre-credentials) */}
        <Route 
          path="/complete-onboarding" 
          element={
            <OnboardingRoute>
              <CompleteOnboardingPage />
            </OnboardingRoute>
          } 
        />

        {/* Admin Dashboard Routes */}
        <Route 
          path="/dev_admin" 
          element={
            <ProtectedRoute permission="access_admin_dashboard">
              <AdminDashboard />
            </ProtectedRoute>
          } 
        />
        
        {/* Super Admin Only Routes */}
        <Route 
          path="/admin-management" 
          element={
            <ProtectedRoute permission="manage_admins">
              <AdminManagement />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/add-admin" 
          element={
            <ProtectedRoute permission="manage_admins">
              <AddAdmin />
            </ProtectedRoute>
          } 
        />

        {/* New Admin Portal Routes */}
        <Route path="/admin/login" element={<AdminLogin />} />
        <Route path="/admin/register" element={<AdminRegister />} />
        <Route path="/admin/reset-password" element={<AdminResetPassword />} />
        <Route
          path="/admin/onboarding"
          element={
            <NewAdminProtectedRoute>
              <AdminOnboarding />
            </NewAdminProtectedRoute>
          }
        />
        <Route
          path="/admin/members"
          element={
            <NewAdminProtectedRoute requireOnboardingComplete>
              <AdminMembersPage />
            </NewAdminProtectedRoute>
          }
        />
        <Route
          path="/admin/dashboard"
          element={
            <NewAdminProtectedRoute requireOnboardingComplete>
              <NewAdminDashboard />
            </NewAdminProtectedRoute>
          }
        />
        <Route
          path="/admin/all-members"
          element={
            <NewAdminProtectedRoute requireOnboardingComplete>
              <AdminAllMembers />
            </NewAdminProtectedRoute>
          }
        />

        {/* Member Dashboard Routes */}
        <Route 
          path="/chapter-dashboard" 
          element={
            <ProtectedRoute permission="access_chapter_dashboard">
              <ChapterDashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/student-dashboard" 
          element={
            <ProtectedRoute permission="access_student_dashboard">
              <StudentDashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/lecture-dashboard" 
          element={
            <ProtectedRoute permission="access_lecture_dashboard">
              <LectureDashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/selected-materials" 
          element={
            <ProtectedRoute permission="access_chapter_dashboard">
              <SelectedMaterials />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/lecture-editor" 
          element={
            <ProtectedRoute permission="access_chapter_dashboard">
              <LectureEditor />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/generate-lecture" 
          element={
            <ProtectedRoute permission="access_chapter_dashboard">
              <GenerateLecture />
            </ProtectedRoute>
          } 
        />
        <Route
          path="/chapter-dashboard/edit"
          element={
            <ProtectedRoute permission="access_chapter_dashboard">
              <EditChapter />
            </ProtectedRoute>
          }
        />

        {/* Special Routes */}
        <Route path="/unauthorized" element={<UnauthorizedPage />} />
        <Route path="/404" element={<NotFoundPage />} />

        {/* Catch all - 404 */}
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Routes>
    </div>
  );
}

export default App;

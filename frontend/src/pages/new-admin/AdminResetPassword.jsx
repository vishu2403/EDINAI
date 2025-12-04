import React, { useState, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Lock, ShieldCheck } from 'lucide-react';
import toast from 'react-hot-toast';

import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { authAPI } from '../../utils/api';

const AdminResetPassword = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { logout } = useNewAdminAuth();

  const { token, email } = useMemo(() => {
    const searchToken = searchParams.get('token') || '';
    const searchEmail = (searchParams.get('email') || '').trim().toLowerCase();
    return {
      token: searchToken,
      email: searchEmail,
    };
  }, [searchParams]);

  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!token || !email) {
      toast.error('Reset link is invalid or missing required information.');
      return;
    }

    if (!newPassword || newPassword.length < 8) {
      toast.error('Password must be at least 8 characters long.');
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error('Passwords do not match.');
      return;
    }

    setSubmitting(true);
    try {
      const response = await authAPI.resetPassword(email, token, newPassword);
      const message = response.message || 'Password reset successfully. Please sign in with your new password.';
      toast.success(message);
      try {
        await logout();
      } catch (logoutError) {
        console.warn('Failed to clear admin session after password reset', logoutError);
      }
      navigate('/admin/login', { replace: true });
    } catch (error) {
      const message = error.response?.data?.message || error.response?.data?.detail || 'Unable to reset password. The link may be invalid or expired.';
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const resetLinkInvalid = !token || !email;

  return (
    <div className="min-h-screen bg-[#050507] flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md card p-8 border border-white/10 bg-white/[0.04] backdrop-blur-xl"
      >
        <div className="text-center mb-8 space-y-3">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary-600/20 border border-primary-400/30">
            <ShieldCheck className="w-6 h-6 text-primary-400" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold text-dark-text-primary">Set a new password</h1>
            <p className="text-sm text-dark-text-secondary">
              Create a strong password you haven&apos;t used before.
            </p>
          </div>
        </div>

        {resetLinkInvalid ? (
          <div className="rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100 space-y-3">
            <p className="font-semibold">Invalid or expired reset link</p>
            <p>
              The reset link may have expired or is missing required information. Please request a new password reset from the login page.
            </p>
            <Button
              type="button"
              variant="secondary"
              fullWidth
              onClick={() => navigate('/admin/login', { replace: true })}
            >
              Back to sign in
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-5">
            <InputField
              label="New Password"
              type="password"
              placeholder="••••••••"
              icon={Lock}
              showPasswordToggle
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              helperText="Minimum 8 characters"
            />

            <InputField
              label="Confirm Password"
              type="password"
              placeholder="••••••••"
              icon={Lock}
              showPasswordToggle
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />

            <Button
              type="submit"
              variant="primary"
              fullWidth
              loading={submitting}
              disabled={submitting}
            >
              {submitting ? 'Updating password…' : 'Reset password'}
            </Button>

            <Button
              type="button"
              variant="secondary"
              fullWidth
              onClick={() => navigate('/admin/login')}
            >
              Back to sign in
            </Button>
          </form>
        )}
      </motion.div>
    </div>
  );
};

export default AdminResetPassword;
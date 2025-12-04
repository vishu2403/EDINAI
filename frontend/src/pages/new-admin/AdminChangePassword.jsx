import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { Lock, ShieldCheck, Mail, Send, X } from 'lucide-react';

import AdminShell from '../../components/new-admin/AdminShell.jsx';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { authAPI } from '../../utils/api';

const PASSWORD_REQUIREMENTS = [
  {
    label: 'At least 6 characters long',
    test: (value) => value.length >= 6,
  },
  {
    label: 'Contains letters (A-Z)',
    test: (value) => /[A-Za-z]/.test(value),
  },
  {
    label: 'Contains numbers (0-9)',
    test: (value) => /[0-9]/.test(value),
  },
  {
    label: 'Contains special characters (!,@,#, etc.)',
    test: (value) => /[^A-Za-z0-9]/.test(value),
  },
];

const AdminChangePassword = () => {
  const navigate = useNavigate();
  const { changePassword } = useNewAdminAuth();

  const [formValues, setFormValues] = useState({
    oldPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [showForgotPanel, setShowForgotPanel] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotLoading, setForgotLoading] = useState(false);
  const [forgotSubmitted, setForgotSubmitted] = useState(false);

  const passwordScore = useMemo(() => {
    const { newPassword } = formValues;
    const checks = PASSWORD_REQUIREMENTS.map((item) => item.test(newPassword));
    const passed = checks.filter(Boolean).length;

    if (!newPassword) {
      return { label: 'Enter a new password', color: 'text-dark-text-muted', bar: 0, checks };
    }

    if (passed <= 1) {
      return { label: 'Very Weak', color: 'text-red-400', bar: 25, checks };
    }
    if (passed === 2) {
      return { label: 'Weak', color: 'text-orange-400', bar: 50, checks };
    }
    if (passed === 3) {
      return { label: 'Good', color: 'text-blue-400', bar: 75, checks };
    }
    return { label: 'Strong', color: 'text-green-400', bar: 100, checks };
  }, [formValues]);

  const handleChange = (field) => (event) => {
    setFormValues((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const validate = () => {
    const nextErrors = {};
    const trimmedOld = formValues.oldPassword.trim();
    const trimmedNew = formValues.newPassword.trim();
    const trimmedConfirm = formValues.confirmPassword.trim();

    if (!trimmedOld) {
      nextErrors.oldPassword = 'Old password is required';
    }
    if (!trimmedNew) {
      nextErrors.newPassword = 'New password is required';
    }
    if (!trimmedConfirm) {
      nextErrors.confirmPassword = 'Please confirm your new password';
    }

    if (trimmedNew && trimmedOld && trimmedNew === trimmedOld) {
      nextErrors.newPassword = 'New password must be different from old password';
    }

    const requirementFailed = PASSWORD_REQUIREMENTS.some((rule) => !rule.test(trimmedNew));
    if (trimmedNew && requirementFailed) {
      nextErrors.newPassword = 'Password does not meet the minimum requirements';
    }

    if (trimmedNew && trimmedConfirm && trimmedNew !== trimmedConfirm) {
      nextErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) {
      return;
    }

    setSubmitting(true);
    try {
      const result = await changePassword({
        oldPassword: formValues.oldPassword.trim(),
        newPassword: formValues.newPassword.trim(),
      });

      if (result?.success) {
        toast.success(result.message || 'Password updated successfully.');
        navigate('/admin/profile', { replace: true });
      }
    } finally {
      setSubmitting(false);
    }
  };

  const openForgotPasswordPanel = () => {
    setForgotEmail('');
    setForgotSubmitted(false);
    setShowForgotPanel(true);
  };

  const closeForgotPasswordPanel = () => {
    if (!forgotLoading) {
      setShowForgotPanel(false);
    }
  };

  const handleForgotPassword = async (event) => {
    event.preventDefault();
    const normalizedEmail = (forgotEmail || '').trim().toLowerCase();

    if (!normalizedEmail) {
      toast.error('Please enter your email address');
      return;
    }

    setForgotLoading(true);
    try {
      const response = await authAPI.forgotPassword(normalizedEmail);
      const message = response.message || 'If the email exists, a reset link has been sent.';
      toast.success(message);
      setForgotSubmitted(true);
    } catch (error) {
      const message = error.response?.data?.message || 'Unable to process password reset request';
      toast.error(message);
    } finally {
      setForgotLoading(false);
    }
  };

  return (
    <AdminShell
      title="Change Password"
      description="Keep your account secure by updating your credentials regularly."
    >
      <motion.div
        className="bg-[#0E0F14] rounded-2xl border border-white/5 shadow-lg p-6 md:p-8"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-start gap-4 mb-8">
          <div className="w-12 h-12 rounded-full bg-primary-500/20 border border-primary-500/40 flex items-center justify-center">
            <ShieldCheck className="w-6 h-6 text-primary-300" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">Password requirements</h2>
            <p className="text-sm text-dark-text-muted mt-1">
              Password must be at least 6 characters and include letters, numbers, and special symbols.
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <InputField
            label="Old Password"
            type="password"
            placeholder="Enter old password"
            icon={Lock}
            showPasswordToggle
            value={formValues.oldPassword}
            onChange={handleChange('oldPassword')}
            error={errors.oldPassword}
          />

          <div className="text-right -mt-2">
            <button
              type="button"
              className="text-primary-400 hover:text-primary-300 text-sm transition-colors"
              onClick={openForgotPasswordPanel}
            >
              Forgot password?
            </button>
          </div>

          <div className="space-y-4">
            <InputField
              label="New Password"
              type="password"
              placeholder="Enter new password"
              icon={Lock}
              showPasswordToggle
              value={formValues.newPassword}
              onChange={handleChange('newPassword')}
              error={errors.newPassword}
            />

            <div className="bg-[#10131A] border border-white/5 rounded-xl p-4">
              <div className="flex items-center justify-between text-xs text-dark-text-muted mb-2">
                <span>Password strength</span>
                <span className={`font-medium ${passwordScore.color}`}>{passwordScore.label}</span>
              </div>
              <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full bg-primary-400 transition-all duration-300"
                  style={{ width: `${passwordScore.bar}%` }}
                />
              </div>
              <ul className="mt-3 space-y-1 text-[11px]">
                {PASSWORD_REQUIREMENTS.map((rule, index) => (
                  <li
                    key={rule.label}
                    className={passwordScore.checks[index] ? 'text-green-400' : 'text-dark-text-muted'}
                  >
                    • {rule.label}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <InputField
            label="Confirm New Password"
            type="password"
            placeholder="Re-enter new password"
            icon={Lock}
            showPasswordToggle
            value={formValues.confirmPassword}
            onChange={handleChange('confirmPassword')}
            error={errors.confirmPassword}
          />

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <Button
              type="button"
              variant="secondary"
              size="medium"
              onClick={() => navigate(-1)}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="medium"
              loading={submitting}
            >
              Submit
            </Button>
          </div>
        </form>

        {showForgotPanel && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className="mt-8 rounded-3xl border border-white/10 bg-[#10131A] p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-dark-text-primary">Need a reset link?</h3>
                <p className="text-sm text-dark-text-muted mt-1">
                  Enter your registered email and we&apos;ll send you a password reset link.
                </p>
              </div>
              <button
                type="button"
                onClick={closeForgotPasswordPanel}
                className="rounded-full p-2 text-dark-text-muted hover:text-white hover:bg-white/10 transition-colors"
                aria-label="Close forgot password panel"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleForgotPassword} className="mt-5 space-y-5">
              <InputField
                label="Email"
                type="email"
                placeholder="admin@example.com"
                icon={Mail}
                value={forgotEmail}
                onChange={(event) => setForgotEmail(event.target.value)}
                required
              />

              <Button
                type="submit"
                variant="primary"
                fullWidth
                icon={Send}
                loading={forgotLoading}
              >
                {forgotLoading ? 'Sending reset link…' : 'Send reset link'}
              </Button>

              {forgotSubmitted && (
                <div className="rounded-2xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-100">
                  Check your inbox for the reset link. Don&apos;t forget to look in your spam folder.
                </div>
              )}
            </form>
          </motion.div>
        )}
      </motion.div>
    </AdminShell>
  );
};

export default AdminChangePassword;
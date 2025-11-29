import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Lock, Mail } from 'lucide-react';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { useAuth } from '../../contexts/AuthContext.jsx';
import toast from 'react-hot-toast';

const AdminLogin = () => {
  const navigate = useNavigate();
  const { login, loading, isAuthenticated, admin } = useNewAdminAuth();
  const { bootstrapSession } = useAuth();
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  if (!loading && isAuthenticated) {
    const nextRoute = admin?.contact_exists ? '/admin/dashboard' : '/admin/onboarding';
    return <Navigate to={nextRoute} replace />;
  }

  const onSubmit = async (formValues) => {
    setSubmitting(true);
    const result = await login(formValues.email, formValues.password);
    setSubmitting(false);
    if (!result.success) {
      return;
    }

    if (result.userType === 'member') {
      const member = result.member || {};

      if (member) {
        const memberUser = {
          role: 'member',
          id: member.member_id ?? member.admin_id,
          member_id: member.member_id ?? null,
          admin_id: member.admin_id ?? null,
          work_type: member.work_type ?? null,
          email: member.email ?? formValues.email,
          has_inai_credentials: false,
        };

        bootstrapSession({
          token: result.accessToken,
          user: memberUser,
        });
      }

      const workType = member.work_type ?? '';
      const normalizedWorkType = workType.replace('_management', '');
      const routeMap = {
        chapter: '/chapter-dashboard',
        chapter_management: '/chapter-dashboard',
        student: '/student-dashboard',
        student_management: '/student-dashboard',
        lecture: '/lecture-dashboard',
        lecture_management: '/lecture-dashboard',
      };
      const destination = routeMap[normalizedWorkType] || routeMap[workType] || '/chapter-dashboard';

      toast.success('Welcome! Redirecting to your dashboard.');
      navigate(destination, { replace: true });
      return;
    }

    if (result.admin?.contact_exists) {
      navigate('/admin/dashboard', { replace: true });
    } else {
      navigate('/admin/onboarding', { replace: true });
    }
  };

  return (
    <div className="min-h-screen bg-[#050507] flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md card p-8 border border-white/10 bg-white/[0.04] backdrop-blur-xl"
      >
        <div className="text-center mb-8">
          <div className="text-4xl font-bold text-gradient mb-2">INAI Admin</div>
          <p className="text-sm text-dark-text-secondary">
            Sign in to access your administration portal
          </p>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <InputField
            label="Email"
            type="email"
            placeholder="admin@example.com"
            icon={Mail}
            error={errors.email?.message}
            {...register('email', {
              required: 'Email is required',
              pattern: {
                value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                message: 'Please enter a valid email',
              },
            })}
          />

          <InputField
            label="Password"
            type="password"
            placeholder="••••••••"
            icon={Lock}
            showPasswordToggle
            error={errors.password?.message}
            {...register('password', {
              required: 'Password is required',
              minLength: {
                value: 8,
                message: 'Password must be at least 8 characters',
              },
            })}
          />

          <Button
            type="submit"
            variant="primary"
            size="large"
            fullWidth
            loading={submitting}
            disabled={loading}
          >
            {submitting ? 'Signing in...' : 'Sign In'}
          </Button>
        </form>

        <div className="mt-6 text-center space-y-3">
          <button
            type="button"
            className="text-sm text-dark-text-muted hover:text-dark-text-primary"
            onClick={() => navigate('/')}
          >
            ← Back to role selection
          </button>

          <div className="text-sm text-dark-text-secondary">
            Don't have an account?{' '}
            <button
              type="button"
              className="text-primary-400 hover:text-primary-300 font-medium"
              onClick={() => navigate('/admin/register')}
            >
              Create one now
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AdminLogin;

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, Mail, Lock, User } from 'lucide-react';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';

const PACKAGE_PLAN_OPTIONS = [
  { label: '20k', value: '20k' },
  { label: '50k', value: '50k' },
  { label: '100k', value: '100k' },
];

const VALIDITY_OPTIONS = [
  { label: '1 Year', value: '1_year' },
];

const AdminRegister = () => {
  const navigate = useNavigate();
  const { register: registerAdmin, loading, isAuthenticated } = useNewAdminAuth();
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    defaultValues: {
      fullName: '',
      email: '',
      password: '',
      packagePlan: PACKAGE_PLAN_OPTIONS[0].value,
      validity: VALIDITY_OPTIONS[0].value,
    },
  });

  if (!loading && isAuthenticated) {
    return <Navigate to="/admin/members" replace />;
  }

  const onSubmit = async (values) => {
    setSubmitting(true);

    const payload = {
      full_name: values.fullName.trim(),
      email: values.email.trim(),
      password: values.password,
      package_plan: values.packagePlan,
      validity: values.validity,
    };

    const result = await registerAdmin(payload);
    setSubmitting(false);

    if (result.success) {
      navigate('/admin/login', { replace: true });
    }
  };

  return (
    <div className="min-h-screen bg-[#050507] flex items-center justify-center px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-2xl card p-8 border border-white/10 bg-white/[0.04] backdrop-blur-xl"
      >
        <div className="text-center mb-8 space-y-3">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-500/20 text-primary-400">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <div>
            <div className="text-4xl font-bold text-gradient mb-2">Create Admin Account</div>
            <p className="text-sm text-dark-text-secondary">
              Register to access the INAI Administration Portal
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <InputField
            label="Full Name"
            placeholder="Priya Sharma"
            icon={User}
            error={errors.fullName?.message}
            {...register('fullName', {
              required: 'Full name is required',
              minLength: {
                value: 2,
                message: 'Full name must be at least 2 characters',
              },
            })}
          />

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
            placeholder="Create a strong password"
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

          <div>
            <label className="block text-sm font-medium text-dark-text-primary mb-2">
              Package Plan
            </label>
            <select
              className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
              {...register('packagePlan', {
                required: 'Package plan is required',
              })}
            >
              {PACKAGE_PLAN_OPTIONS.map((option) => (
                <option key={option.value} value={option.value} className="bg-[#050507]">
                  {option.label}
                </option>
              ))}
            </select>
            {errors.packagePlan && (
              <p className="mt-1 text-sm text-red-500">{errors.packagePlan.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-dark-text-primary mb-2">
              Validity
            </label>
            <select
              className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
              {...register('validity', {
                required: 'Validity is required',
              })}
            >
              {VALIDITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value} className="bg-[#050507]">
                  {option.label}
                </option>
              ))}
            </select>
            {errors.validity && (
              <p className="mt-1 text-sm text-red-500">{errors.validity.message}</p>
            )}
          </div>

          <div className="md:col-span-2">
            <Button
              type="submit"
              variant="primary"
              size="large"
              fullWidth
              loading={submitting}
              disabled={loading}
            >
              {submitting ? 'Creating account...' : 'Create Account'}
            </Button>
          </div>
        </form>

        <div className="mt-8 text-center space-y-3">
          <div className="text-sm text-dark-text-secondary">
            Already have an account?{' '}
            <button
              type="button"
              className="text-primary-400 hover:text-primary-300 font-medium"
              onClick={() => navigate('/admin/login')}
            >
              Sign in instead
            </button>
          </div>

          <button
            type="button"
            className="text-sm text-dark-text-muted hover:text-dark-text-primary"
            onClick={() => navigate('/')}
          >
            ← Back to role selection
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default AdminRegister;
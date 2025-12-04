/**
 * Login Page Component
 * Unified login for both admins and members with INAI design
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import InputField from '../../components/ui/InputField';
import Button from '../../components/ui/Button';
import { Mail, Lock, LogIn } from 'lucide-react';
import toast from 'react-hot-toast';

const LoginPage = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
  } = useForm({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const emailValue = watch('email');
  const passwordValue = watch('password');

  const onSubmit = async (data) => {
    setIsLoading(true);
    
    try {
      const result = await login(data.email, data.password);
      
      if (result.success) {
        const user = result.data;
        
        // Redirect based on user type and onboarding status
        if (user.role === 'admin') {
          if (!user.has_inai_credentials) {
            navigate('/complete-onboarding');
          } else {
            navigate('/dev_admin');
          }
        } else if (user.role === 'member') {
          navigate(`/${user.work_type}-dashboard`);
        }
      }
    } catch (error) {
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* INAI Logo and Branding */}
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-6xl font-bold text-gradient mb-2">
            INAI
          </div>
          <div className="text-sm text-dark-text-secondary tracking-widest uppercase mb-4">
            Education System
          </div>
          <h1 className="text-2xl font-bold text-dark-text-primary mb-2">
            Sign in your account
          </h1>
          <p className="text-dark-text-secondary text-sm">
            Access your account to explore INAI's features
          </p>
        </motion.div>

        {/* Login Form */}
        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Email Field */}
            <InputField
              label="Email"
              type="email"
              placeholder="8747-0404-0387-0420"
              icon={Mail}
              error={errors.email?.message}
              {...register('email', {
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Invalid email address',
                },
              })}
              value={emailValue}
              onChange={(e) => setValue('email', e.target.value)}
            />

            {/* Password Field */}
            <InputField
              label="Password"
              type="password"
              placeholder="8747-0404-0387-0420"
              icon={Lock}
              showPasswordToggle
              error={errors.password?.message}
              {...register('password', {
                required: 'Password is required',
                minLength: {
                  value: 6,
                  message: 'Password must be at least 6 characters',
                },
              })}
              value={passwordValue}
              onChange={(e) => setValue('password', e.target.value)}
            />

            {/* Forgot Password Link */}
            <div className="text-right">
              <button
                type="button"
                className="text-primary-400 hover:text-primary-300 text-sm transition-colors"
                onClick={() => toast('Contact your administrator for password reset', { icon: 'ℹ️' })}
              >
                Forgot password?
              </button>
            </div>

            {/* Login Button */}
            <Button
              type="submit"
              variant="gradient"
              size="large"
              fullWidth
              loading={isLoading}
              icon={LogIn}
              className="mt-8"
            >
              {isLoading ? 'Signing in...' : 'Done'}
            </Button>
          </form>

          {/* Additional Info */}
          <div className="mt-6 text-center">
            <p className="text-xs text-dark-text-muted">
              By signing in, you agree to INAI's Terms of Service and Privacy Policy
            </p>
          </div>
        </motion.div>

        {/* Demo Credentials */}
        <motion.div
          className="mt-6 p-4 bg-dark-card/50 rounded-xl border border-dark-border"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <h3 className="text-sm font-medium text-dark-text-primary mb-2">
            Demo Credentials
          </h3>
          <div className="text-xs text-dark-text-secondary space-y-1">
            <div>
              <strong>Super Admin:</strong> admin@inai.edu / superadmin123
            </div>
            <div>
              <strong>Note:</strong> Create additional users via super admin dashboard
            </div>
          </div>
        </motion.div>

        {/* Role Selection Cards */}
        <motion.div
          className="mt-8 grid grid-cols-2 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {/* Admin Card */}
          <div className="card-interactive p-4 text-center">
            <div className="w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center mx-auto mb-3">
              <div className="text-white text-sm font-bold">👨‍💼</div>
            </div>
            <h3 className="font-medium text-dark-text-primary text-sm mb-1">
              Admin
            </h3>
            <p className="text-xs text-dark-text-secondary">
              Manage your institution and team members
            </p>
          </div>

          {/* Student Card */}
          <button
            type="button"
            onClick={() => navigate('/school-portal')}
            className="card-interactive p-4 text-center w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
          >
            <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-3">
              <div className="text-white text-sm font-bold">👨‍🎓</div>
            </div>
            <h3 className="font-medium text-dark-text-primary text-sm mb-1">
              Student
            </h3>
            <p className="text-xs text-dark-text-secondary">
              Access your courses and learning materials
            </p>
          </button>
        </motion.div>
      </div>
    </div>
  );
};

export default LoginPage;

/**
 * Change Password Page - First step of admin onboarding
 * Mandatory password change for new admin accounts
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import InputField from '../../components/ui/InputField';
import Button from '../../components/ui/Button';
import { Lock, ArrowRight, Shield } from 'lucide-react';
import toast from 'react-hot-toast';

const ChangePasswordPage = () => {
  const { changePassword, user } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm({
    defaultValues: {
      oldPassword: '',
      newPassword: '',
      confirmPassword: '',
    },
  });

  const newPassword = watch('newPassword');
  const oldPassword = watch('oldPassword');
  const confirmPassword = watch('confirmPassword');

  const onSubmit = async (data) => {
    if (data.newPassword !== data.confirmPassword) {
      toast.error('New passwords do not match');
      return;
    }

    setIsLoading(true);
    
    try {
      const result = await changePassword(data.oldPassword, data.newPassword);
      
      if (result.success) {
        toast.success('Password changed successfully!');
        // Navigate to next onboarding step
        navigate('/contact-person');
      }
    } catch (error) {
      console.error('Password change error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getPasswordStrength = (password) => {
    if (!password) return { score: 0, label: '', color: '' };
    
    let score = 0;
    if (password.length >= 8) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;

    const strength = {
      0: { label: 'Very Weak', color: 'bg-red-500' },
      1: { label: 'Weak', color: 'bg-red-400' },
      2: { label: 'Fair', color: 'bg-yellow-500' },
      3: { label: 'Good', color: 'bg-blue-500' },
      4: { label: 'Strong', color: 'bg-green-500' },
      5: { label: 'Very Strong', color: 'bg-green-600' },
    };

    return { score, ...strength[score] };
  };

  const passwordStrength = getPasswordStrength(newPassword);

  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4">
      <div className="w-full max-w-lg">
        {/* Header */}
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Shield className="w-8 h-8 text-white" />
          </div>
          
          <h1 className="text-2xl font-bold text-dark-text-primary mb-2">
            Change your Password
          </h1>
          <p className="text-dark-text-secondary">
            Welcome to INAI! For security, please change your initial password.
          </p>
        </motion.div>

        {/* Progress Indicator */}
        <motion.div
          className="mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex items-center justify-between text-sm text-dark-text-muted mb-2">
            <span>Onboarding Progress</span>
            <span>1 of 4</span>
          </div>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: '25%' }} />
          </div>
        </motion.div>

        {/* Form */}
        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Current Password */}
            <InputField
              label="Enter Old Password"
              type="password"
              placeholder="Enter Current Password"
              icon={Lock}
              showPasswordToggle
              error={errors.oldPassword?.message}
              {...register('oldPassword', {
                required: 'Current password is required',
              })}
              value={oldPassword}
              onChange={(e) => setValue('oldPassword', e.target.value)}
            />

            {/* New Password */}
            <div className="space-y-2">
              <InputField
                label="Current Password"
                type="password"
                placeholder="Enter Current Password"
                icon={Lock}
                showPasswordToggle
                error={errors.newPassword?.message}
                {...register('newPassword', {
                  required: 'New password is required',
                  minLength: {
                    value: 8,
                    message: 'Password must be at least 8 characters',
                  },
                  validate: (value) => {
                    if (value === oldPassword) {
                      return 'New password must be different from current password';
                    }
                    return true;
                  },
                })}
                value={newPassword}
                onChange={(e) => setValue('newPassword', e.target.value)}
              />
              
              {/* Password Strength Indicator */}
              {newPassword && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-dark-text-muted">Password Strength</span>
                    <span className={`font-medium ${
                      passwordStrength.score >= 3 ? 'text-green-400' : 
                      passwordStrength.score >= 2 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {passwordStrength.label}
                    </span>
                  </div>
                  <div className="flex space-x-1">
                    {[1, 2, 3, 4, 5].map((level) => (
                      <div
                        key={level}
                        className={`h-1 flex-1 rounded-full ${
                          level <= passwordStrength.score
                            ? passwordStrength.color
                            : 'bg-gray-700'
                        }`}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </div>

            {/* Confirm Password */}
            <InputField
              label="Re-Current Password"
              type="password"
              placeholder="Enter Re-Current Password"
              icon={Lock}
              showPasswordToggle
              error={errors.confirmPassword?.message}
              {...register('confirmPassword', {
                required: 'Please confirm your new password',
                validate: (value) => {
                  if (value !== newPassword) {
                    return 'Passwords do not match';
                  }
                  return true;
                },
              })}
              value={confirmPassword}
              onChange={(e) => setValue('confirmPassword', e.target.value)}
            />

            {/* Password Requirements */}
            <div className="bg-dark-bg/50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-dark-text-primary mb-2">
                Password Requirements:
              </h4>
              <ul className="text-xs text-dark-text-muted space-y-1">
                <li className={newPassword?.length >= 8 ? 'text-green-400' : ''}>
                  • At least 8 characters long
                </li>
                <li className={/[a-z]/.test(newPassword) ? 'text-green-400' : ''}>
                  • Contains lowercase letters
                </li>
                <li className={/[A-Z]/.test(newPassword) ? 'text-green-400' : ''}>
                  • Contains uppercase letters
                </li>
                <li className={/[0-9]/.test(newPassword) ? 'text-green-400' : ''}>
                  • Contains numbers
                </li>
                <li className={/[^A-Za-z0-9]/.test(newPassword) ? 'text-green-400' : ''}>
                  • Contains special characters
                </li>
              </ul>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              variant="primary"
              size="large"
              fullWidth
              loading={isLoading}
              icon={ArrowRight}
              iconPosition="right"
              disabled={passwordStrength.score < 3}
            >
              {isLoading ? 'Updating...' : 'Submit'}
            </Button>
          </form>
        </motion.div>

        {/* Footer */}
        <motion.div
          className="text-center mt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <p className="text-xs text-dark-text-muted">
            Having trouble? Contact your system administrator for assistance.
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default ChangePasswordPage;
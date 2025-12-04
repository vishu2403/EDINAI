/**
 * Add Administrator Page - For Super Admins only
 * Create new administrator accounts
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { adminAPI } from '../../utils/api';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { 
  Crown, 
  ArrowLeft, 
  Shield, 
  Package,
  Calendar,
  Mail,
  User,
  Save,
  Lock
} from 'lucide-react';
import toast from 'react-hot-toast';

const AddAdmin = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch
  } = useForm();

  React.useEffect(() => {
    // Check if user is super admin
    if (!user?.is_super_admin) {
      navigate('/dev_admin');
      return;
    }
  }, [user, navigate]);

  const onSubmit = async (data) => {
    setLoading(true);
    try {
      const response = await adminAPI.createAdmin({
        name: data.name,
        email: data.email,
        package: data.package,
        password: data.password,
        expiry_months: parseInt(data.expiry_months, 10)
      });

      if (response.status) {
        toast.success('Administrator created successfully!');
        navigate('/admin-management');
      }
    } catch (error) {
      console.error('Create admin error:', error);
      toast.error(error.response?.data?.detail || 'Failed to create administrator');
    } finally {
      setLoading(false);
    }
  };

  const packages = [
    { id: 'p1', name: 'Basic', description: '5 members, 10MB files, 2 files/upload' },
    { id: 'p2', name: 'Standard', description: '15 members, 50MB files, 5 files/upload, 30min lectures' },
    { id: 'p3', name: 'Premium', description: '50 members, 100MB files, 10 files/upload, 60min lectures' }
  ];

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <div className="bg-dark-card border-b border-dark-border px-6 py-4">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="medium"
            icon={ArrowLeft}
            onClick={() => navigate('/admin-management')}
          >
            Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-dark-text-primary flex items-center gap-2">
              <Shield className="w-6 h-6 text-purple-400" />
              Add Administrator
            </h1>
            <p className="text-sm text-dark-text-muted">
              Create a new administrator account
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <div className="max-w-2xl mx-auto">
          <motion.div
            className="card p-8"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="mb-8">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <Crown className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-center text-dark-text-primary mb-2">
                Create Administrator
              </h2>
              <p className="text-center text-dark-text-muted">
                Set up a new administrator account with package and expiry settings
              </p>
            </div>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* Name */}
              <InputField
                label="Full Name"
                type="text"
                placeholder="Enter administrator's full name"
                icon={User}
                {...register('name', { 
                  required: 'Full name is required',
                  minLength: { value: 2, message: 'Name must be at least 2 characters' }
                })}
                error={errors.name?.message}
              />

              {/* Email */}
              <InputField
                label="Email Address"
                type="email"
                placeholder="Enter administrator's email"
                icon={Mail}
                {...register('email', { 
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address'
                  }
                })}
                error={errors.email?.message}
              />

              {/* Password */}
              <InputField
                label="Password"
                type="password"
                placeholder="Enter a secure password"
                icon={Lock}
                showPasswordToggle
                {...register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 8,
                    message: 'Password must be at least 8 characters'
                  }
                })}
                error={errors.password?.message}
              />

              {/* Package Selection */}
              <div>
                <label className="block text-sm font-medium text-dark-text-primary mb-3">
                  <Package className="inline w-4 h-4 mr-2" />
                  Package Plan
                </label>
                <div className="grid grid-cols-1 gap-3">
                  {packages.map((pkg) => (
                    <label key={pkg.id} className="cursor-pointer">
                      <input
                        type="radio"
                        value={pkg.id}
                        {...register('package', { required: 'Package selection is required' })}
                        className="sr-only"
                      />
                      <div className={`p-4 border rounded-lg transition-all ${
                        watch('package') === pkg.id
                          ? 'border-primary-500 bg-primary-500/10'
                          : 'border-dark-border hover:border-dark-border-hover'
                      }`}>
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="font-medium text-dark-text-primary">{pkg.name}</h3>
                            <p className="text-sm text-dark-text-muted">{pkg.description}</p>
                          </div>
                          <div className={`w-4 h-4 border-2 rounded-full ${
                            watch('package') === pkg.id
                              ? 'border-primary-500 bg-primary-500'
                              : 'border-dark-border'
                          }`}>
                            {watch('package') === pkg.id && (
                              <div className="w-full h-full rounded-full bg-white transform scale-50"></div>
                            )}
                          </div>
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
                {errors.package && (
                  <p className="mt-2 text-sm text-red-400">{errors.package.message}</p>
                )}
              </div>

              {/* Expiry Period */}
              <div>
                <label className="block text-sm font-medium text-dark-text-primary mb-3">
                  <Calendar className="inline w-4 h-4 mr-2" />
                  Account Validity (Months)
                </label>
                <select
                  {...register('expiry_months', { required: 'Expiry period is required' })}
                  className="w-full px-4 py-3 bg-dark-card border border-dark-border rounded-lg text-dark-text-primary focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all"
                >
                  <option value="">Select validity period</option>
                  <option value="3">3 Months</option>
                  <option value="6">6 Months</option>
                  <option value="12">12 Months (1 Year)</option>
                  <option value="24">24 Months (2 Years)</option>
                  <option value="36">36 Months (3 Years)</option>
                </select>
                {errors.expiry_months && (
                  <p className="mt-2 text-sm text-red-400">{errors.expiry_months.message}</p>
                )}
              </div>

              {/* Submit Button */}
              <div className="pt-6">
                <Button
                  type="submit"
                  variant="primary"
                  size="large"
                  fullWidth
                  icon={Save}
                  loading={loading}
                >
                  Create Administrator
                </Button>
              </div>
            </form>

            {/* Info Box */}
            <div className="mt-8 p-4 bg-blue-600/10 border border-blue-600/20 rounded-lg">
              <h4 className="font-medium text-blue-400 mb-2">Important Information</h4>
              <ul className="text-sm text-dark-text-secondary space-y-1">
                <li>• Share the password securely with the new administrator</li>
                <li>• The new administrator must change their password on first login</li>
                <li>• They will need to complete the onboarding process</li>
                <li>• Account expires based on the selected validity period</li>
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default AddAdmin;

/**
 * Add Member Page - Admin can add new team members
 * Includes form validation and role assignment
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { memberAPI } from '../../utils/api';
import InputField from '../../components/ui/InputField';
import Button from '../../components/ui/Button';
import { User, Mail, Phone, Briefcase, ArrowLeft, UserPlus, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const AddMemberPage = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [addedMember, setAddedMember] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset,
  } = useForm({
    defaultValues: {
      name: '',
      email: '',
      password: '',
      designation: '',
      workType: '',
      phoneNumber: '',
    },
  });

  const formValues = watch();

  const onSubmit = async (data) => {
    setIsLoading(true);
    
    try {
      const memberData = {
        name: data.name,
        email: data.email,
        password: data.password,
        designation: data.designation,
        work_type: data.workType,
        phone_number: data.phoneNumber || null,
      };

      const response = await memberAPI.createMember(memberData);
      
      if (response.status) {
        setAddedMember(response.data);
        toast.success('Member added successfully!');
        reset();
      }
    } catch (error) {
      console.error('Add member error:', error);
      toast.error('Failed to add member');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <div className="bg-dark-card border-b border-dark-border px-6 py-4">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="medium"
            icon={ArrowLeft}
            onClick={() => navigate('/dev_admin')}
          />
          <div>
            <h1 className="text-xl font-bold text-dark-text-primary">Add Member</h1>
            <p className="text-sm text-dark-text-muted">Add a new team member to your organization</p>
          </div>
        </div>
      </div>

      <div className="max-w-2xl mx-auto p-6">
        {/* Success Message */}
        {addedMember && (
          <motion.div
            className="mb-8 card p-6 border-green-500/20 bg-green-500/5"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-green-500 mt-1" />
              <div>
                <h3 className="font-medium text-green-400 mb-1">Member Added Successfully!</h3>
                <div className="text-sm text-dark-text-secondary space-y-1">
                  <p><strong>Name:</strong> {addedMember.name}</p>
                  <p><strong>Email:</strong> {addedMember.email}</p>
                  <p><strong>Role ID:</strong> {addedMember.role_id}</p>
                  <p><strong>Work Type:</strong> {addedMember.work_type}</p>
                </div>
                <div className="mt-3 flex space-x-2">
                  <Button
                    size="small"
                    variant="secondary"
                    onClick={() => setAddedMember(null)}
                  >
                    Add Another
                  </Button>
                  <Button
                    size="small"
                    variant="primary"
                    onClick={() => navigate('/all-members')}
                  >
                    View All Members
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Form */}
        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center">
              <UserPlus className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-dark-text-primary">Member Information</h2>
              <p className="text-sm text-dark-text-muted">Fill in the details for the new team member</p>
            </div>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Name */}
            <InputField
              label="Name Of Person"
              placeholder="Enter name"
              icon={User}
              error={errors.name?.message}
              {...register('name', {
                required: 'Name is required',
                minLength: {
                  value: 2,
                  message: 'Name must be at least 2 characters',
                },
              })}
              value={formValues.name}
              onChange={(e) => setValue('name', e.target.value)}
            />

            {/* Designation */}
            <InputField
              label="Designation"
              placeholder="Your Designation"
              icon={Briefcase}
              error={errors.designation?.message}
              {...register('designation', {
                required: 'Designation is required',
              })}
              value={formValues.designation}
              onChange={(e) => setValue('designation', e.target.value)}
            />

            {/* Work Type */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-dark-text-primary">
                Work Type <span className="text-red-500">*</span>
              </label>
              <select
                className="input-primary"
                {...register('workType', {
                  required: 'Work type is required',
                })}
                value={formValues.workType}
                onChange={(e) => setValue('workType', e.target.value)}
              >
                <option value="">Select Your Work Type</option>
                <option value="chapter">Chapter Management</option>
                <option value="student">Student Management</option>
                <option value="lecture">Lecture Management</option>
              </select>
              {errors.workType && (
                <p className="text-sm text-red-500">{errors.workType.message}</p>
              )}
            </div>

            {/* Phone Number */}
            <InputField
              label="Phone Number"
              type="tel"
              placeholder="Enter Your Number"
              icon={Phone}
              error={errors.phoneNumber?.message}
              {...register('phoneNumber', {
                pattern: {
                  value: /^[+]?[\d\s\-\(\)]{10,}$/,
                  message: 'Please enter a valid phone number',
                },
              })}
              value={formValues.phoneNumber}
              onChange={(e) => setValue('phoneNumber', e.target.value)}
            />

            {/* Email */}
            <InputField
              label="Email"
              type="email"
              placeholder="Enter Your Email"
              icon={Mail}
              error={errors.email?.message}
              {...register('email', {
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Please enter a valid email address',
                },
              })}
              value={formValues.email}
              onChange={(e) => setValue('email', e.target.value)}
            />

            {/* Password */}
            <InputField
              label="Password"
              type="password"
              placeholder="••••••••"
              showPasswordToggle
              error={errors.password?.message}
              {...register('password', {
                required: 'Password is required',
                minLength: {
                  value: 6,
                  message: 'Password must be at least 6 characters',
                },
              })}
              value={formValues.password}
              onChange={(e) => setValue('password', e.target.value)}
              helperText="Default password that the member can change after first login"
            />

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <Button
                type="button"
                variant="secondary"
                size="large"
                icon={ArrowLeft}
                onClick={() => navigate('/dev_admin')}
                className="sm:w-auto"
              >
                Cancel
              </Button>
              
              <Button
                type="submit"
                variant="primary"
                size="large"
                fullWidth
                loading={isLoading}
                icon={UserPlus}
                iconPosition="right"
                className="sm:flex-1"
              >
                {isLoading ? 'Adding Member...' : 'Next'}
              </Button>
            </div>
          </form>
        </motion.div>

        {/* Info Box */}
        <motion.div
          className="mt-6 bg-primary-500/10 border border-primary-500/20 rounded-lg p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-start space-x-3">
            <div className="w-5 h-5 bg-primary-500 rounded-full flex items-center justify-center mt-0.5">
              <span className="text-white text-xs font-bold">i</span>
            </div>
            <div className="text-sm">
              <p className="text-dark-text-primary font-medium mb-1">
                Member Role Assignment
              </p>
              <p className="text-dark-text-secondary">
                Each member will receive a unique role ID (e.g., CM001, SM001, LM001) based on their work type. 
                They can use their email and the provided password to log in to their respective dashboard.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AddMemberPage;

import React, { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { motion } from 'framer-motion';
import { CheckCircle, Trash2, Users, UserPlus, ArrowLeft } from 'lucide-react';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminMemberAPI } from '../../utils/newAdminApi.js';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

const WORK_TYPE_OPTIONS = [
  { value: 'chapter_management', label: 'Chapter Management' },
  { value: 'student_management', label: 'Student Management' },
  { value: 'lecture_management', label: 'Lecture Management' },
];

const AdminMembersPage = () => {
  const navigate = useNavigate();
  const { admin, logout } = useNewAdminAuth();
  const adminId = admin?.admin_id;

  const [members, setMembers] = useState([]);
  const [membersLoading, setMembersLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [filter, setFilter] = useState('all');
  const [recentMember, setRecentMember] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
  } = useForm({
    defaultValues: {
      name: '',
      designation: '',
      workType: '',
      phoneNumber: '',
      email: '',
      password: '',
    },
  });

  const watchValues = watch();
  const workTypeField = register('workType', {
    required: 'Work type is required',
  });

  const filteredMembers = useMemo(() => {
    if (filter === 'all') {
      return members;
    }
    return members.filter((member) => member.work_type === filter);
  }, [members, filter]);

  useEffect(() => {
    if (adminId) {
      loadMembers(filter);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminId, filter]);

  const loadMembers = async (workType) => {
    if (!adminId) {
      return;
    }
    try {
      setMembersLoading(true);
      const response = await newAdminMemberAPI.listMembers({
        adminId,
        workType,
      });

      if (response?.members) {
        setMembers(response.members || []);
      } else if (Array.isArray(response)) {
        setMembers(response);
      } else {
        setMembers([]);
      }
    } catch (error) {
      console.error('Load members error:', error);
      toast.error(error.response?.data?.detail || 'Failed to load members');
      setMembers([]);
    } finally {
      setMembersLoading(false);
    }
  };

  const onSubmit = async (data) => {
    if (!adminId) {
      toast.error('Admin information missing. Please log in again.');
      return;
    }

    setCreating(true);
    try {
      const payload = {
        name: data.name.trim(),
        designation: data.designation.trim(),
        work_type: data.workType,
        phone_number: data.phoneNumber.trim(),
        email: data.email.trim(),
        password: data.password,
      };

      const response = await newAdminMemberAPI.createMember({
        adminId,
        data: payload,
      });

      toast.success('Member added successfully');
      setRecentMember(response);
      reset();
      await loadMembers(filter);
    } catch (error) {
      console.error('Create member error:', error);
      const message = error.response?.data?.detail || 'Failed to add member';
      toast.error(message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (memberId) => {
    if (!adminId) {
      toast.error('Admin information missing. Please log in again.');
      return;
    }

    const confirmed = window.confirm('Are you sure you want to delete this member?');
    if (!confirmed) return;

    try {
      await newAdminMemberAPI.deleteMember({ adminId, memberId });
      toast.success('Member removed');
      await loadMembers(filter);
    } catch (error) {
      console.error('Delete member error:', error);
      const message = error.response?.data?.detail || 'Failed to delete member';
      toast.error(message);
    }
  };

  if (!adminId) {
    return (
      <div className="min-h-screen bg-dark-bg flex flex-col items-center justify-center text-dark-text-secondary space-y-4">
        <p>Admin details not available. Please log in again.</p>
        <Button variant="primary" onClick={logout}>
          Log Out
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      <div className="bg-dark-card border-b border-dark-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="medium"
              icon={ArrowLeft}
              onClick={() => navigate('/')}
            />
            <div>
              <h1 className="text-2xl font-bold text-dark-text-primary">Admin Portal</h1>
              <p className="text-sm text-dark-text-muted">
                Manage your INAI team members from a single dashboard
              </p>
            </div>
          </div>

          <Button variant="danger" size="medium" onClick={logout}>
            Logout
          </Button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 space-y-8">
        {recentMember && (
          <motion.div
            className="card p-6 border border-green-500/20 bg-green-500/5"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className="flex items-start space-x-3">
              <CheckCircle className="w-6 h-6 text-green-500 mt-1" />
              <div>
                <h3 className="font-medium text-green-400 mb-1">Member Added Successfully!</h3>
                <div className="text-sm text-dark-text-secondary space-y-1">
                  <p><strong>Name:</strong> {recentMember.name}</p>
                  <p><strong>Email:</strong> {recentMember.email}</p>
                  <p><strong>Work Type:</strong> {recentMember.work_type}</p>
                </div>
                <div className="mt-3">
                  <Button variant="secondary" size="small" onClick={() => setRecentMember(null)}>
                    Dismiss
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center">
              <UserPlus className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-dark-text-primary">Add New Member</h2>
              <p className="text-sm text-dark-text-muted">
                Provide access to your team members with appropriate roles
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <InputField
              label="Name"
              placeholder="Enter name"
              error={errors.name?.message}
              {...register('name', {
                required: 'Name is required',
                minLength: { value: 2, message: 'Name must be at least 2 characters' },
              })}
              value={watchValues.name}
            />

            <InputField
              label="Designation"
              placeholder="Enter designation"
              error={errors.designation?.message}
              {...register('designation', {
                required: 'Designation is required',
              })}
              value={watchValues.designation}
            />

            <div className="space-y-2">
              <label className="block text-sm font-medium text-dark-text-primary">
                Work Type <span className="text-red-500">*</span>
              </label>
              <select
                className="input-primary"
                {...workTypeField}
                value={watchValues.workType}
                onChange={(event) => {
                  workTypeField.onChange(event);
                }}
              >
                <option value="">Select work type</option>
                {WORK_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.workType && (
                <p className="text-sm text-red-500">{errors.workType.message}</p>
              )}
            </div>

            <InputField
              label="Phone Number"
              placeholder="10-digit phone number"
              error={errors.phoneNumber?.message}
              {...register('phoneNumber', {
                required: 'Phone number is required',
                pattern: {
                  value: /^\d{10}$/,
                  message: 'Phone number must be 10 digits',
                },
              })}
              value={watchValues.phoneNumber}
            />

            <InputField
              label="Email"
              type="email"
              placeholder="name@example.com"
              error={errors.email?.message}
              {...register('email', {
                required: 'Email is required',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Please enter a valid email address',
                },
              })}
              value={watchValues.email}
            />

            <InputField
              label="Password"
              type="password"
              placeholder="Minimum 8 characters with at least one number"
              showPasswordToggle
              error={errors.password?.message}
              {...register('password', {
                required: 'Password is required',
                minLength: {
                  value: 8,
                  message: 'Password must be at least 8 characters',
                },
              })}
              value={watchValues.password}
            />

            <Button
              type="submit"
              variant="primary"
              size="large"
              fullWidth
              loading={creating}
            >
              {creating ? 'Adding Member...' : 'Add Member'}
            </Button>
          </form>
        </motion.div>

        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-dark-text-primary">Team Members</h2>
              <p className="text-sm text-dark-text-muted">
                Overview of all members associated with your account
              </p>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex space-x-1 bg-dark-card rounded-lg p-1">
                {[{ key: 'all', label: 'All' }, ...WORK_TYPE_OPTIONS.map((opt) => ({
                  key: opt.value,
                  label: opt.label.split(' ')[0],
                }))].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setFilter(tab.key)}
                    className={`px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                      filter === tab.key
                        ? 'bg-primary-600 text-white'
                        : 'text-dark-text-muted hover:text-dark-text-primary'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {membersLoading ? (
            <div className="flex justify-center py-8">
              <div className="spinner w-8 h-8" />
            </div>
          ) : filteredMembers.length === 0 ? (
            <div className="text-center py-12">
              <Users className="w-16 h-16 text-dark-text-muted mx-auto mb-4" />
              <h3 className="text-lg font-medium text-dark-text-primary mb-2">
                No members found
              </h3>
              <p className="text-dark-text-muted mb-4">
                {filter === 'all'
                  ? "You haven't added any team members yet."
                  : 'No members found for the selected work type.'}
              </p>
              <Button variant="primary" onClick={() => setFilter('all')}>
                View All Members
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {filteredMembers.map((member, index) => (
                <motion.div
                  key={member.member_id}
                  className="card p-6 border border-dark-border/70"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: index * 0.05 }}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-primary-600/60 rounded-lg flex items-center justify-center">
                        <span className="text-white text-lg font-semibold">
                          {member.name?.charAt(0)?.toUpperCase() || 'M'}
                        </span>
                      </div>
                      <div>
                        <h3 className="font-semibold text-dark-text-primary">{member.name}</h3>
                        <p className="text-xs text-dark-text-muted uppercase tracking-wide">
                          {member.designation}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="small"
                      icon={Trash2}
                      className="text-red-400 hover:text-red-200 !p-2"
                      onClick={() => handleDelete(member.member_id)}
                    />
                  </div>

                  <div className="space-y-3 text-sm text-dark-text-secondary">
                    <div>
                      <p className="text-xs text-dark-text-muted uppercase tracking-wide">Work Type</p>
                      <p className="text-dark-text-primary font-medium">
                        {WORK_TYPE_OPTIONS.find((option) => option.value === member.work_type)?.label || member.work_type}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-dark-text-muted uppercase tracking-wide">Email</p>
                      <p className="text-dark-text-primary">{member.email}</p>
                    </div>
                    {member.phone_number && (
                      <div>
                        <p className="text-xs text-dark-text-muted uppercase tracking-wide">Phone</p>
                        <p className="text-dark-text-primary">{member.phone_number}</p>
                      </div>
                    )}
                    {member.last_login && (
                      <div>
                        <p className="text-xs text-dark-text-muted uppercase tracking-wide">Last Login</p>
                        <p className="text-dark-text-primary">
                          {new Date(member.last_login).toLocaleString()}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default AdminMembersPage;

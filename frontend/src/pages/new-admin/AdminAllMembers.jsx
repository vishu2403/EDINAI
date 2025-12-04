import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users,
  UserPlus,
  GraduationCap,
  BookOpen,
  Layers,
  Filter,
  X,
  Trash2,
  Mail,
  Phone,
  Lock,
  BadgeIcon,
} from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

import AdminShell from '../../components/new-admin/AdminShell.jsx';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField.jsx';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminMemberAPI } from '../../utils/newAdminApi.js';

const FILTER_OPTIONS = [
  { value: 'all', label: 'All Management' },
  { value: 'chapter_management', label: 'Chapter Management' },
  { value: 'student_management', label: 'Student Management' },
  { value: 'lecture_management', label: 'Lecture Management' },
];

const WORK_TYPE_ALIASES = {
  chapter_management: 'chapter',
  chapter: 'chapter',
  student_management: 'student',
  student: 'student',
  lecture_management: 'lecture',
  lecture: 'lecture',
};

const normalizeWorkType = (value) => WORK_TYPE_ALIASES[value] || value;

const WORK_TYPE_LABELS = {
  chapter: 'Chapter management',
  student: 'Student management',
  lecture: 'Lecture management',
};

const iconByWorkType = {
  chapter: Layers,
  student: GraduationCap,
  lecture: BookOpen,
};

const ManagementListCard = ({ member, onDelete, onEdit }) => {
  const normalizedWorkType = normalizeWorkType(member.work_type);
  const Icon = iconByWorkType[normalizedWorkType] || Users;

  return (
    <motion.div
      layout
      className="rounded-2xl border border-white/5 bg-white/[0.03] backdrop-blur-xl p-6 space-y-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-primary-500/10 flex items-center justify-center">
            <Icon className="w-7 h-7 text-primary-300" />
          </div>
          <div>
            <div className="text-sm text-dark-text-muted">Name :</div>
            <div className="text-lg font-semibold text-white">{member.name}</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-white/10 text-white/60 hover:text-white transition-colors"
            title="Edit member"
            onClick={onEdit}
          >
            <BadgeIcon className="w-4 h-4" />
          </button>
          <button
            type="button"
            className="w-10 h-10 inline-flex items-center justify-center rounded-full bg-red-500/10 text-red-400 hover:text-red-300 transition-colors"
            onClick={onDelete}
            title="Remove member"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-dark-text-secondary">
        <div>
          <span className="text-dark-text-muted">Management :</span>
          <span className="block text-white capitalize">
            {WORK_TYPE_LABELS[normalizedWorkType] || member.work_type}
          </span>
        </div>
        <div>
          <span className="text-dark-text-muted">Designation :</span>
          <span className="block text-white">{member.designation || '—'}</span>
        </div>
        <div>
          <span className="text-dark-text-muted">Email :</span>
          <span className="block text-white break-all flex items-center gap-2">
            <Mail className="w-4 h-4 text-primary-300" />
            {member.email}
          </span>
        </div>
        <div>
          <span className="text-dark-text-muted">Phone Number :</span>
          <span className="block text-white flex items-center gap-2">
            <Phone className="w-4 h-4 text-primary-300" />
            {member.phone_number || '—'}
          </span>
        </div>
        <div>
          <span className="text-dark-text-muted">Password :</span>
          <span className="block text-white flex items-center gap-2">
            <Lock className="w-4 h-4 text-primary-300" />
            ******
          </span>
        </div>
        {member.login_id && (
          <div>
            <span className="text-dark-text-muted">Login ID :</span>
            <span className="block text-white">{member.login_id}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const AddMemberDrawer = ({ open, onClose, onSubmit, loading, mode = 'create', member }) => {
  const defaultForm = {
    name: '',
    designation: '',
    work_type: 'chapter_management',
    phone_number: '',
    email: '',
    password: '',
  };
  const [form, setForm] = useState(defaultForm);

  useEffect(() => {
    if (!open) {
      setForm(defaultForm);
      return;
    }

    if (mode === 'edit' && member) {
      setForm({
        name: member.name || '',
        designation: member.designation || '',
        work_type: FILTER_OPTIONS.find((option) => normalizeWorkType(option.value) === normalizeWorkType(member.work_type))?.value || 'chapter_management',
        phone_number: member.phone_number || '',
        email: member.email || '',
        password: '',
      });
    } else {
      setForm(defaultForm);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode, member]);

  const handleChange = (field) => (event) => {
    setForm((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    await onSubmit(form, mode);
    setForm(defaultForm);
  };

  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50">
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          <motion.aside
            className="absolute top-0 right-0 h-full w-full max-w-md bg-[#0B0C10] border-l border-white/10 shadow-2xl overflow-y-auto"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 260 }}
          >
            <div className="flex items-center justify-between px-6 py-6 border-b border-white/10">
              <div>
                <h2 className="text-xl font-semibold text-white">
                  {mode === 'edit' ? 'Edit Member' : 'Member Information'}
                </h2>
                <p className="text-sm text-dark-text-muted">
                  {mode === 'edit'
                    ? 'Update details for this management member'
                    : 'Create access for a new management member'}
                </p>
              </div>
              <Button variant="ghost" size="medium" icon={X} onClick={onClose} />
            </div>

            <form onSubmit={handleSubmit} className="px-6 py-6 space-y-6">
              <InputField
                label="Name Of Person"
                placeholder="Enter name"
                value={form.name}
                onChange={handleChange('name')}
                required
              />
              <InputField
                label="Designation"
                placeholder="Your Designation"
                value={form.designation}
                onChange={handleChange('designation')}
                required
              />

              <div className="space-y-2">
                <label className="block text-sm font-medium text-white">Work Type</label>
                <div className="relative">
                  <select
                    className="input-primary appearance-none w-full"
                    value={form.work_type}
                    onChange={handleChange('work_type')}
                  >
                    {FILTER_OPTIONS.filter((option) => option.value !== 'all').map((option) => (
                      <option key={option.value} value={option.value} className="bg-[#050507]">
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-dark-text-muted" />
                </div>
              </div>

              <InputField
                label="Phone Number"
                placeholder="Enter Your Number"
                value={form.phone_number}
                onChange={handleChange('phone_number')}
                required
                maxLength={10}
              />

              <InputField
                label="Email"
                placeholder="Enter Your Email"
                value={form.email}
                onChange={handleChange('email')}
                type="email"
                required
              />

              <InputField
                label="Password"
                placeholder={mode === 'edit' ? 'Leave blank to keep current password' : '********'}
                value={form.password}
                onChange={handleChange('password')}
                type="password"
                required={mode === 'create'}
                showPasswordToggle
              />

              <Button type="submit" variant="primary" size="large" fullWidth loading={loading}>
                {loading ? 'Saving...' : mode === 'edit' ? 'Update Member' : 'Save Member'}
              </Button>
            </form>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
};

const AdminAllMembers = () => {
  const { admin } = useNewAdminAuth();
  const adminId = admin?.admin_id;

  const [filter, setFilter] = useState('all');
  const [members, setMembers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerMode, setDrawerMode] = useState('create');
  const [editingMember, setEditingMember] = useState(null);

  const filteredMembers = useMemo(() => {
    if (filter === 'all') return members;
    const normalizedFilter = normalizeWorkType(filter);
    return members.filter((member) => normalizeWorkType(member.work_type) === normalizedFilter);
  }, [filter, members]);

  const memberCounts = useMemo(() => {
    return members.reduce((acc, member) => {
      const key = normalizeWorkType(member.work_type);
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
  }, [members]);

  const loadMembers = async (workType = 'all') => {
    if (!adminId) return;
    setFetching(true);
    try {
      const response = await newAdminMemberAPI.listMembers({ adminId, workType });
      const memberPayload = response?.members || response?.data?.members || [];
      setMembers(memberPayload);
    } catch (error) {
      console.error('Failed to load members', error);
      toast.error(error.response?.data?.detail || 'Failed to load members');
    } finally {
      setFetching(false);
    }
  };

  useEffect(() => {
    if (adminId) {
      loadMembers(filter);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminId, filter]);

  const handleCreate = async (formValues) => {
    if (!adminId) return;

    setLoading(true);
    try {
      await newAdminMemberAPI.createMember({
        adminId,
        data: {
          name: formValues.name.trim(),
          designation: formValues.designation.trim(),
          work_type: formValues.work_type,
          phone_number: formValues.phone_number.trim(),
          email: formValues.email.trim(),
          password: formValues.password,
        },
      });

      toast.success('Member added successfully');
      setDrawerOpen(false);
      await loadMembers(filter);
    } catch (error) {
      console.error('Failed to create member', error);
      toast.error(error.response?.data?.detail || 'Failed to create member');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async (formValues) => {
    if (!adminId || !editingMember) return;

    const payload = {};
    const trimmedName = formValues.name.trim();
    if (trimmedName && trimmedName !== editingMember.name) {
      payload.name = trimmedName;
    }

    const trimmedDesignation = formValues.designation.trim();
    if (trimmedDesignation && trimmedDesignation !== editingMember.designation) {
      payload.designation = trimmedDesignation;
    }

    const normalizedWorkType = normalizeWorkType(formValues.work_type);
    if (normalizedWorkType && normalizedWorkType !== normalizeWorkType(editingMember.work_type)) {
      payload.work_type = normalizedWorkType;
    }

    const trimmedPhone = formValues.phone_number.trim();
    if ((trimmedPhone || editingMember.phone_number) && trimmedPhone !== (editingMember.phone_number || '')) {
      payload.phone_number = trimmedPhone;
    }

    const trimmedEmail = formValues.email.trim();
    if (trimmedEmail && trimmedEmail !== editingMember.email) {
      payload.email = trimmedEmail;
    }

    if (formValues.password) {
      payload.password = formValues.password;
    }

    if (Object.keys(payload).length === 0) {
      toast('No changes to update');
      return;
    }

    setLoading(true);
    try {
      await newAdminMemberAPI.updateMember({
        adminId,
        memberId: editingMember.member_id,
        data: payload,
      });

      toast.success('Member updated successfully');
      setDrawerOpen(false);
      setEditingMember(null);
      await loadMembers(filter);
    } catch (error) {
      console.error('Failed to update member', error);
      toast.error(error.response?.data?.detail || 'Failed to update member');
    } finally {
      setLoading(false);
    }
  };

  const handleDrawerSubmit = async (formValues, mode) => {
    if (mode === 'edit') {
      await handleUpdate(formValues);
    } else {
      await handleCreate(formValues);
    }
  };

  const openCreateDrawer = () => {
    setDrawerMode('create');
    setEditingMember(null);
    setDrawerOpen(true);
  };

  const openEditDrawer = (member) => {
    setDrawerMode('edit');
    setEditingMember(member);
    setDrawerOpen(true);
  };

  const handleDelete = async (memberId) => {
    if (!adminId) return;

    const confirmed = window.confirm('Are you sure you want to delete this member?');
    if (!confirmed) return;

    try {
      await newAdminMemberAPI.deleteMember({ adminId, memberId });
      toast.success('Member removed');
      await loadMembers(filter);
    } catch (error) {
      console.error('Failed to delete member', error);
      toast.error(error.response?.data?.detail || 'Failed to delete member');
    }
  };

  return (
    <AdminShell title="Management List" description="Organize chapter, lecture and student management teams.">
      <div className="space-y-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-dark-text-muted text-sm">Select management to view members created by you.</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <select
                className="bg-white/[0.04] border border-white/10 rounded-xl px-4 py-2.5 pr-12 text-sm text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                {FILTER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value} className="bg-[#050507]">
                    {option.label}
                  </option>
                ))}
              </select>
              <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-text-muted" />
            </div>
            <Button variant="primary" icon={UserPlus} onClick={openCreateDrawer}>
              Add Member
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
                  <Users className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm text-dark-text-muted">Total Members</p>
                  <p className="text-xl font-semibold text-white">{members.length}</p>
                </div>
              </div>
            </div>
            <div className="space-y-2 text-sm text-dark-text-muted">
              {FILTER_OPTIONS.filter((option) => option.value !== 'all').map((option) => {
                const normalizedOption = normalizeWorkType(option.value);
                const count = memberCounts[normalizedOption] ?? 0;
                return (
                  <div key={option.value} className="flex items-center justify-between">
                    <span>{option.label}</span>
                    <span className="text-white">{count}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="md:col-span-2 lg:col-span-2 space-y-5">
            {fetching ? (
              <div className="w-full flex items-center justify-center py-16">
                <div className="spinner w-10 h-10" />
              </div>
            ) : filteredMembers.length === 0 ? (
              <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-10 text-center space-y-4">
                <Users className="w-12 h-12 mx-auto text-dark-text-muted" />
                <h3 className="text-lg font-semibold text-white">No members found</h3>
                <p className="text-dark-text-muted">Add your first member to start managing your portal access.</p>
                <Button variant="primary" icon={UserPlus} onClick={() => setDrawerOpen(true)}>
                  Add Member
                </Button>
              </div>
            ) : (
              <motion.div layout className="space-y-4">
                {filteredMembers.map((member) => (
                  <ManagementListCard
                    key={member.member_id}
                    member={member}
                    onDelete={() => handleDelete(member.member_id)}
                    onEdit={() => openEditDrawer(member)}
                  />
                ))}
              </motion.div>
            )}
          </div>
        </div>
      </div>

      <AddMemberDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onSubmit={handleDrawerSubmit}
        loading={loading}
        mode={drawerMode}
        member={editingMember}
      />
    </AdminShell>
  );
};

export default AdminAllMembers;

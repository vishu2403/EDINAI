import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mail, Phone, Shield } from 'lucide-react';

import AdminShell from '../../components/new-admin/AdminShell.jsx';
import Button from '../../components/ui/Button';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminContactAPI } from '../../utils/newAdminApi.js';

const resolveAssetUrl = (path) => {
  if (!path) {
    return null;
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  if (path.startsWith('/')) {
    return path;
  }

  const normalized = path.replace(/^\/+/, '');
  return `/${normalized}`;
};

const formatDate = (value) => {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

const AdminProfile = () => {
  const navigate = useNavigate();
  const { admin, updateAdmin } = useNewAdminAuth();
  const adminId = admin?.admin_id;

  const [contact, setContact] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let isMounted = true;

    const fetchContact = async () => {
      if (!adminId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError('');

      try {
        const response = await newAdminContactAPI.getContact(adminId);
        if (!isMounted) return;

        if (response?.status && response?.contact) {
          setContact(response.contact);
          const imagePath = response.contact.image_path?.[0];
          if (imagePath) {
            updateAdmin({ avatar_url: resolveAssetUrl(imagePath) });
          }
        } else {
          setContact(null);
        }
      } catch (fetchError) {
        if (!isMounted) return;

        if (fetchError?.response?.status === 404) {
          setContact(null);
        } else {
          console.error('Failed to load admin contact profile', fetchError);
          const message =
            fetchError?.response?.data?.detail ||
            fetchError?.response?.data?.message ||
            'Unable to load your profile right now.';
          setError(message);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchContact();

    return () => {
      isMounted = false;
    };
  }, [adminId]);

  const profile = useMemo(() => {
    const firstName = contact?.first_name ?? admin?.first_name ?? '';
    const lastName = contact?.last_name ?? admin?.last_name ?? '';
    const fallbackFullName =
      admin?.full_name || [admin?.first_name, admin?.last_name].filter(Boolean).join(' ');

    const fullName = [firstName, lastName].filter(Boolean).join(' ') || fallbackFullName || 'Administrator';

    const email =
      contact?.inai_email ||
      contact?.email ||
      admin?.email ||
      'Not provided';

    const phone =
      contact?.phone_number ||
      admin?.phone_number ||
      admin?.phone ||
      'Not provided';

    const designation = contact?.designation || admin?.designation || 'Not provided';
    const avatarPath = resolveAssetUrl(contact?.image_path?.[0] || admin?.avatar_url);
    const avatarLetter = fullName.trim().charAt(0).toUpperCase() || 'A';
    const lastUpdated = contact?.updated_at || admin?.last_contact_update || admin?.updated_at || null;

    return {
      fullName,
      email,
      phone,
      designation,
      avatarUrl: avatarPath,
      avatarLetter,
      lastUpdated: formatDate(lastUpdated),
    };
  }, [contact, admin]);

  return (
    <AdminShell
      title="Profile"
      description="Review and manage your administrator contact details."
    >
      <motion.div
        className="bg-[#0E0F14] rounded-2xl border border-white/5 shadow-lg p-6 md:p-8"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-20 h-20 overflow-hidden rounded-full border border-primary-500/40 bg-primary-500/20 flex items-center justify-center text-3xl font-semibold">
              {profile.avatarUrl ? (
                <img
                  src={profile.avatarUrl}
                  alt={`${profile.fullName}'s avatar`}
                  className="h-full w-full object-cover"
                />
              ) : (
                profile.avatarLetter
              )}
            </div>
            <div>
              <h2 className="text-2xl font-semibold">{profile.fullName}</h2>
              <p className="text-sm text-primary-300 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                {profile.designation}
              </p>
              {profile.lastUpdated && (
                <p className="mt-2 text-xs text-dark-text-muted">
                  Last updated {profile.lastUpdated}
                </p>
              )}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <Button
              variant="secondary"
              size="medium"
              onClick={() => navigate(-1)}
            >
              Back
            </Button>
            <Button
              variant="primary"
              size="medium"
              onClick={() => navigate('/admin/profile/edit')}
            >
              Edit Profile
            </Button>
          </div>
        </div>

        {error && (
          <div className="mt-6 rounded-xl border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-200">
            {error}
          </div>
        )}

        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          <motion.div
            className="rounded-2xl border border-white/5 bg-[#10131A] p-5"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 }}
          >
            <div className="text-xs uppercase tracking-wide text-dark-text-muted">Email</div>
            <div className="mt-2 flex items-center gap-3">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-primary-500/10 text-primary-400">
                <Mail className="h-5 w-5" />
              </span>
              <div>
                <div className="text-sm text-dark-text-secondary">Primary</div>
                <div className="text-lg font-medium break-all">{profile.email}</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="rounded-2xl border border-white/5 bg-[#10131A] p-5"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="text-xs uppercase tracking-wide text-dark-text-muted">Phone</div>
            <div className="mt-2 flex items-center gap-3">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-primary-500/10 text-primary-400">
                <Phone className="h-5 w-5" />
              </span>
              <div>
                <div className="text-sm text-dark-text-secondary">Contact Number</div>
                <div className="text-lg font-medium">{profile.phone}</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="rounded-2xl border border-white/5 bg-[#10131A] p-5"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            <div className="text-xs uppercase tracking-wide text-dark-text-muted">Designation</div>
            <div className="mt-3 flex items-center gap-3">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-primary-500/10 text-primary-400">
                <Shield className="h-5 w-5" />
              </span>
              <div>
                <div className="text-lg font-medium">{profile.designation}</div>
                <div className="text-sm text-dark-text-muted">Role within your institution</div>
              </div>
            </div>
          </motion.div>
        </div>

        {loading && (
          <div className="mt-6 flex items-center gap-3 text-sm text-dark-text-muted">
            <div className="spinner w-5 h-5" aria-hidden="true" />
            Loading profile details...
          </div>
        )}
      </motion.div>
    </AdminShell>
  );
};

export default AdminProfile;
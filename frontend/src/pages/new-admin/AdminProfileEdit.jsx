import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { Camera, Lock, Phone, User } from 'lucide-react';

import AdminShell from '../../components/new-admin/AdminShell.jsx';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminContactAPI } from '../../utils/newAdminApi.js';

const ACCEPTED_AVATAR_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const MAX_AVATAR_SIZE = 5 * 1024 * 1024; // 5MB

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

const splitFullName = (fullName) => {
  if (!fullName) {
    return { firstName: '', lastName: '' };
  }

  const parts = fullName.trim().split(/\s+/);
  if (parts.length === 1) {
    return { firstName: parts[0], lastName: '' };
  }

  const firstName = parts.slice(0, -1).join(' ');
  const lastName = parts.slice(-1).join(' ');
  return { firstName, lastName };
};

const buildFormData = ({
  contact,
  admin,
  fullName,
  phone,
  avatarFile,
}) => {
  const { firstName, lastName } = splitFullName(fullName);
  const payload = new FormData();

  payload.append('first_name', firstName);
  payload.append('last_name', lastName);
  payload.append('phone_number', phone);

  // Preserve existing values for fields not editable here to avoid accidental data loss
  payload.append('designation', contact?.designation || admin?.designation || '');
  payload.append('address', contact?.address || admin?.address || '');
  payload.append('education_center_name', contact?.education_center_name || admin?.education_center_name || '');
  payload.append('dob', contact?.dob || '');
  const inaiEmail = contact?.inai_email || admin?.email || '';
  if (inaiEmail) {
    payload.append('inai_email', inaiEmail);
  }

  // INAI password is stored but never surfaced; keep the same value if API expects it
  if (contact?.inai_password) {
    payload.append('inai_password', contact.inai_password);
  }

  if (avatarFile) {
    payload.append('image', avatarFile);
  }

  return payload;
};

const AdminProfileEdit = () => {
  const navigate = useNavigate();
  const { admin, updateAdmin } = useNewAdminAuth();
  const adminId = admin?.admin_id;

  const [contact, setContact] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState({});
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const fetchContact = async () => {
      if (!adminId) {
        setLoading(false);
        return;
      }

      try {
        const response = await newAdminContactAPI.getContact(adminId);
        if (!isMounted) return;

        if (response?.status && response?.contact) {
          setContact(response.contact);
        } else {
          setContact(null);
        }
      } catch (error) {
        if (!isMounted) return;

        if (error?.response?.status === 404) {
          setContact(null);
        } else {
          console.error('Failed to load contact profile for editing', error);
          toast.error(
            error?.response?.data?.detail ||
            error?.response?.data?.message ||
            'Unable to load profile details.'
          );
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

  const contactAvatarUrl = useMemo(() => {
    const path = contact?.image_path?.[0];
    return resolveAssetUrl(path);
  }, [contact]);

  useEffect(() => {
    if (!avatarFile) {
      setAvatarPreview(contactAvatarUrl);
    }
  }, [contactAvatarUrl, avatarFile]);

  const initialValues = useMemo(() => {
    const firstName = contact?.first_name || admin?.first_name || '';
    const lastName = contact?.last_name || admin?.last_name || '';
    const fallbackFullName = admin?.full_name || [admin?.first_name, admin?.last_name].filter(Boolean).join(' ');
    const fullName = [firstName, lastName].filter(Boolean).join(' ') || fallbackFullName || '';

    const phone = contact?.phone_number || admin?.phone_number || admin?.phone || '';

    return { fullName, phone };
  }, [contact, admin]);

  const [formValues, setFormValues] = useState(initialValues);

  useEffect(() => {
    setFormValues(initialValues);
  }, [initialValues]);

  const handleChange = (field) => (event) => {
    setFormValues((prev) => ({ ...prev, [field]: event.target.value }));
  };

  const handleAvatarClick = () => {
    fileInputRef.current?.click();
  };

  const handleAvatarChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (!ACCEPTED_AVATAR_TYPES.includes(file.type)) {
      toast.error('Please select a PNG, JPG, or WebP image.');
      event.target.value = '';
      return;
    }

    if (file.size > MAX_AVATAR_SIZE) {
      toast.error('Profile image must be 5MB or smaller.');
      event.target.value = '';
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setAvatarPreview((prev) => {
      if (prev && prev.startsWith('blob:')) {
        URL.revokeObjectURL(prev);
      }
      return objectUrl;
    });
    setAvatarFile(file);
    event.target.value = '';
  };

  useEffect(() => {
    return () => {
      if (avatarPreview && avatarPreview.startsWith('blob:')) {
        URL.revokeObjectURL(avatarPreview);
      }
    };
  }, [avatarPreview]);

  const validate = () => {
    const nextErrors = {};
    const trimmedName = formValues.fullName.trim();
    const trimmedPhone = formValues.phone.trim();

    if (!trimmedName) {
      nextErrors.fullName = 'Name is required';
    }

    if (!trimmedPhone) {
      nextErrors.phone = 'Phone number is required';
    } else if (!/^\+?\d{10,15}$/.test(trimmedPhone.replace(/\s+/g, ''))) {
      nextErrors.phone = 'Enter a valid phone number (10-15 digits)';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) {
      return;
    }

    if (!adminId) {
      toast.error('Admin information missing. Please log in again.');
      return;
    }

    setSubmitting(true);
    try {
      const payload = buildFormData({
        contact,
        admin,
        fullName: formValues.fullName.trim(),
        phone: formValues.phone.trim(),
        avatarFile,
      });
      payload.append('admin_id', adminId);

      const response = await newAdminContactAPI.upsertContact(payload);

      const { firstName, lastName } = splitFullName(formValues.fullName.trim());
      const imagePath = response?.image_path?.[0] || response?.contact?.image_path?.[0];
      updateAdmin({
        first_name: firstName,
        last_name: lastName,
        full_name: formValues.fullName.trim(),
        phone_number: formValues.phone.trim(),
        phone: formValues.phone.trim(),
        last_contact_update: new Date().toISOString(),
        avatar_url: resolveAssetUrl(imagePath) || undefined,
      });

      toast.success('Profile updated successfully.');
      navigate('/admin/profile', { replace: true });
    } catch (error) {
      console.error('Failed to update admin profile', error);
      const message =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        'Unable to update profile right now.';
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AdminShell
      title="Edit Profile"
      description="Update your administrator display name and contact number."
    >
      <motion.div
        className="bg-[#0E0F14] rounded-2xl border border-white/5 shadow-lg p-6 md:p-8"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_AVATAR_TYPES.join(',')}
          className="hidden"
          onChange={handleAvatarChange}
        />

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={handleAvatarClick}
              className="group relative h-20 w-20 overflow-hidden rounded-full border border-white/10 bg-primary-500/20 transition hover:border-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-400"
              title="Change profile photo"
              aria-label="Change profile photo"
            >
              {avatarPreview ? (
                <img
                  src={avatarPreview}
                  alt="Profile preview"
                  className="h-full w-full object-cover"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-2xl font-semibold text-primary-100">
                  {(formValues.fullName?.trim() || admin?.full_name || 'A').charAt(0).toUpperCase() || 'A'}
                </span>
              )}
              <span className="pointer-events-none absolute inset-0 flex items-end justify-center bg-gradient-to-t from-black/60 to-transparent p-2 text-xs font-medium text-white opacity-0 transition group-hover:opacity-100">
                <span className="flex items-center gap-1"><Camera className="h-3.5 w-3.5" /> Update</span>
              </span>
            </button>
            <div className="space-y-1">
              <p className="text-sm font-medium text-dark-text-secondary">Profile photo</p>
              <p className="text-xs text-dark-text-muted">PNG, JPG or WebP up to 5MB. Click the avatar to upload.</p>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <InputField
              label="Username"
              placeholder="Enter full name"
              icon={User}
              value={formValues.fullName}
              onChange={handleChange('fullName')}
              error={errors.fullName}
            />

            <InputField
              label="Phone Number"
              placeholder="Enter phone number"
              icon={Phone}
              value={formValues.phone}
              onChange={handleChange('phone')}
              error={errors.phone}
            />
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex flex-col gap-2 text-xs text-dark-text-muted">
              <span>Only your display name and contact number can be edited here.</span>
              <Button
                type="button"
                variant="ghost"
                size="small"
                icon={Lock}
                onClick={() => navigate('/admin/profile/change-password')}
                className="justify-start sm:justify-center text-primary-300 hover:text-primary-200"
              >
                Change password
              </Button>
            </div>
            <div className="flex items-center gap-3">
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
                Save Changes
              </Button>
            </div>
          </div>
        </form>

        {loading && (
          <div className="mt-6 flex items-center gap-3 text-sm text-dark-text-muted">
            <div className="spinner w-5 h-5" aria-hidden="true" />
            Loading current profile details...
          </div>
        )}
      </motion.div>
    </AdminShell>
  );
};

export default AdminProfileEdit;
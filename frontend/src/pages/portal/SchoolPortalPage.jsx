import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Lock, Eye, EyeOff } from 'lucide-react';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

import { schoolPortalAPI, getFileUrl } from '../../utils/api';

const formFields = [
  [
    { name: 'firstName', label: 'First Name', placeholder: 'Enter first name' },
    { name: 'fatherName', label: 'Father Name', placeholder: 'Enter father name' },
  ],
  [
    { name: 'classStream', label: 'Class / Stream', placeholder: 'Enter class / stream' },
    { name: 'division', label: 'Division', placeholder: 'Enter division' },
  ],
  [
    { name: 'classHead', label: 'Class Head', placeholder: 'Enter class head' },
    { name: 'enrollmentNumber', label: 'Enrollment Number', placeholder: 'Enter enrollment number' },
  ],
  [
    { name: 'mobileNumber', label: 'Mobile Number', placeholder: 'Enter mobile number' },
    { name: 'parentsNumber', label: 'Parents Number', placeholder: 'Enter parents number' },
  ],
  [
    { name: 'email', label: 'Email Address', placeholder: 'Enter email address', type: 'email' },
  ],
];

const initialFormState = formFields
  .flat()
  .reduce((acc, field) => ({ ...acc, [field.name]: '' }), { photo: null });

const PROFILE_KEY = 'inai_student_profile_complete';
const PROFILE_DATA_KEY = 'inai_student_profile_data';
const PROFILE_ENROLLMENT_KEY = 'inai_student_profile_enrollment';
const STUDENT_TOKEN_KEY = 'inai_student_token';
const VIEW_LOGIN = 'login';
const VIEW_PROFILE = 'profile';

const SchoolPortalPage = () => {
  const [formValues, setFormValues] = useState(initialFormState);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeView, setActiveView] = useState(VIEW_LOGIN);
  const [isReady, setIsReady] = useState(false);
  const [loginValues, setLoginValues] = useState({ enrollmentNumber: '', password: '' });
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const renderBackButton = () => (
    <div className="px-6 pt-6 lg:px-10">
      <button
        type="button"
        onClick={() => navigate('/')}
        className="text-sm font-medium text-white/60 transition hover:text-white"
      >
        ← Back to role selection
      </button>
    </div>
  );

  const applyPrefill = (prefill) => {
    if (!prefill) return;
    setFormValues((prev) => ({
      ...prev,
      firstName: prefill.first_name || prefill.firstName || '',
      fatherName: prefill.father_name || prefill.fatherName || '',
      classStream: prefill.class_stream || prefill.classStream || '',
      division: prefill.division || '',
      classHead: prefill.class_head || prefill.classHead || '',
      enrollmentNumber: prefill.enrollment_number || prefill.enrollmentNumber || '',
      mobileNumber: prefill.mobile_number || prefill.mobileNumber || '',
      parentsNumber: prefill.parents_number || prefill.parentsNumber || '',
      email: prefill.email || '',
      photo: null,
    }));
  };

  const fetchProfileStatus = async (enrollmentNumber) => {
    if (!enrollmentNumber) return null;
    try {
      const statusResponse = await schoolPortalAPI.getProfileStatus(enrollmentNumber);
      if (statusResponse?.prefill) {
        applyPrefill(statusResponse.prefill);
      }
      return statusResponse;
    } catch (error) {
      console.error('Failed to fetch profile status', error);
      return null;
    }
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const savedEnrollment = window.localStorage.getItem(PROFILE_ENROLLMENT_KEY) || '';
    if (savedEnrollment) {
      setLoginValues((prev) => ({ ...prev, enrollmentNumber: savedEnrollment }));
    }

    setIsReady(true);
  }, []);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const handlePhotoChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    setFormValues((prev) => ({ ...prev, photo: file }));

    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setPhotoPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setPhotoPreview(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);

    try {
      const formData = new FormData();
      const requiredFields = [
        ['first_name', formValues.firstName],
        ['father_name', formValues.fatherName],
        ['class_stream', formValues.classStream],
        ['enrollment_number', formValues.enrollmentNumber],
      ];
      requiredFields.forEach(([key, value]) => {
        formData.append(key, (value || '').trim());
      });

      const optionalFields = [
        ['division', formValues.division],
        ['class_head', formValues.classHead],
        ['mobile_number', formValues.mobileNumber],
        ['parents_number', formValues.parentsNumber],
        ['email', formValues.email],
      ];
      optionalFields.forEach(([key, value]) => {
        const cleaned = (value || '').trim();
        if (cleaned) {
          formData.append(key, cleaned);
        }
      });

      if (formValues.photo) {
        formData.append('photo', formValues.photo);
      }

      const response = await schoolPortalAPI.saveProfile(formData);

      const storedProfile = {
        firstName: response.first_name,
        fatherName: response.father_name,
        classStream: response.class_stream,
        division: response.division ?? '',
        classHead: response.class_head ?? '',
        enrollmentNumber: response.enrollment_number,
        mobileNumber: response.mobile_number ?? '',
        parentsNumber: response.parents_number ?? '',
        email: response.email ?? '',
        advisorEmail: null,
        photoDataUrl: response.photo_path ? getFileUrl(response.photo_path) : photoPreview,
      };

      if (typeof window !== 'undefined') {
        window.localStorage.setItem(PROFILE_DATA_KEY, JSON.stringify(storedProfile));
        window.localStorage.setItem(PROFILE_KEY, 'true');
        window.localStorage.setItem(PROFILE_ENROLLMENT_KEY, response.enrollment_number);
      }

      setLoginValues((prev) => ({ ...prev, enrollmentNumber: response.enrollment_number }));

      toast.success('Student details saved successfully.');
      setFormValues(initialFormState);
      setPhotoPreview(null);
      setActiveView(VIEW_LOGIN);
    } catch (error) {
      const message = error.response?.data?.detail || 'Unable to save student details right now.';
      toast.error(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const photoLabel = formValues.photo?.name || 'Click to upload or drag and drop JPEG';

  const handleLoginChange = (event) => {
    const { name, value } = event.target;
    setLoginValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleLoginSubmit = async (event) => {
    event.preventDefault();
    const enrollmentNumber = loginValues.enrollmentNumber.trim();
    const password = loginValues.password.trim();

    if (!enrollmentNumber) {
      toast.error('Enrollment number is required.');
      return;
    }

    if (!password) {
      toast.error('Password is required.');
      return;
    }

    setIsLoggingIn(true);
    try {
      const response = await schoolPortalAPI.login(enrollmentNumber, password);

      if (!response?.status) {
        toast.error(response?.message || 'Unable to sign in right now.');
        return;
      }

      if (typeof window !== 'undefined') {
        if (response.token) {
          window.localStorage.setItem(STUDENT_TOKEN_KEY, response.token);
        }
        window.localStorage.setItem(PROFILE_ENROLLMENT_KEY, enrollmentNumber);
      }

      const profileComplete = Boolean(response.profile_complete);

      if (typeof window !== 'undefined') {
        window.localStorage.setItem(PROFILE_KEY, profileComplete ? 'true' : 'false');
        if (!profileComplete) {
          window.localStorage.removeItem(PROFILE_DATA_KEY);
        }
      }

      if (!profileComplete) {
        await fetchProfileStatus(enrollmentNumber);
        setActiveView(VIEW_PROFILE);
        toast.success('Login successful. Please complete your profile to continue.');
        return;
      }

      toast.success(response.message || 'Login successful.');
      navigate('/school-portal/dashboard', { replace: true });
    } catch (error) {
      const statusCode = error.response?.status;
      let message =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Unable to sign in right now.';

      if (statusCode === 401) {
        message = 'Invalid enrollment number or password. Please try again.';
      }

      toast.error(message);
    } finally {
      setIsLoggingIn(false);
    }
  };

  if (!isReady) {
    return null;
  }

  if (activeView === VIEW_LOGIN) {
    return (
      <div className="min-h-screen bg-black text-white">
        {renderBackButton()}
        <div className="flex min-h-screen flex-col items-center justify-center px-6 py-12">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <div className="text-[82px] font-black leading-none">INAI</div>
            <div className="text-[22px] tracking-[0.6em] uppercase">Verse</div>
            <p className="mt-6 text-sm text-white/70">Student Portal</p>
            <p className="mt-1 text-xs text-white/50">
              Enter your credentials to proceed with your account
            </p>
          </motion.div>

          <motion.form
            onSubmit={handleLoginSubmit}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-10 w-full max-w-lg rounded-3xl bg-[#141720] p-8 shadow-2xl"
          >
            <header className="mb-6">
              <h2 className="text-lg font-semibold">Sign In Your Account</h2>
              <p className="mt-1 text-xs text-white/60">
                Access your account to explore INAI&apos;s feature
              </p>
            </header>

            <div className="space-y-4">
              <label className="block text-sm">
                <span className="text-xs font-medium uppercase tracking-wide text-white/60">
                  Enrollment Number
                </span>
                <input
                  type="text"
                  name="enrollmentNumber"
                  value={loginValues.enrollmentNumber}
                  onChange={handleLoginChange}
                  placeholder="Enter enrollment number"
                  className="mt-2 w-full rounded-xl border border-white/10 bg-[#1E2233] px-4 py-3 text-sm text-white placeholder:text-white/40 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                />
              </label>

              <label className="block text-sm">
                <span className="text-xs font-medium uppercase tracking-wide text-white/60">Password</span>
                <div className="relative mt-2">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    value={loginValues.password}
                    onChange={handleLoginChange}
                    placeholder="Enter Password"
                    className="w-full rounded-xl border border-white/10 bg-[#1E2233] px-10 py-3 pr-12 text-sm text-white placeholder:text-white/40 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                  />
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/40" />
                  <button
                    type="button"
                    onClick={() => setShowPassword((prev) => !prev)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1 text-white/60 transition hover:text-white"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </label>
            </div>

            <button
              type="submit"
              disabled={isLoggingIn}
              className="mt-8 flex w-full items-center justify-center rounded-full bg-white py-3 text-sm font-semibold text-black transition hover:scale-[1.01] focus:outline-none focus:ring-2 focus:ring-primary-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoggingIn ? 'Signing in...' : 'Login'}
            </button>

            <div className="mt-6 text-center text-[11px] text-white/40">
              Having trouble? Contact your administrator.
            </div>
          </motion.form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {renderBackButton()}
      <div className="flex min-h-screen flex-col lg:flex-row">
        {/* Left branding panel */}
        <motion.section
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="flex w-full items-center justify-center bg-black px-10 py-12 lg:w-1/2"
        >
          <div className="text-center">
            <p className="mb-5 text-[11px] uppercase tracking-[0.6em] text-white/50">INAI</p>
            <div className="text-[80px] font-black leading-none">INAI</div>
            <div className="text-[20px] tracking-[0.6em] uppercase">Verse</div>
          </div>
        </motion.section>

        {/* Right form panel */}
        <motion.section
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex w-full items-center justify-center bg-[#0F111A] px-6 py-12 sm:px-8 lg:w-1/2"
        >
          <div className="w-full max-w-2xl rounded-2xl border border-white/5 bg-[#161924] p-6 shadow-2xl">
            <header className="mb-6">
              <h1 className="text-xl font-semibold">Student Details</h1>
              <p className="mt-1.5 text-xs text-white/60">
                Fill in the information below to submit your profile for verification.
              </p>
            </header>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="grid gap-3 md:grid-cols-2">
                {formFields.flat().map(({ name, label, placeholder, type = 'text' }) => (
                  <label key={name} className="flex flex-col gap-2 text-sm">
                    <span className="text-xs font-medium uppercase tracking-wide text-white/60">{label}</span>
                    <input
                      type={type}
                      name={name}
                      value={formValues[name]}
                      onChange={handleInputChange}
                      placeholder={placeholder}
                      className="w-full rounded-xl border border-white/10 bg-[#1E2233] px-3.5 py-2.5 text-sm text-white placeholder:text-white/40 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-400/40"
                    />
                  </label>
                ))}
              </div>

              <div className="space-y-2.5">
                <span className="block text-xs font-medium uppercase tracking-wide text-white/60">
                  Upload Student Photo
                </span>
                <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-primary-500/50 bg-[#1B2032] px-5 py-8 text-center transition hover:border-primary-400 hover:bg-[#21273D]">
                  <UploadCloud className="h-8 w-8 text-primary-400" />
                  <p className="text-sm text-white/70">{photoLabel}</p>
                  <span className="text-[11px] text-white/40">JPEG, PNG up to 5MB</span>
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handlePhotoChange}
                    className="sr-only"
                  />
                </label>
              </div>

              <div className="flex justify-center">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="inline-flex min-w-[140px] items-center justify-center rounded-full bg-white py-2.5 text-sm font-semibold text-black transition hover:scale-[1.01] focus:outline-none focus:ring-2 focus:ring-primary-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSubmitting ? 'Saving...' : 'Save'}
                </button>
              </div>
            </form>
          </div>
        </motion.section>
      </div>
    </div>
  );
};

export default SchoolPortalPage;

import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  ArrowLeft,
  ArrowRight,
  Building,
  Calendar,
  CheckCircle,
  Lock,
  Mail,
  MapPin,
  Phone,
  Shield,
  User,
} from 'lucide-react';

import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';
import FileUploader from '../../components/ui/FileUploader';
import { useNewAdminAuth } from '../../contexts/NewAdminAuthContext.jsx';
import { newAdminContactAPI } from '../../utils/newAdminApi';

const STEP_CONTACT = 0;
const STEP_CENTER = 1;
const STEP_CREDENTIALS = 2;

const steps = [
  {
    key: STEP_CONTACT,
    title: 'Contact Person Details',
    description: 'Tell us about the primary point of contact for this account',
  },
  {
    key: STEP_CENTER,
    title: 'Education Center Details',
    description: 'Share information about your institution and media assets',
  },
  {
    key: STEP_CREDENTIALS,
    title: 'INAI Credentials (Provided by Us)',
    description: 'Securely store the INAI portal credentials for your team',
  },
];

const DEFAULT_FORM = {
  firstName: '',
  lastName: '',
  address: '',
  designation: '',
  phoneNumber: '',
  dob: '',
  educationCenterName: '',
  inaiEmail: '',
  inaiPassword: '',
};

const AdminOnboarding = () => {
  const navigate = useNavigate();
  const { admin, updateAdmin } = useNewAdminAuth();

  const [step, setStep] = useState(STEP_CONTACT);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formValues, setFormValues] = useState(DEFAULT_FORM);
  const [imageFiles, setImageFiles] = useState([]);
  const [centerPhotos, setCenterPhotos] = useState([]);
  const [logoFiles, setLogoFiles] = useState([]);
  const [activityPhotos, setActivityPhotos] = useState([]);

  const isLastStep = useMemo(() => step === STEP_CREDENTIALS, [step]);

  useEffect(() => {
    if (!admin?.admin_id) {
      return;
    }

    if (admin?.contact_exists) {
      navigate('/admin/members', { replace: true });
      return;
    }

    const loadContact = async () => {
      setLoading(true);
      try {
        const response = await newAdminContactAPI.getContact(admin.admin_id);
        if (response?.status && response.contact) {
          const contact = response.contact;
          setFormValues((prev) => ({
            ...prev,
            firstName: contact.first_name || '',
            lastName: contact.last_name || '',
            address: contact.address || '',
            designation: contact.designation || '',
            phoneNumber: contact.phone_number || '',
            dob: contact.dob || '',
            educationCenterName: contact.education_center_name || '',
            inaiEmail: contact.inai_email || admin.email || '',
            inaiPassword: '',
          }));
        } else if (admin?.email) {
          setFormValues((prev) => ({
            ...prev,
            inaiEmail: admin.email,
          }));
        }
      } catch (error) {
        if (error.response?.status !== 404) {
          console.error('Failed to load contact details', error);
          toast.error(
            error.response?.data?.detail || 'Unable to load existing contact details.'
          );
        } else if (admin?.email) {
          setFormValues((prev) => ({
            ...prev,
            inaiEmail: admin.email,
          }));
        }
      } finally {
        setLoading(false);
      }
    };

    loadContact();
  }, [admin, navigate]);

  const handleInputChange = (field) => (event) => {
    const { value } = event.target;
    setFormValues((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const goNextStep = () => {
    if (step < STEP_CREDENTIALS) {
      setStep((prev) => prev + 1);
    }
  };

  const goPreviousStep = () => {
    if (step > STEP_CONTACT) {
      setStep((prev) => prev - 1);
    }
  };

  const validateCurrentStep = () => {
    if (step === STEP_CONTACT) {
      if (!formValues.firstName.trim()) {
        toast.error('First name is required.');
        return false;
      }
      if (!formValues.phoneNumber.trim() || !/^\d{10}$/.test(formValues.phoneNumber)) {
        toast.error('Enter a valid 10 digit phone number.');
        return false;
      }
      if (!formValues.designation.trim()) {
        toast.error('Designation is required.');
        return false;
      }
      if (!formValues.address.trim()) {
        toast.error('Address is required.');
        return false;
      }
      if (!formValues.dob.trim()) {
        toast.error('Date of birth is required.');
        return false;
      }
    }

    if (step === STEP_CENTER) {
      if (!formValues.educationCenterName.trim()) {
        toast.error('Education center name is required.');
        return false;
      }
    }

    if (step === STEP_CREDENTIALS) {
      if (!formValues.inaiEmail.trim()) {
        toast.error('INAI email is required.');
        return false;
      }
      if (!/^[\w-.]+@[\w-]+\.[a-z]{2,}$/i.test(formValues.inaiEmail)) {
        toast.error('Enter a valid INAI email address.');
        return false;
      }
      if (!formValues.inaiPassword.trim() || formValues.inaiPassword.length < 6) {
        toast.error('INAI password must be at least 6 characters.');
        return false;
      }
    }

    return true;
  };

  const handleNext = () => {
    if (!validateCurrentStep()) {
      return;
    }

    if (isLastStep) {
      void handleSubmit();
    } else {
      goNextStep();
    }
  };

  const handleSubmit = async () => {
    if (!admin?.admin_id) {
      toast.error('Admin information missing. Please log in again.');
      return;
    }

    setSubmitting(true);
    try {
      const payload = new FormData();
      payload.append('first_name', formValues.firstName.trim());
      payload.append('last_name', formValues.lastName.trim());
      payload.append('address', formValues.address.trim());
      payload.append('designation', formValues.designation.trim());
      payload.append('phone_number', formValues.phoneNumber.trim());
      payload.append('dob', formValues.dob);
      payload.append('education_center_name', formValues.educationCenterName.trim());
      payload.append('inai_email', formValues.inaiEmail.trim());
      payload.append('inai_password', formValues.inaiPassword);

      if (imageFiles.length > 0) {
        payload.append('image', imageFiles[0]);
      }

      centerPhotos.forEach((file) => {
        payload.append('center_photos', file);
      });

      if (logoFiles.length > 0) {
        payload.append('logo', logoFiles[0]);
      }

      activityPhotos.forEach((file) => {
        payload.append('other_activity', file);
      });

      const response = await newAdminContactAPI.upsertContact(payload);
      toast.success('Details saved successfully!');

      updateAdmin({
        contact_exists: true,
        last_contact_update: new Date().toISOString(),
      });

      navigate('/admin/members', { replace: true });
    } catch (error) {
      console.error('Onboarding submission failed', error);
      toast.error(error.response?.data?.detail || 'Failed to save contact details.');
    } finally {
      setSubmitting(false);
    }
  };

  const renderContactStep = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <InputField
          label="First Name"
          placeholder="Enter first name"
          icon={User}
          required
          value={formValues.firstName}
          onChange={handleInputChange('firstName')}
        />
        <InputField
          label="Last Name"
          placeholder="Enter last name"
          icon={User}
          value={formValues.lastName}
          onChange={handleInputChange('lastName')}
        />
        <InputField
          label="Phone Number"
          placeholder="10 digit phone number"
          icon={Phone}
          required
          value={formValues.phoneNumber}
          onChange={handleInputChange('phoneNumber')}
        />
        <InputField
          label="Designation"
          placeholder="Enter designation"
          icon={Shield}
          required
          value={formValues.designation}
          onChange={handleInputChange('designation')}
        />
        <InputField
          label="Date of Birth"
          type="date"
          icon={Calendar}
          required
          value={formValues.dob}
          onChange={handleInputChange('dob')}
        />
        <InputField
          label="INAI Email"
          type="email"
          icon={Mail}
          disabled
          value={formValues.inaiEmail}
          onChange={handleInputChange('inaiEmail')}
        />
      </div>
      <InputField
        label="Address (Current & Permanent)"
        placeholder="Enter full address"
        icon={MapPin}
        required
        value={formValues.address}
        onChange={handleInputChange('address')}
      />
    </div>
  );

  const renderCenterStep = () => (
    <div className="space-y-6">
      <InputField
        label="Name of Education Center"
        placeholder="Enter name of education center"
        icon={Building}
        required
        value={formValues.educationCenterName}
        onChange={handleInputChange('educationCenterName')}
      />
      <FileUploader
        label="Upload Image"
        description="Click to upload or drag and drop (max 5MB)"
        maxSize={5 * 1024 * 1024}
        multiple={false}
        onFilesChange={setImageFiles}
      />
      <FileUploader
        label="Education Center Photos"
        description="Upload supporting center photos (up to 5)"
        maxSize={5 * 1024 * 1024}
        multiple
        maxFiles={5}
        onFilesChange={setCenterPhotos}
      />
      <FileUploader
        label="Logo"
        description="Upload your institution logo"
        maxSize={5 * 1024 * 1024}
        multiple={false}
        onFilesChange={setLogoFiles}
      />
      <FileUploader
        label="Other Activities of Center"
        description="Optional media showing other activities (up to 5)"
        maxSize={5 * 1024 * 1024}
        multiple
        maxFiles={5}
        onFilesChange={setActivityPhotos}
      />
    </div>
  );

  const renderCredentialsStep = () => (
    <div className="space-y-6">
      <InputField
        label="INAI Email"
        type="email"
        placeholder="Enter INAI portal email"
        icon={Mail}
        required
        value={formValues.inaiEmail}
        onChange={handleInputChange('inaiEmail')}
      />
      <InputField
        label="INAI Password"
        type="password"
        placeholder="Enter INAI portal password"
        icon={Lock}
        required
        showPasswordToggle
        value={formValues.inaiPassword}
        onChange={handleInputChange('inaiPassword')}
      />
      <div className="p-4 rounded-lg bg-dark-card border border-dark-border text-sm text-dark-text-secondary">
        <p className="flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-primary-500" />
          Your INAI credentials are encrypted using industry-standard algorithms before storage.
        </p>
      </div>
    </div>
  );

  const renderStepContent = () => {
    switch (step) {
      case STEP_CONTACT:
        return renderContactStep();
      case STEP_CENTER:
        return renderCenterStep();
      case STEP_CREDENTIALS:
        return renderCredentialsStep();
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-[#050507] text-white">
      <div className="max-w-4xl mx-auto px-4 py-10 space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="medium"
              icon={ArrowLeft}
              onClick={() => navigate('/admin/login')}
            >
              Back to Login
            </Button>
            <div>
              <h1 className="text-3xl font-semibold">Complete Your Onboarding</h1>
              <p className="text-dark-text-muted">
                Finish these simple steps to unlock your admin dashboard.
              </p>
            </div>
          </div>
          <div className="w-14 h-14 bg-primary-600 rounded-full flex items-center justify-center">
            <Shield className="w-7 h-7" />
          </div>
        </div>

        <div className="bg-dark-card border border-dark-border rounded-2xl p-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
            <div>
              <p className="text-sm uppercase tracking-widest text-dark-text-muted">Step {step + 1} of 3</p>
              <h2 className="text-2xl font-semibold text-white mt-1">{steps[step].title}</h2>
              <p className="text-dark-text-muted mt-1">{steps[step].description}</p>
            </div>
            <div className="flex items-center gap-2">
              {steps.map((item, index) => (
                <div
                  key={item.key}
                  className={`h-2 rounded-full transition-all duration-300 ${
                    index <= step ? 'w-12 bg-primary-500' : 'w-8 bg-dark-border'
                  }`}
                />
              ))}
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25 }}
              className="mt-6"
            >
              {loading ? (
                <div className="py-20 text-center text-dark-text-muted">Loading details...</div>
              ) : (
                renderStepContent()
              )}
            </motion.div>
          </AnimatePresence>

          <div className="mt-10 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="text-sm text-dark-text-muted">
              Need help? Contact support at <span className="text-primary-400">support@inai.edu</span>
            </div>
            <div className="flex items-center gap-3">
              {step > STEP_CONTACT && (
                <Button
                  variant="secondary"
                  size="medium"
                  icon={ArrowLeft}
                  onClick={goPreviousStep}
                  disabled={submitting}
                >
                  Previous
                </Button>
              )}
              <Button
                variant="primary"
                size="medium"
                icon={isLastStep ? CheckCircle : ArrowRight}
                iconPosition="right"
                onClick={handleNext}
                loading={submitting}
                disabled={loading}
              >
                {isLastStep ? 'Save & Continue' : 'Next'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminOnboarding;

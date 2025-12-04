/**
 * Complete Onboarding Page - Single step admin onboarding
 * Collects contact person, education center, and INAI credentials in one form
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import InputField from '../../components/ui/InputField';
import Button from '../../components/ui/Button';
import { 
  User, 
  Mail, 
  Phone, 
  MapPin, 
  Calendar, 
  Briefcase, 
  Building, 
  Shield, 
  ArrowRight,
  Lock
} from 'lucide-react';
import { onboardingAPI } from '../../utils/api';
import toast from 'react-hot-toast';
import FileUploader from '../../components/ui/FileUploader';

const CompleteOnboardingPage = () => {
  const { updateToken, updateUser, user } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [uploadImageFile, setUploadImageFile] = useState([]);
  const [centerPhotosFiles, setCenterPhotosFiles] = useState([]);
  const [logoFile, setLogoFile] = useState([]);
  const [otherActivitiesFiles, setOtherActivitiesFiles] = useState([]);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
  } = useForm({
    defaultValues: {
      firstName: '',
      lastName: '',
      address: '',
      designation: '',
      phoneNumber: '',
      dateOfBirth: '',
      educationCenterName: '',
      inaiEmail: '',
      inaiPassword: '',
    },
  });

  const formValues = watch();

  useEffect(() => {
    if (user?.email) {
      setValue('inaiEmail', user.email);
    }
  }, [user?.email, setValue]);

  const convertFileToBase64 = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  }, []);

  const onSubmit = async (data) => {
    setIsLoading(true);
    
    try {
      const uploadImage = uploadImageFile.length ? await convertFileToBase64(uploadImageFile[0]) : null;
      const centerPhotos = centerPhotosFiles.length ? await Promise.all(centerPhotosFiles.map(convertFileToBase64)) : [];
      const logo = logoFile.length ? await convertFileToBase64(logoFile[0]) : null;
      const otherActivities = otherActivitiesFiles.length ? await Promise.all(otherActivitiesFiles.map(convertFileToBase64)) : [];

      const onboardingData = {
        first_name: data.firstName,
        last_name: data.lastName,
        address: data.address,
        designation: data.designation,
        phone_number: data.phoneNumber,
        date_of_birth: data.dateOfBirth,
        education_center_name: data.educationCenterName,
        inai_email: data.inaiEmail,
        inai_password: data.inaiPassword,
        upload_image: uploadImage,
        center_photos: centerPhotos,
        logo: logo,
        other_activities: otherActivities,
      };

      const formattedData = {
        ...onboardingData,
        date_of_birth: data.dateOfBirth
          ? (() => {
              const [year, month, day] = data.dateOfBirth.split('-');
              return `${day}-${month}-${year}`;
            })()
          : '',
      };

      const response = await onboardingAPI.completeOnboarding(formattedData);
      
      if (response.data.status) {
        toast.success('Onboarding completed successfully!');
        
        // Update token and user data
        const { new_token, admin_info } = response.data.data;
        updateToken(new_token);
        updateUser(admin_info);
        
        // Navigate to admin dashboard
        navigate('/dev_admin');
      }
    } catch (error) {
      console.error('Onboarding error:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to complete onboarding';
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
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
          
          <h1 className="text-3xl font-bold text-dark-text-primary mb-2">
            Complete Your Setup
          </h1>
          <p className="text-dark-text-secondary">
            Please provide your details to complete your INAI account setup.
          </p>
        </motion.div>

        {/* Form */}
        <motion.div
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
            
            {/* Contact Person Section */}
            <div>
              <h2 className="text-xl font-semibold text-dark-text-primary mb-6 flex items-center gap-2">
                <User className="w-5 h-5" />
                Contact Person Details
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <InputField
                  label="First Name"
                  placeholder="Enter first name"
                  icon={User}
                  error={errors.firstName?.message}
                  {...register('firstName', {
                    required: 'First name is required',
                    minLength: {
                      value: 2,
                      message: 'First name must be at least 2 characters',
                    },
                  })}
                  value={formValues.firstName}
                  onChange={(e) => setValue('firstName', e.target.value)}
                />

                <InputField
                  label="Last Name"
                  placeholder="Enter last name"
                  icon={User}
                  error={errors.lastName?.message}
                  {...register('lastName', {
                    required: 'Last name is required',
                    minLength: {
                      value: 2,
                      message: 'Last name must be at least 2 characters',
                    },
                  })}
                  value={formValues.lastName}
                  onChange={(e) => setValue('lastName', e.target.value)}
                />

                <InputField
                  label="Email Address"
                  type="email"
                  placeholder="Mirrors the INAI login email"
                  icon={Mail}
                  value={formValues.inaiEmail || ''}
                  disabled
                />

                <InputField
                  label="Phone Number"
                  placeholder="Enter phone number"
                  icon={Phone}
                  error={errors.phoneNumber?.message}
                  {...register('phoneNumber', {
                    required: 'Phone number is required',
                    pattern: {
                      value: /^\+?[\d\s\-()]+$/,
                      message: 'Invalid phone number format',
                    },
                  })}
                  value={formValues.phoneNumber}
                  onChange={(e) => setValue('phoneNumber', e.target.value)}
                />

                <InputField
                  label="Designation"
                  placeholder="Enter designation"
                  icon={Briefcase}
                  error={errors.designation?.message}
                  {...register('designation', {
                    required: 'Designation is required',
                  })}
                  value={formValues.designation}
                  onChange={(e) => setValue('designation', e.target.value)}
                />

                <InputField
                  label="Date of Birth"
                  type="date"
                  placeholder="Select date"
                  icon={Calendar}
                  error={errors.dateOfBirth?.message}
                  {...register('dateOfBirth', {
                    required: 'Date of birth is required',
                  })}
                  value={formValues.dateOfBirth}
                  onChange={(e) => setValue('dateOfBirth', e.target.value, { shouldValidate: true })}
                />
              </div>

              <div className="mt-6">
                <InputField
                  label="Address"
                  placeholder="Enter full address"
                  icon={MapPin}
                  error={errors.address?.message}
                  {...register('address', {
                    required: 'Address is required',
                  })}
                  value={formValues.address}
                  onChange={(e) => setValue('address', e.target.value)}
                />
              </div>
            </div>

            {/* Education Center Section */}
            <div>
              <h2 className="text-xl font-semibold text-dark-text-primary mb-6 flex items-center gap-2">
                <Building className="w-5 h-5" />
                Education Center
              </h2>
              
              <InputField
                label="Education Center Name"
                placeholder="Enter your institution name"
                icon={Building}
                error={errors.educationCenterName?.message}
                {...register('educationCenterName', {
                  required: 'Education center name is required',
                  minLength: {
                    value: 2,
                    message: 'Name must be at least 2 characters',
                  },
                })}
                value={formValues.educationCenterName}
                onChange={(e) => setValue('educationCenterName', e.target.value)}
              />

              <div className="mt-6 space-y-6">
                <FileUploader
                  label="Upload Image"
                  description="Click to upload or drag and drop (max 5MB)"
                  maxSize={5 * 1024 * 1024}
                  multiple={false}
                  onFilesChange={setUploadImageFile}
                />

                <FileUploader
                  label="Education Center Photos"
                  description="Upload up to 5 photos"
                  maxSize={5 * 1024 * 1024}
                  multiple
                  maxFiles={5}
                  onFilesChange={setCenterPhotosFiles}
                />

                <FileUploader
                  label="Logo"
                  description="Upload your institution logo"
                  maxSize={3 * 1024 * 1024}
                  multiple={false}
                  onFilesChange={setLogoFile}
                />

                <FileUploader
                  label="Other Activities"
                  description="Upload supporting images for activities (optional)"
                  maxSize={5 * 1024 * 1024}
                  multiple
                  maxFiles={5}
                  onFilesChange={setOtherActivitiesFiles}
                />
              </div>
            </div>

            {/* INAI Credentials Section */}
            <div>
              <h2 className="text-xl font-semibold text-dark-text-primary mb-6 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                INAI Portal Credentials
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <InputField
                  label="INAI Email"
                  type="email"
                  placeholder="Enter INAI portal email"
                  icon={Mail}
                  error={errors.inaiEmail?.message}
                  {...register('inaiEmail', {
                    required: 'INAI email is required',
                    pattern: {
                      value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                      message: 'Invalid email address',
                    },
                  })}
                  value={formValues.inaiEmail}
                  onChange={(e) => setValue('inaiEmail', e.target.value)}
                />

                <InputField
                  label="INAI Password"
                  type="password"
                  placeholder="Enter INAI portal password"
                  icon={Lock}
                  showPasswordToggle
                  error={errors.inaiPassword?.message}
                  {...register('inaiPassword', {
                    required: 'INAI password is required',
                    minLength: {
                      value: 6,
                      message: 'Password must be at least 6 characters',
                    },
                  })}
                  value={formValues.inaiPassword}
                  onChange={(e) => setValue('inaiPassword', e.target.value)}
                />
              </div>
            </div>

            {/* Submit Button */}
            <div className="pt-6">
              <Button
                type="submit"
                variant="primary"
                size="large"
                fullWidth
                loading={isLoading}
                icon={ArrowRight}
                iconPosition="right"
              >
                {isLoading ? 'Completing Setup...' : 'Complete Setup'}
              </Button>
            </div>
          </form>

          {/* Info Box */}
          <div className="mt-8 p-4 bg-blue-600/10 border border-blue-600/20 rounded-lg">
            <h4 className="font-medium text-blue-400 mb-2">Almost Done!</h4>
            <ul className="text-sm text-dark-text-secondary space-y-1">
              <li>• All information is securely encrypted and stored</li>
              <li>• You can update these details later from your dashboard</li>
              <li>• Files for your education center can be uploaded after setup</li>
              <li>• Contact support if you need any assistance</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default CompleteOnboardingPage;
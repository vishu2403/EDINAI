import { useState } from 'react';

const initialState = {
  firstName: '',
  lastName: '',
  educationCenterName: '',
  address: '',
  designation: '',
  phoneNumber: '',
  dob: '',
  inaiEmail: '',
  inaiPassword: '',
  uploadImage: [],
  centerPhotos: [],
  logo: [],
  otherActivities: [],
};

export const useContactForm = () => {
  const [formState, setFormState] = useState(initialState);

  const updateField = (field, value) => {
    setFormState((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const resetForm = () => setFormState(initialState);

  return {
    formState,
    updateField,
    resetForm,
  };
};

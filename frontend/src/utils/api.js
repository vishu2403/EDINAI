/**
 * API utility functions for INAI Education Management System
 * Handles HTTP requests with automatic token attachment and error handling
 */

import axios from 'axios';
import toast from 'react-hot-toast';

const API_TIMEOUT = Number(import.meta?.env?.VITE_API_TIMEOUT ?? 0);

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  timeout: Number.isFinite(API_TIMEOUT) ? API_TIMEOUT : 0,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const STUDENT_PORTAL_TOKEN_KEY = 'inai_student_token';

const getStudentPortalAuthHeaders = () => {
  if (typeof window === 'undefined') return {};
  const token = window.localStorage.getItem(STUDENT_PORTAL_TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const hasCustomAuthorization = Boolean(config?.headers?.Authorization);
    const requestUrl = config?.url || '';
    const isSchoolPortalRequest = requestUrl.startsWith('/school-portal/');
    const token = !isSchoolPortalRequest ? localStorage.getItem('inai_token') : null;

    if (!hasCustomAuthorization && token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    const message = error.response?.data?.message || error.message || 'An error occurred';

    // Handle specific error codes
    if (error.response?.status === 401) {
      const requestUrl = error.config?.url || '';
      const isSchoolPortalRequest = requestUrl.startsWith('/school-portal/');

      if (!isSchoolPortalRequest) {
        // Token expired or invalid for admin/member apps
        localStorage.removeItem('inai_token');
        localStorage.removeItem('inai_user');
        window.location.href = '/login';
      }

      return Promise.reject(error);
    }

    if (error.response?.status === 403) {
      toast.error('Access denied. Insufficient permissions.');
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.');
    } else if (!error.response) {
      toast.error('Network error. Please check your connection.');
    }

    return Promise.reject(error);
  }
);

// Authentication API
export const authAPI = {
  login: async (email, password) => {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  },

  changePassword: async (oldPassword, newPassword) => {
    const response = await api.post('/auth/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  forgotPassword: async (email) => {
    const response = await api.post('/auth/forgot-password', { email });
    return response.data;
  },

resetPassword: async (email, resetToken, newPassword) => {
    const response = await api.post('/auth/reset-password', {
      email,
      reset_token: resetToken,
      new_password: newPassword,
    });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// Chapter Materials API
export const chapterMaterialsAPI = {
  uploadMaterial: async (formData) => {
    const response = await api.post('/chapter-materials/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000,
    });
    return response.data;
  },

  listMaterials: async (filters = {}) => {
    const response = await api.post('/chapter-materials/chapter-suggestion', filters);
    return response.data;
  },

  updateMaterial: async (materialId, formData) => {
    const response = await api.put(`/chapter-materials/${materialId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000,
    });
    return response.data;
  },

  deleteMaterial: async (materialId) => {
    const response = await api.delete(`/chapter-materials/${materialId}`);
    return response.data;
  },

  // Extract topics from materials using AI
  extractTopics: async (materialIds) => {
    const response = await api.post('/chapter-materials/extract-topics', {
      material_ids: materialIds
    }, {
      timeout: 0, // No timeout - let it run as long as needed
    });
    return response.data;
  },

  // Create merged lecture from selected materials and topics
  createMergedLecture: async (lectureData) => {
    const response = await api.post('/chapter-materials/create-merged-lecture', lectureData);
    return response.data;
  },

  // Generate AI lecture from selected topics
  generateLectureFromTopics: async ({ materialId, topicIndices, options = {} }) => {
    const payload = {
      material_id: materialId,
      selected_topic_indices: topicIndices,
      ...options,
    };
    const response = await api.post('/chapter-materials/generate-lecture', payload, {
      timeout: 0,
    });
    return response.data;
  },

  editLectureSlide: async ({ lectureId, slideNumber, instruction }) => {
    const payload = {
      lecture_id: lectureId,
      slide_number: slideNumber,
      instruction,
    };
    const response = await api.post('/chapter-materials/edit-lecture-slide', payload, {
      timeout: 0,
    });
    return response.data;
  },
  recentTopics: async (limit = 10) => {
    const response = await api.get('/chapter-materials/recent-topics', {
      params: { limit },
      timeout: 0,
    });
    return response.data;
  },

  getMaterialTopics: async (materialId) => {
    const response = await api.get(`/chapter-materials/${materialId}/topics`, {
      timeout: 30000,
    });
    return response.data;
  },

  getRecentMaterials: async (limit = 5) => {
    const response = await api.get('/chapter-materials/recent', {
      params: { limit },
      timeout: 0,
    });
    return response.data;
  },

  addManualTopic: async (materialId, topicData) => {
    const response = await api.post(`/chapter-materials/${materialId}/topics`, topicData, {
      timeout: 0,
    });
    return response.data;
  },

  /**
   * Get AI-powered topic suggestions from the assistant chatbot
   * @param {number} materialId - The material ID to get suggestions for
   * @param {object} params - Query parameters
   * @param {string} params.user_query - User's question or request (required)
   * @param {number} params.temperature - AI creativity level (0-1, default 0.2)
   * @param {string} params.plan_label - User's plan: '20k', '50k', or '100k'
   * @returns {Promise<object>} Response with AI suggestions
   */
  assistantSuggestTopics: async (materialId, params) => {
    try {
      const response = await api.post(
        `/chapter-materials/${materialId}/assistant-suggest-topics`,
        params,
        {
          timeout: 0, // 60 seconds for AI generation
        },
      );
      return response.data;
    } catch (error) {
      console.error('Assistant suggest topics error:', error);
      throw error;
    }
  },

  /**
   * Add selected AI suggestions to material's topics
   * @param {number} materialId - The material ID to add topics to
   * @param {object} data - Request data
   * @param {Array<object>} data.selected_suggestions - Array of suggestion objects
   * Each suggestion should have: { title: string, summary: string, supporting_quote: string }
   * @returns {Promise<object>} Response with added topics info
   */
  assistantAddTopics: async (materialId, data) => {
    try {
      const response = await api.post(
        `/chapter-materials/${materialId}/assistant-add-topics`,
        data,
        {
          timeout: 0, // 30 seconds
        },
      );
      return response.data;
    } catch (error) {
      console.error('Assistant add topics error:', error);
      throw error;
    }
  },

  selectMultipleChapters: async (payload) => {
    const response = await api.post('/chapter-materials/select-multiple-chapters', payload);
    return response.data;
  },
};
export { api };

// Lecture Generation API
export const lectureAPI = {
  getConfig: async ({ language, duration } = {}) => {
    const payload = {};
    if (language) payload.language = language;
    if (duration) payload.duration = duration;

    const response = await api.post('/chapter-materials/chapter_lecture/config', payload);
    return response.data;
  },

  listGeneratedLectures: async (filters = {}) => {
    const params = {};
    if (filters.std) params.std = filters.std;
    if (filters.subject) params.subject = filters.subject;

    const response = await api.get('/chapter-materials/chapter_lectures', { params });
    return response.data;
  },

  getLectureOverview: async (filters = {}) => {
    const response = await api.get('/chapter-materials/chapter_lectures', { params: filters });
    return response.data;
  },

  generateLecture: async ({
    materialId,
    selectedTopicIndices,
    selectedTopicIds,
    language,
    duration,
    style,
    title,
  }) => {
    const payload = {
      material_id: materialId,
    };

    if (Array.isArray(selectedTopicIds) && selectedTopicIds.length) {
      payload.selected_topic_ids = selectedTopicIds;
    }
    if (Array.isArray(selectedTopicIndices) && selectedTopicIndices.length) {
      payload.selected_topic_indices = selectedTopicIndices;
    }

    if (language) payload.language = language;
    if (duration) payload.duration = duration;
    if (style) payload.style = style;
    if (title) payload.title = title;

    const response = await api.post('/chapter-materials/chapter_lecture/generate', payload, {
      timeout: 0, // allow long-running generation requests
    });
    return response.data;
  },

  chat: async ({ lectureId, question, answerType, isEditCommand = false, contextOverride }) => {
    const response = await api.post(
      `/chapter-materials/chapter_lecture/${lectureId}/chat`,
      {
        question,
        answer_type: answerType,
        is_edit_command: isEditCommand,
        context_override: contextOverride,
      },
      { timeout: 0 },
    );
    return response.data;
  },
};

// Admin Management API
export const adminAPI = {
  getAllAdmins: async (params = {}) => {
    const response = await api.get('/admin/', { params });
    return response.data;
  },

  getAdmin: async (adminId) => {
    const response = await api.get(`/admin/${adminId}`);
    return response.data;
  },

  createAdmin: async (adminData) => {
    const response = await api.post('/admin/add', adminData);
    return response.data;
  },

  updateAdmin: async (adminId, adminData) => {
    const response = await api.put(`/admin/${adminId}`, adminData);
    return response.data;
  },

  deleteAdmin: async (adminId) => {
    const response = await api.delete(`/admin/${adminId}`);
    return response.data;
  },

  extendSubscription: async (adminId, days) => {
    const response = await api.post(`/admin/${adminId}/extend-subscription`, { days });
    return response.data;
  },

  exportAdmins: async () => {
    const response = await api.get('/admin/export/csv');
    return response.data;
  },
};

// Member Management API
export const memberAPI = {
  getMembers: async (params = {}) => {
    const response = await api.get('/members/', { params });
    return response.data;
  },

  getMember: async (memberId) => {
    const response = await api.get(`/members/${memberId}`);
    return response.data;
  },

  createMember: async (memberData) => {
    const response = await api.post('/members/add', memberData);
    return response.data;
  },

  updateMember: async (memberId, memberData) => {
    const response = await api.put(`/members/${memberId}`, memberData);
    return response.data;
  },

  deleteMember: async (memberId) => {
    const response = await api.delete(`/members/${memberId}`);
    return response.data;
  },

  getMembersByWorkType: async (workType, activeOnly = true) => {
    const response = await api.get(`/members/by-work-type/${workType}`, {
      params: { active_only: activeOnly },
    });
    return response.data;
  },

  toggleMemberStatus: async (memberId) => {
    const response = await api.put(`/members/${memberId}/toggle-status`);
    return response.data;
  },

  exportMembers: async () => {
    const response = await api.get('/members/export/csv');
    return response.data;
  },
};

// Student Management API (Student Portal)
export const studentAPI = {
  createStudent: async (formValues) => {
    const formData = new FormData();

    Object.entries(formValues).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        formData.append(key, value);
      }
    });

    const response = await api.post('/studentm/create_student', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },
};

// Student portal (student-facing) API
export const schoolPortalAPI = {
  signup: async (enrollmentNumber, password) => {
    const response = await api.post('/school-portal/auth/signup', {
      enrolment_number: enrollmentNumber,
      password,
    });
    return response.data;
  },

  login: async (enrollmentNumber, password) => {
    const response = await api.post('/school-portal/auth/login', {
      enrolment_number: enrollmentNumber,
      password,
    });
    return response.data;
  },

  changePassword: async (currentPassword, newPassword) => {
    const response = await api.post(
      '/school-portal/auth/change-password',
      {
        current_password: currentPassword,
        new_password: newPassword,
      },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  saveProfile: async (formData) => {
    const response = await api.post('/school-portal/profile', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getProfile: async (enrollmentNumber) => {
    const response = await api.get(`/school-portal/profile/${encodeURIComponent(enrollmentNumber)}`);
    return response.data;
  },

  getProfileStatus: async (enrollmentNumber) => {
    const response = await api.get(
      `/school-portal/profile-status/${encodeURIComponent(enrollmentNumber)}`,
    );
    return response.data;
  },

  getChatPeers: async () => {
    const response = await api.get('/school-portal/chat/peers', {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getChatMessages: async (peerEnrollment) => {
    const response = await api.get(
      `/school-portal/chat/messages/${encodeURIComponent(peerEnrollment)}`,
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  sendChatMessage: async (peerEnrollment, message, shareMetadata) => {
    const response = await api.post(
      '/school-portal/chat/messages',
      { peer_enrollment: peerEnrollment, message, share_metadata: shareMetadata ?? undefined },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  sendChatAttachment: async (formData) => {
    const response = await api.post('/school-portal/chat/messages/attachment', formData, {
      headers: {
        ...getStudentPortalAuthHeaders(),
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  sendChatSignal: async (peerEnrollment, signalType, payload) => {
    const response = await api.post(
      '/school-portal/chat/signals',
      {
        peer_enrollment: peerEnrollment,
        signal_type: signalType,
        payload,
      },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  pollChatSignals: async (peerEnrollment) => {
    const response = await api.get(`/school-portal/chat/signals/${encodeURIComponent(peerEnrollment)}`, {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getBooks: async (subject) => {
    const response = await api.get('/school-portal/books', {
      params: subject ? { subject } : undefined,
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getDashboardVideos: async (limit = 6) => {
    const response = await api.get('/school-portal/dashboard/videos', {
      params: { limit },
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getSavedVideos: async () => {
    const response = await api.get('/school-portal/videos/saved', {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getVideoDetail: async (videoId) => {
    const response = await api.get(`/school-portal/videos/${videoId}`, {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getVideoComments: async (videoId) => {
    const response = await api.get(`/school-portal/videos/${videoId}/comments`, {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  getAllVideoComments: async () => {
    const response = await api.get('/school-portal/videos/comments', {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },

  addVideoComment: async (videoId, comment) => {
    const response = await api.post(
      `/school-portal/videos/${videoId}/comments`,
      { comment },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  setVideoLike: async (videoId, liked) => {
    const response = await api.post(
      `/school-portal/videos/${videoId}/like`,
      { liked },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  setVideoSubscribe: async (videoId, subscribed) => {
    const response = await api.post(
      `/school-portal/videos/${videoId}/subscribe`,
      { subscribed },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  recordVideoWatch: async (videoId, watchSeconds) => {
    const response = await api.post(
      `/school-portal/videos/${videoId}/watch`,
      { watch_seconds: watchSeconds },
      {
        headers: {
          ...getStudentPortalAuthHeaders(),
        },
      },
    );
    return response.data;
  },

  getWatchedLectures: async () => {
    const response = await api.get('/school-portal/watched-lectures', {
      headers: {
        ...getStudentPortalAuthHeaders(),
      },
    });
    return response.data;
  },
};

// Onboarding API
export const onboardingAPI = {
  completeOnboarding: async (onboardingData) => {
    const response = await api.post('/complete-onboarding/', onboardingData);
    return response.data;
  },

  // Contact
  createContact: async (contactData) => {
    const response = await api.post('/contact/', contactData);
    return response.data;
  },

  getContact: async () => {
    const response = await api.get('/contact/');
    return response.data;
  },

  updateContact: async (contactData) => {
    const response = await api.put('/contact/', contactData);
    return response.data;
  },

  // Education Center
  createEducationCenter: async (formData) => {
    const response = await api.post('/education-center/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getEducationCenter: async () => {
    const response = await api.get('/education-center/');
    return response.data;
  },

  updateEducationCenter: async (formData) => {
    const response = await api.put('/education-center/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // INAI Credentials
  createINAICredentials: async (credentialsData) => {
    const response = await api.post('/inai-credentials/', credentialsData);
    return response.data;
  },

  getINAICredentials: async () => {
    const response = await api.get('/inai-credentials/');
    return response.data;
  },

  updateINAICredentials: async (credentialsData) => {
    const response = await api.put('/inai-credentials/', credentialsData);
    return response.data;
  },

  // Post-onboarding credentials access
  viewINAICredentials: async () => {
    const response = await api.get('/inai-credentials/view');
    return response.data;
  },

  updateINAICredentialsPostOnboarding: async (credentialsData) => {
    const response = await api.put('/inai-credentials/update', credentialsData);
    return response.data;
  },
};

// Dashboard API
export const dashboardAPI = {
  getAdminDashboard: async () => {
    const response = await api.get('/dashboard/admin');
    return response.data;
  },

  getChapterDashboard: async () => {
    const response = await api.get('/chapter-materials/dashboard');
    return response.data;
  },

  getStudentDashboard: async () => {
    const response = await api.get('/dashboard/student');
    return response.data;
  },

  getLectureDashboard: async () => {
    const response = await api.get('/dashboard/lecture');
    return response.data;
  },

  getDashboardSummary: async () => {
    const response = await api.get('/dashboard/summary');
    return response.data;
  },
};

export const studentManagementAPI = {
  downloadTemplate: async () => {
    return api.get('/student-management/template', {
      responseType: 'blob',
    });
  },

  uploadRoster: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/student-management/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  fetchRoster: async () => {
    const response = await api.get('/student-management/roster');
    return response.data;
  },

  fetchStudentProfiles: async (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.classFilter) {
      params.append('class_filter', filters.classFilter);
    }
    if (filters.divisionFilter) {
      params.append('division_filter', filters.divisionFilter);
    }
    if (filters.enrollmentFilter) {
      params.append('enrollment_filter', filters.enrollmentFilter);
    }
    const query = params.toString();
    const response = await api.get(`/student-management/students${query ? `?${query}` : ''}`);
    return response.data;
  },

  fetchStudentFilters: async (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.classFilter) {
      params.append('class_filter', filters.classFilter);
    }
    const query = params.toString();
    const response = await api.get(`/student-management/students/filters${query ? `?${query}` : ''}`);
    return response.data;
  },

  deleteRosterStudent: async (enrollmentNumber) => {
    const response = await api.delete(`/student-management/roster/${enrollmentNumber}`);
    return response.data;
  },
};

// Utility functions
export const downloadCSV = (csvContent, filename) => {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const getFileUrl = (filePath) => {
  if (!filePath) return null;

  // Normalize the file path
  let normalizedPath = filePath.replace(/\\/g, '/').replace(/^\.?\/?/, '');

  // If the path already starts with 'uploads/', don't add another 'uploads/' prefix
  if (normalizedPath.startsWith('uploads/')) {
    return `/${normalizedPath}`;
  }

  // Otherwise, add the uploads prefix
  return `/uploads/${normalizedPath}`;
};

// File upload helper
export const uploadFile = async (file, endpoint, additionalData = {}) => {
  const formData = new FormData();
  formData.append('file', file);

  // Add additional form data
  Object.keys(additionalData).forEach((key) => {
    formData.append(key, additionalData[key]);
  });

  const response = await api.post(endpoint, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// PDF Processing API functions
export const processPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/pdf/process', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 120000, // 2 minutes for PDF processing
  });

  return response.data;
};

export const analyzeTextContent = async (text) => {
  const response = await api.post('/pdf/analyze-text', { text });
  return response.data;
};

export const getPDFProcessingCapabilities = async () => {
  const response = await api.get('/pdf/processing-status');
  return response.data;
};

// Generate PDF with selected topics
export const generateTopicPDF = async (materialId, selectedTopicIndices) => {
  const formData = new FormData();
  formData.append('material_id', materialId);
  formData.append('selected_topic_indices', JSON.stringify(selectedTopicIndices));

  const response = await api.post('/pdf/generate-topic-pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 1 minute for PDF generation
  });

  return response.data;
};

export default api;

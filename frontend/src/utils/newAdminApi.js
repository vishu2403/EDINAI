import axios from 'axios';

const NEW_ADMIN_API_BASE_URL =
  import.meta.env.VITE_NEW_ADMIN_API_BASE_URL || '/admin-api';

const adminApi = axios.create({
  baseURL: NEW_ADMIN_API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

adminApi.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('new_admin_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

adminApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('new_admin_token');
      localStorage.removeItem('new_admin_user');
      if (window.location.pathname !== '/admin/login') {
        window.location.href = '/admin/login';
      }
    }
    return Promise.reject(error);
  }
);

export const newAdminAuthAPI = {
  register: async (payload) => {
    const response = await adminApi.post('/registration/register', payload);
    return response.data;
  },
  login: async (email, password) => {
    const response = await adminApi.post('/auth/login', { email, password });
    return response.data;
  },
  changePassword: async ({ oldPassword, newPassword }) => {
    const response = await adminApi.post('/admin-portal/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },
  logout: async () => {
    return adminApi.post('/auth/logout');
  },
};

export const newAdminMemberAPI = {
  listMembers: async ({ adminId, workType = 'all' }) => {
    const params = {
      admin_id: adminId,
    };

    if (workType && workType !== 'all') {
      params.work_type = workType;
    }

    const response = await adminApi.get('/admin-portal/members', {
      params: {
        ...params,
      },
    });
    return response.data;
  },
  createMember: async ({ adminId, data }) => {
    const response = await adminApi.post(`/admin-portal/members/${adminId}`, data);
    return response.data;
  },
  updateMember: async ({ adminId, memberId, data }) => {
    const response = await adminApi.put(`/admin-portal/members/${adminId}/${memberId}`, data);
    return response.data;
  },
  deleteMember: async ({ adminId, memberId }) => {
    const response = await adminApi.delete(`/admin-portal/members/${adminId}/${memberId}`);
    return response.data;
  },
};

export const newAdminDashboardAPI = {
  getAdminDashboard: async () => {
    const response = await adminApi.get('/admin-portal/dashboard');
    return response.data;
  },
};

export const newAdminContactAPI = {
  getContact: async (adminId) => {
    const response = await adminApi.get(`/contact/${adminId}`);
    return response.data;
  },
  upsertContact: async (formData) => {
    const response = await adminApi.post('/contact/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default adminApi;
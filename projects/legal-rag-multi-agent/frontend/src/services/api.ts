import axios from 'axios';
import type { LoginRequest, LoginResponse, AnalysisRequest, AnalysisResponse } from '@/types';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: async (credentials: LoginRequest): Promise<LoginResponse> => {
    const { data } = await api.post<LoginResponse>('/api/v1/login', credentials);
    return data;
  },
  
  logout: () => {
    localStorage.removeItem('access_token');
  },
};

export const legalApi = {
  analyze: async (request: AnalysisRequest): Promise<AnalysisResponse> => {
    const { data } = await api.post<AnalysisResponse>('/api/v1/analyze', request);
    return data;
  },

  getModels: async () => {
    const { data } = await api.get('/api/v1/models');
    return data;
  },

  healthCheck: async () => {
    const { data } = await api.get('/health');
    return data;
  },
};

export default api;

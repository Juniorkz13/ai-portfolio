import axios from 'axios';
import type { LoginRequest, LoginResponse, AnalysisRequest, AnalysisResponse } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'test-key';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
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

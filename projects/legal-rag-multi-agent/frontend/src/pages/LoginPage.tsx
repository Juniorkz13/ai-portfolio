import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Scale, Lock, Mail } from 'lucide-react';
import toast from 'react-hot-toast';
import { AxiosError } from 'axios';
import { authApi } from '@/services/api';
import { useAuthStore } from '@/store/authStore';
import { Button } from '@/components/Common/Button';

type LoginError = {
  detail?: string;
};

export default function LoginPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);

    try {
      const response = await authApi.login({
        username: formData.username.trim(),
        password: formData.password,
      });

      if (!response?.access_token) {
        throw new Error('Token não recebido da API');
      }

      setAuth({ username: formData.username.trim() }, response.access_token);
      toast.success('Login realizado com sucesso!');
      navigate('/');
    } catch (err: unknown) {
      const axiosErr = err as AxiosError<LoginError>;
      const message =
        axiosErr.response?.data?.detail ||
        axiosErr.message ||
        'Falha no login';
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-4 bg-[#2f3540] border border-[#3e4451]">
            <Scale className="w-8 h-8 text-[var(--od-orange)]" />
          </div>
          <h1 className="text-3xl font-bold od-title">Legal RAG IA</h1>
          <p className="od-muted mt-2">Sistema Multiagente de Análise Jurídica</p>
        </div>

        <div className="od-card p-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm mb-2 od-muted">E-mail / Usuário</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#7f848e]" />
                <input
                  type="text"
                  required
                  value={formData.username}
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                  className="od-input pl-10"
                  placeholder="admin@example.com"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm mb-2 od-muted">Senha</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[#7f848e]" />
                <input
                  type="password"
                  required
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="od-input pl-10"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <Button type="submit" loading={loading} className="w-full">
              Entrar
            </Button>
          </form>

          <div className="mt-6 od-card-soft p-4">
            <p className="text-sm text-[var(--od-orange)] font-medium mb-2">Credenciais de teste</p>
            <p className="text-xs od-muted">E-mail: admin@example.com</p>
            <p className="text-xs od-muted">Senha: admin123</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

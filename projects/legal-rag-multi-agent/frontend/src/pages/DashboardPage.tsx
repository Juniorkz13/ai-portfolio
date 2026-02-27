import { useState } from 'react';
import { motion } from 'framer-motion';
import { Scale, LogOut, Send, History, Trash2, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

import { useAuthStore } from '@/store/authStore';
import { useHistoryStore } from '@/store/historyStore';
import { legalApi } from '@/services/api';
import { Button } from '@/components/Common/Button';
import type { AnalysisResponse } from '@/types';

export default function DashboardPage() {
  const { user, logout } = useAuthStore();
  const { items: history, addItem, clearHistory } = useHistoryStore();

  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [showHistory, setShowHistory] = useState(true);

  const getRiskBadgeClass = (level: string) => {
    const normalized = (level || '').toLowerCase();
    if (normalized === 'baixo') {
      return 'bg-[#2f4b3a] text-[#98c379] border border-[#3f5f49]';
    }
    if (normalized === 'médio' || normalized === 'medio') {
      return 'bg-[#4c452f] text-[#e5c07b] border border-[#665b3d]';
    }
    if (normalized === 'alto') {
      return 'bg-[#4a3036] text-[#e06c75] border border-[#644049]';
    }
    return 'bg-[#323846] text-[#abb2bf] border border-[#3e4451]';
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();

    const trimmed = question.trim();
    if (!trimmed) {
      toast.error('Digite uma pergunta para analisar.');
      return;
    }

    setLoading(true);
    setAnalysis(null);

    try {
      const result = await legalApi.analyze({ question: trimmed });
      setAnalysis(result);
      addItem(trimmed, result);
      toast.success('Análise concluída com sucesso.');
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || 'Falha ao analisar a pergunta.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    toast.success('Sessão encerrada.');
  };

  return (
    <div className="min-h-screen bg-[#1e2127] text-[#abb2bf]">
      <header className="sticky top-0 z-20 bg-[#232833]/90 backdrop-blur border-b border-[#3e4451]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <Scale className="w-8 h-8 text-[#d19a66]" />
              <div>
                <h1 className="text-xl font-bold text-[#e6edf3]">Legal RAG IA</h1>
                <p className="text-sm text-[#7f848e]">Bem-vindo, {user?.username}</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setShowHistory((v) => !v)}>
                <History className="w-4 h-4" />
                Histórico ({history.length})
              </Button>
              <Button variant="outline" size="sm" onClick={handleLogout}>
                <LogOut className="w-4 h-4" />
                Sair
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Coluna principal */}
          <section className="lg:col-span-2 space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-xl border border-[#3e4451] bg-[#282c34] p-6"
            >
              <h2 className="text-lg font-semibold text-[#e6edf3] mb-4">
                Faça sua pergunta jurídica
              </h2>

              {/* IMPORTANTE: submit dentro do form */}
              <form onSubmit={handleAnalyze} className="space-y-4">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Descreva sua situação jurídica e o que você deseja entender..."
                  className="w-full h-36 px-4 py-3 rounded-xl border border-[#3e4451] bg-[#20242c] text-[#e6edf3] placeholder:text-[#7f848e] focus:outline-none focus:ring-2 focus:ring-[#d19a66] focus:border-transparent resize-none"
                  required
                />
                <Button type="submit" loading={loading} className="w-full">
                  <Send className="w-4 h-4" />
                  Analisar pergunta
                </Button>
              </form>
            </motion.div>

            {analysis && (
              <motion.div
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl border border-[#3e4451] bg-[#282c34] p-6 space-y-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-semibold text-[#e6edf3]">Resultado da análise</h3>
                    <p className="text-sm text-[#7f848e] mt-1">{analysis.domain}</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskBadgeClass(analysis.risk_level)}`}>
                    Risco: {analysis.risk_level}
                  </span>
                </div>

                <div className="bg-[#232833] border-l-4 border-[#d19a66] rounded p-4">
                  <p className="text-sm font-medium text-[#e6edf3]">
                    {analysis.analysis.summary}
                  </p>
                </div>

                <div className="whitespace-pre-wrap text-[#c9d1d9] leading-relaxed">
                  {analysis.analysis.answer}
                </div>

                {analysis.analysis.recommendations?.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-[#e6edf3] mb-2">Recomendações</h4>
                    <ul className="list-disc pl-5 space-y-1 text-[#abb2bf]">
                      {analysis.analysis.recommendations.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="p-4 bg-[#3a3130] border border-[#5a4743] rounded-lg">
                  <div className="flex gap-2">
                    <AlertCircle className="w-5 h-5 text-[#e5c07b] flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-[#f1d7a3]">{analysis.analysis.disclaimer}</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-[#3e4451] text-sm text-[#8b93a3] flex flex-wrap gap-3">
                  <span>Tempo: {analysis.metadata.processing_time_ms} ms</span>
                  <span>•</span>
                  <span>Confiança: {analysis.analysis.confidence_score}</span>
                </div>
              </motion.div>
            )}
          </section>

          {/* Sidebar */}
          <aside className="space-y-6">
            {showHistory && (
              <motion.div
                initial={{ opacity: 0, x: 14 }}
                animate={{ opacity: 1, x: 0 }}
                className="rounded-xl border border-[#3e4451] bg-[#282c34] p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-[#e6edf3]">Histórico recente</h3>
                  {history.length > 0 && (
                    <button
                      onClick={() => {
                        clearHistory();
                        toast.success('Histórico limpo.');
                      }}
                      className="text-[#e06c75] hover:text-[#ff8b94] transition-colors"
                      title="Limpar histórico"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {history.length === 0 ? (
                  <p className="text-sm text-[#7f848e] text-center py-4">Sem histórico ainda.</p>
                ) : (
                  <div className="space-y-3">
                    {history.slice(0, 10).map((item) => (
                      <button
                        key={item.id}
                        onClick={() => {
                          setQuestion(item.question);
                          if (item.analysis) setAnalysis(item.analysis);
                        }}
                        className="w-full text-left p-3 rounded-lg border border-[#3e4451] bg-[#232833] hover:bg-[#2c313c] transition-colors"
                      >
                        <p className="text-sm font-medium text-[#e6edf3] line-clamp-2">
                          {item.question}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className="text-xs text-[#a9b0bd]">{item.domain}</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${getRiskBadgeClass(item.risk_level)}`}>
                            {item.risk_level}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </motion.div>
            )}

            <div className="rounded-xl border border-[#3e4451] bg-[#282c34] p-6 border-l-4 border-l-[#d19a66]">
              <h3 className="font-semibold text-[#e6edf3] mb-2">Como funciona</h3>
              <ul className="text-sm text-[#abb2bf] space-y-2">
                <li>• 6 agentes de IA analisam sua pergunta</li>
                <li>• O modelo gera orientação estruturada</li>
                <li>• A resposta inclui avaliação de risco</li>
                <li>• Consulte um advogado para decisão final</li>
              </ul>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

export interface User {
  username: string;
  email?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface AnalysisRequest {
  question: string;
  documents?: string[];
}

export interface AnalysisResponse {
  request_id: string;
  question: string;
  status: 'completed' | 'error';
  risk_level: 'baixo' | 'médio' | 'alto';
  domain: string;
  analysis: {
    answer: string;
    disclaimer: string;
    summary: string;
    documents_processed: number;
    queries_generated: number;
    has_conflicts: boolean;
    is_ambiguous: boolean;
    missing_info: string[];
    recommendations: string[];
    confidence_score: string;
  };
  agents_used: string[];
  metadata: {
    workflow_version: string;
    processing_time_ms: number;
    language: string;
  };
}

export interface HistoryItem {
  id: string;
  question: string;
  domain: string;
  risk_level: string;
  timestamp: Date;
  analysis?: AnalysisResponse;
}

export const LEGAL_DOMAINS = {
  Trabalhista: { icon: '🏢', color: 'blue', label: 'Labor Law' },
  Civil: { icon: '📋', color: 'gray', label: 'Civil Law' },
  Consumidor: { icon: '🛍️', color: 'green', label: 'Consumer Law' },
  Família: { icon: '👨‍👩‍👧', color: 'pink', label: 'Family Law' },
  Penal: { icon: '⚡', color: 'red', label: 'Criminal Law' },
  Tributário: { icon: '💰', color: 'yellow', label: 'Tax Law' },
  Empresarial: { icon: '🏭', color: 'purple', label: 'Business Law' },
  Previdenciário: { icon: '🏦', color: 'indigo', label: 'Social Security' },
  Imobiliário: { icon: '🏠', color: 'orange', label: 'Real Estate Law' },
} as const;

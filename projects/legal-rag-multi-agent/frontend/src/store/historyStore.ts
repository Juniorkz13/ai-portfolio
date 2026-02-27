import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { HistoryItem, AnalysisResponse } from '@/types';

interface HistoryState {
  items: HistoryItem[];
  addItem: (question: string, analysis: AnalysisResponse) => void;
  clearHistory: () => void;
  getItem: (id: string) => HistoryItem | undefined;
}

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set, get) => ({
      items: [],
      
      addItem: (question, analysis) => {
        const newItem: HistoryItem = {
          id: analysis.request_id,
          question,
          domain: analysis.domain,
          risk_level: analysis.risk_level,
          timestamp: new Date(),
          analysis,
        };
        
        set((state) => ({
          items: [newItem, ...state.items].slice(0, 50),
        }));
      },
      
      clearHistory: () => set({ items: [] }),
      
      getItem: (id) => get().items.find((item) => item.id === id),
    }),
    {
      name: 'history-storage',
    }
  )
);

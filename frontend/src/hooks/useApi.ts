import { useMutation, useQuery } from '@tanstack/react-query';
import * as api from '../api/endpoints';
import type { ChatRequest, PatientInput, XAIMethod } from '../api/types';

export function useDemoMode() {
  return useQuery({
    queryKey: ['demoMode'],
    queryFn: api.getDemoMode,
    retry: false,
    staleTime: 60_000,
  });
}

export function useModelInfo() {
  return useQuery({
    queryKey: ['modelInfo'],
    queryFn: api.getModelInfo,
    retry: 1,
    staleTime: Infinity,
  });
}

export function usePredictAll() {
  return useMutation({
    mutationFn: (patient: PatientInput) => api.predictAll(patient),
  });
}

export function useSurvival(patient: PatientInput) {
  return useQuery({
    queryKey: ['survival', patient],
    queryFn: () => api.predictSurvival(patient),
    staleTime: Infinity,
    retry: 0,
  });
}

type ExplainMethod = XAIMethod | 'comparison';

const EXPLAINERS = {
  shap: api.explainShap,
  lime: api.explainLime,
  dalex: api.explainDalex,
  ebm: api.explainEbm,
  comparison: api.explainComparison,
} as const;

type ExplainResult<M extends ExplainMethod> = Awaited<ReturnType<(typeof EXPLAINERS)[M]>>;

/**
 * Explanation for a patient, cached per (method, model, patient): switching XAI tabs
 * or models back and forth does not repeat slow requests.
 */
export function useExplanation<M extends ExplainMethod>(method: M, patient: PatientInput, modelKey = 'xgboost') {
  return useQuery<ExplainResult<M>>({
    queryKey: ['explain', method, modelKey, patient],
    queryFn: () =>
      EXPLAINERS[method]({
        patient,
        method: method === 'comparison' ? 'shap' : method,
        num_features: 20,
        model_key: modelKey,
      }) as Promise<ExplainResult<M>>,
    staleTime: Infinity,
    retry: 0,
  });
}

export function useChat() {
  return useMutation({
    mutationFn: (req: ChatRequest) => api.chat(req),
  });
}

export function useAgentChat() {
  return useMutation({
    mutationFn: (req: api.AgentChatPayload) => api.agentChat(req),
  });
}

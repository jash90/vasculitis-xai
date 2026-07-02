import type { PatientInput, PredictionOutput, RiskLevel } from '../api/types';

function getRiskLevel(probability: number): RiskLevel {
  if (probability < 0.3) return 'low';
  if (probability < 0.7) return 'moderate';
  return 'high';
}

export function getDemoPrediction(patient: PatientInput): PredictionOutput {
  let riskScore = 0;

  riskScore += patient.liczba_zajetych_narzadow * 0.1;
  if (patient.manifestacja_nerki) riskScore += 0.15;
  if (patient.zaostrz_wymagajace_oit) riskScore += 0.25;
  if (patient.zaostrz_wymagajace_hospital) riskScore += 0.1;
  if (patient.plazmaferezy) riskScore += 0.1;

  const probability = Math.min(Math.max(riskScore, 0.05), 0.95);

  return {
    probability,
    risk_level: getRiskLevel(probability),
    prediction: probability > 0.5 ? 1 : 0,
  };
}

export interface DemoFactor {
  feature: string;
  contribution: number;
}

export function getDemoExplanation(patient: PatientInput): {
  risk_factors: DemoFactor[];
  protective_factors: DemoFactor[];
  base_value: number;
} {
  const risk_factors: DemoFactor[] = [];
  const protective_factors: DemoFactor[] = [];

  if (patient.manifestacja_nerki) {
    risk_factors.push({ feature: 'Zajecie nerek', contribution: 0.12 });
  }
  if (patient.zaostrz_wymagajace_oit) {
    risk_factors.push({ feature: 'Zaostrzenia OIT', contribution: 0.2 });
  }
  if (patient.liczba_zajetych_narzadow <= 2) {
    protective_factors.push({ feature: 'Liczba narzadow', contribution: -0.08 });
  }

  return { risk_factors, protective_factors, base_value: 0.15 };
}

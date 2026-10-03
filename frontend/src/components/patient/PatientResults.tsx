import { GaugeChart } from '../charts/GaugeChart';
import { RiskBadge } from '../common/RiskBadge';
import { QueryState } from '../common/QueryState';
import { FactorsList } from './FactorsList';
import { SurvivalPanel } from './SurvivalPanel';
import { useExplanation } from '../../hooks/useApi';
import { featureWithValue } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { MultiModelPredictionOutput, PatientInput } from '../../api/types';

interface PatientResultsProps {
  patient: PatientInput;
  result: MultiModelPredictionOutput;
}

export function PatientResults({ patient, result }: PatientResultsProps) {
  // Same query key as the SHAP tab (xgboost) -> computed once, reused below.
  const shap = useExplanation('shap', patient, 'xgboost');
  const factors = shap.data
    ? [...shap.data.risk_factors, ...shap.data.protective_factors].map((f) => ({
        feature: featureWithValue(f.feature, f.value),
        contribution: f.contribution,
      }))
    : [];

  return (
    <div className="space-y-8">
      <section aria-labelledby="models-title">
        <h3 id="models-title" className="mb-4 text-lg font-semibold text-blue-300">{pl.results.models}</h3>
        <div className="grid gap-4 md:grid-cols-3">
          {result.models.map((m) => (
            <div key={m.model_name} className="rounded-xl border border-gray-700/60 bg-gray-800/30 p-4 text-center">
              <p className="text-sm font-semibold text-gray-300">{m.model_name}</p>
              <GaugeChart probability={m.probability} height={190} />
              <div className="flex justify-center">
                <RiskBadge level={m.risk_level} compact />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-col items-center gap-2 rounded-xl border border-blue-500/30 bg-blue-900/10 p-4 sm:flex-row sm:justify-center sm:gap-6">
          <div className="text-center">
            <p className="text-sm text-gray-400">{pl.results.average}</p>
            <p className="text-3xl font-bold text-blue-200 tabular-nums">{(result.ensemble_probability * 100).toFixed(1)}%</p>
          </div>
          <RiskBadge level={result.ensemble_risk_level} />
        </div>
      </section>

      <SurvivalPanel patient={patient} />

      <section aria-labelledby="factors-title">
        <h3 id="factors-title" className="mb-3 text-lg font-semibold text-gray-200">{pl.results.keyFactors}</h3>
        <QueryState isLoading={shap.isLoading} error={shap.error} onRetry={() => shap.refetch()}
          loadingText="Obliczanie wkładu cech (SHAP)…" minHeight={120}>
          <FactorsList factors={factors} />
        </QueryState>
      </section>
    </div>
  );
}

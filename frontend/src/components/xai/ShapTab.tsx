import { useState } from 'react';
import { useExplanation } from '../../hooks/useApi';
import { ContributionChart } from '../charts/ContributionChart';
import { QueryState } from '../common/QueryState';
import { ModelSelector } from './ModelSelector';
import { featureWithValue } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

export function ShapTab({ patient }: { patient: PatientInput }) {
  const [modelKey, setModelKey] = useState('xgboost');
  const query = useExplanation('shap', patient, modelKey);
  const factors = query.data
    ? query.data.feature_contributions.map((f) => ({ feature: featureWithValue(f.feature, f.value), contribution: f.contribution }))
    : [];

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.shapTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">{pl.xai.shapDesc}</p>
      <ModelSelector value={modelKey} onChange={setModelKey} />
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Obliczanie wartości SHAP…">
        {query.data && (
          <>
            <p className="mb-2 text-xs text-gray-500">
              Średnie ryzyko w danych (punkt odniesienia): {(query.data.base_value * 100).toFixed(1)}%.
            </p>
            <ContributionChart factors={factors} xTitle="Wkład w prawdopodobieństwo zgonu" />
          </>
        )}
      </QueryState>
    </div>
  );
}

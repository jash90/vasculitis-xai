import { useState } from 'react';
import { useExplanation } from '../../hooks/useApi';
import { ContributionChart } from '../charts/ContributionChart';
import { QueryState } from '../common/QueryState';
import { ModelSelector } from './ModelSelector';
import { prettyCondition } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

export function LimeTab({ patient }: { patient: PatientInput }) {
  const [modelKey, setModelKey] = useState('xgboost');
  const query = useExplanation('lime', patient, modelKey);
  const factors = (query.data?.feature_weights ?? []).map((f) => ({
    feature: prettyCondition(String(f.condition ?? f.feature ?? '')),
    contribution: Number(f.weight ?? 0),
  }));

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.limeTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">{pl.xai.limeDesc}</p>
      <ModelSelector value={modelKey} onChange={setModelKey} />
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Dopasowywanie lokalnego modelu LIME…">
        {query.data && (
          <>
            <p className="mb-2 text-xs text-gray-500">
              Predykcja modelu: {(query.data.prediction.probability * 100).toFixed(1)}% · predykcja lokalnego modelu
              LIME: {(query.data.local_prediction * 100).toFixed(1)}%.
            </p>
            <ContributionChart factors={factors} xTitle="Waga warunku w modelu lokalnym" />
          </>
        )}
      </QueryState>
    </div>
  );
}

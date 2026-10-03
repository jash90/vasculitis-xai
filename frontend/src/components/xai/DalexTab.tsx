import { useState } from 'react';
import { useExplanation } from '../../hooks/useApi';
import { ContributionChart } from '../charts/ContributionChart';
import { ImportanceChart } from '../charts/ImportanceChart';
import { QueryState } from '../common/QueryState';
import { ModelSelector } from './ModelSelector';
import { featureLabel } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

export function DalexTab({ patient }: { patient: PatientInput }) {
  const [modelKey, setModelKey] = useState('xgboost');
  const query = useExplanation('dalex', patient, modelKey);
  const data = query.data;
  const breakdown = data
    ? [...data.risk_factors, ...data.protective_factors].map((f) => ({ feature: featureLabel(f.feature), contribution: f.contribution }))
    : [];
  const vi = data?.variable_importance
    ? Object.entries(data.variable_importance).map(([feature, contribution]) => ({ feature: featureLabel(feature), contribution }))
    : [];

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.dalexTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">{pl.xai.dalexDesc}</p>
      <ModelSelector value={modelKey} onChange={setModelKey} />
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Obliczanie rozkładu predykcji DALEX…">
        {data && (
          <div className="space-y-6">
            <div>
              <h4 className="mb-1 text-sm font-semibold text-gray-300">Rozkład predykcji na cechy</h4>
              <p className="mb-2 text-xs text-gray-500">
                Średnia w danych: {(data.intercept * 100).toFixed(1)}% → predykcja dla pacjenta: {(data.prediction * 100).toFixed(1)}%.
              </p>
              <ContributionChart factors={breakdown} />
            </div>
            {vi.length > 0 && (
              <div>
                <h4 className="mb-1 text-sm font-semibold text-gray-300">Globalna ważność cech (permutacja)</h4>
                <p className="mb-2 text-xs text-gray-500">Spadek jakości modelu (1 − AUC) po losowym przemieszaniu cechy.</p>
                <ImportanceChart items={vi} xTitle="Wzrost straty (1 − AUC)" />
              </div>
            )}
          </div>
        )}
      </QueryState>
    </div>
  );
}

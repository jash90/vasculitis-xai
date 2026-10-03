import { useState } from 'react';
import { useExplanation } from '../../hooks/useApi';
import { QueryState } from '../common/QueryState';
import { ModelSelector } from './ModelSelector';
import { featureLabel } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

const TOP = 5;

function Ranking({ title, items, common }: { title: string; items: string[]; common: Set<string> }) {
  return (
    <div className="rounded-lg bg-gray-800/40 p-4">
      <h4 className="mb-2 font-medium text-gray-300">{title}</h4>
      <ol className="list-inside list-decimal space-y-1 text-sm">
        {items.slice(0, TOP).map((f) => (
          <li key={f} className={common.has(f) ? 'font-medium text-blue-200' : 'text-gray-400'}>
            {featureLabel(f)}
          </li>
        ))}
      </ol>
    </div>
  );
}

export function ComparisonTab({ patient }: { patient: PatientInput }) {
  const [modelKey, setModelKey] = useState('xgboost');
  const query = useExplanation('comparison', patient, modelKey);
  const data = query.data;
  const common = new Set(data?.common_top_features ?? []);
  const rho = data?.spearman_correlations?.SHAP_vs_LIME;

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.compTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">
        Porównanie {TOP} najważniejszych cech według obu metod; wyróżnione cechy występują w obu rankingach.
      </p>
      <ModelSelector value={modelKey} onChange={setModelKey} />
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Porównywanie wyjaśnień SHAP i LIME…">
        {data && (
          <>
            <div className="grid gap-4 md:grid-cols-2">
              <Ranking title="Ranking SHAP" items={data.individual_rankings.SHAP ?? []} common={common} />
              <Ranking title="Ranking LIME" items={data.individual_rankings.LIME ?? []} common={common} />
            </div>
            <dl className="mt-4 grid gap-3 rounded-lg bg-blue-900/20 p-4 text-sm text-blue-200 sm:grid-cols-2">
              <div>
                <dt className="text-xs text-blue-300/80">Wspólne cechy w top {TOP} (Jaccard)</dt>
                <dd className="text-lg font-semibold">{(data.ranking_agreement * 100).toFixed(0)}%</dd>
              </div>
              {rho !== undefined && (
                <div>
                  <dt className="text-xs text-blue-300/80">Korelacja rang Spearmana (wszystkie cechy)</dt>
                  <dd className="text-lg font-semibold">{rho.toFixed(2)}</dd>
                </div>
              )}
            </dl>
            <p className="mt-3 text-xs text-gray-500">{pl.xai.compInfo}</p>
          </>
        )}
      </QueryState>
    </div>
  );
}

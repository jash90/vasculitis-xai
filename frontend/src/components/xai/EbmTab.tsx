import { useExplanation, useModelInfo } from '../../hooks/useApi';
import { ContributionChart } from '../charts/ContributionChart';
import { ImportanceChart } from '../charts/ImportanceChart';
import { QueryState } from '../common/QueryState';
import { RiskBadge } from '../common/RiskBadge';
import { featureLabel, labelInText } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

export function EbmTab({ patient }: { patient: PatientInput }) {
  const query = useExplanation('ebm', patient, 'ebm');
  const info = useModelInfo();
  const data = query.data;
  const nTrain = info.data?.details?.n_train;

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.ebmTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">
        {pl.xai.ebmDesc}
        {nTrain ? ` Dane treningowe: ${nTrain} pacjentów, 20 cech.` : ''}
      </p>
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Obliczanie wyjaśnienia EBM (pierwsze wywołanie trenuje model i może potrwać kilkanaście sekund)…">
        {data && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center gap-4 rounded-lg border border-gray-600 bg-gray-800/50 p-4">
              <div>
                <p className="text-sm text-gray-400">Predykcja EBM</p>
                <p className="text-2xl font-bold text-white tabular-nums">{(data.probability * 100).toFixed(1)}%</p>
              </div>
              <RiskBadge level={data.risk_level} />
            </div>
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-300">Lokalny wkład cech (skala logitu)</h4>
              <ContributionChart
                factors={data.local_contributions.map((f) => ({ feature: featureLabel(f.feature), contribution: f.contribution }))}
                xTitle="Wkład (logit)"
              />
            </div>
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-300">Globalna ważność cech</h4>
              <ImportanceChart
                items={Object.entries(data.global_importance).map(([f, v]) => ({ feature: labelInText(f), contribution: v }))}
                xTitle="Średni bezwzględny wkład"
              />
            </div>
            {data.interactions.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-semibold text-gray-300">Wykryte interakcje cech</h4>
                <ul className="flex flex-wrap gap-2">
                  {data.interactions.map((x) => (
                    <li key={x} className="rounded-full bg-gray-700 px-3 py-1 text-xs text-gray-300">{labelInText(x)}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </QueryState>
    </div>
  );
}

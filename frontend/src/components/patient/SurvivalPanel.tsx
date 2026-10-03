import { useSurvival } from '../../hooks/useApi';
import { QueryState } from '../common/QueryState';
import { SurvivalCurveChart } from '../charts/SurvivalCurveChart';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

function RiskCard({ years, risk }: { years: number; risk: number }) {
  const color = risk < 0.3 ? 'text-green-300' : risk < 0.7 ? 'text-yellow-300' : 'text-red-300';
  return (
    <div className="rounded-lg bg-gray-800/60 p-3 text-center">
      <p className="text-xs text-gray-400">{years === 1 ? 'Po 1 roku' : `Po ${years} latach`}</p>
      <p className={`mt-1 text-2xl font-bold tabular-nums ${color}`}>{(risk * 100).toFixed(1)}%</p>
    </div>
  );
}

export function SurvivalPanel({ patient }: { patient: PatientInput }) {
  const query = useSurvival(patient);
  const m = query.data?.metrics;

  return (
    <section aria-labelledby="survival-title" className="rounded-xl border border-gray-700/60 bg-gray-800/30 p-5">
      <h3 id="survival-title" className="text-lg font-semibold text-blue-300">{pl.results.survivalTitle}</h3>
      <p className="mb-4 mt-1 text-sm text-gray-400">{pl.results.survivalDesc}</p>
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()}
        loadingText="Obliczanie ryzyka w czasie…" minHeight={160}>
        {query.data && (
          <>
            <div className="grid grid-cols-3 gap-3">
              <RiskCard years={1} risk={query.data.risk_1y} />
              <RiskCard years={3} risk={query.data.risk_3y} />
              <RiskCard years={5} risk={query.data.risk_5y} />
            </div>
            <div className="mt-4">
              <SurvivalCurveChart curve={query.data.survival_curve} />
            </div>
            {m && (
              <p className="mt-2 text-xs text-gray-500">
                Jakość modelu: C-index {m.cv_harrell_c?.toFixed(2)} (walidacja krzyżowa)
                {m.holdout_harrell_c != null && <>, {m.holdout_harrell_c.toFixed(2)} (zbiór testowy)</>}
                {m.holdout_auc_5y != null && <>; AUC dla 5 lat: {m.holdout_auc_5y.toFixed(2)}</>}.
              </p>
            )}
          </>
        )}
      </QueryState>
    </section>
  );
}

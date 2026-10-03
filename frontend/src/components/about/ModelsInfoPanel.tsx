import { useModelInfo } from '../../hooks/useApi';
import { QueryState } from '../common/QueryState';
import type { ClassifierInfo, ModelsMetadata } from '../../api/types';

const pct = (v?: number) => (v === undefined || v === null ? '—' : v.toFixed(2));

function ClassifierRow({ info }: { info: ClassifierInfo }) {
  return (
    <tr className="text-gray-300">
      <th scope="row" className="px-3 py-2 text-left font-medium text-gray-200">{info.label}</th>
      <td className="px-3 py-2 tabular-nums">
        {pct(info.cv.roc_auc)} <span className="text-xs text-gray-500">± {pct(info.cv.roc_auc_sd)}</span>
      </td>
      <td className="px-3 py-2 tabular-nums">
        {pct(info.holdout.roc_auc)}{' '}
        <span className="text-xs text-gray-500">({pct(info.holdout.roc_auc_ci[0])}–{pct(info.holdout.roc_auc_ci[1])})</span>
      </td>
      <td className="px-3 py-2 tabular-nums">{pct(info.holdout.pr_auc)}</td>
      <td className="px-3 py-2 tabular-nums">{info.holdout.brier.toFixed(3)}</td>
    </tr>
  );
}

function Details({ meta }: { meta: ModelsMetadata }) {
  const s = meta.survival;
  const loco = meta.classifiers.xgboost?.leave_one_centre_out_auc;
  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-400">
        {meta.task}. Dane: {meta.n_patients} pacjentów ({meta.n_deaths} zgonów); trening na {meta.n_train}, ocena końcowa na
        odłożonym zbiorze testowym ({meta.n_holdout}). Model trenowany: {meta.trained_at}.
      </p>

      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-sm">
          <caption className="bg-gray-800 px-3 py-2 text-left text-sm font-semibold text-gray-200">
            Klasyfikatory (20 cech z chwili rozpoznania)
          </caption>
          <thead className="bg-gray-800/60 text-xs text-gray-400">
            <tr>
              <th scope="col" className="px-3 py-2 text-left">Model</th>
              <th scope="col" className="px-3 py-2 text-left">AUC — walidacja krzyżowa</th>
              <th scope="col" className="px-3 py-2 text-left">AUC — zbiór testowy (95% CI)</th>
              <th scope="col" className="px-3 py-2 text-left">PR-AUC</th>
              <th scope="col" className="px-3 py-2 text-left">Brier</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {Object.entries(meta.classifiers).map(([k, v]) => <ClassifierRow key={k} info={v} />)}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {s && (
          <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-4 text-sm text-gray-300">
            <h4 className="mb-2 font-semibold text-gray-200">Model przeżycia ({s.model})</h4>
            <ul className="space-y-1">
              <li>C-index: {pct(s.cv.harrell_c_mean)} (walidacja krzyżowa), {pct(s.holdout.harrell_c)} (test)</li>
              <li>AUC dla horyzontu 1 / 3 / 5 lat (test): {pct(s.holdout.auc_1y)} / {pct(s.holdout.auc_3y)} / {pct(s.holdout.auc_5y)}</li>
              <li>Mediana czasu obserwacji: {s.median_follow_up_years.toFixed(1)} roku</li>
            </ul>
          </div>
        )}
        {loco !== undefined && (
          <div className="rounded-lg border border-gray-700 bg-gray-800/30 p-4 text-sm text-gray-300">
            <h4 className="mb-2 font-semibold text-gray-200">Przenoszalność między ośrodkami</h4>
            <p>
              Walidacja „leave-one-centre-out” (trening bez danego ośrodka, test na nim): AUC {pct(loco)}. To realistyczna
              oczekiwana jakość w nowym ośrodku.
            </p>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-yellow-600/30 bg-yellow-900/10 p-4 text-sm text-yellow-100">
        <h4 className="mb-2 font-semibold">Ograniczenia</h4>
        <ul className="list-inside list-disc space-y-1 text-yellow-100/90">
          {meta.limitations.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </div>
    </div>
  );
}

export function ModelsInfoPanel() {
  const query = useModelInfo();
  return (
    <section aria-labelledby="about-title" className="mx-auto max-w-4xl">
      <h2 id="about-title" className="mb-4 text-2xl font-bold text-blue-300">O modelach</h2>
      <QueryState isLoading={query.isLoading} error={query.error} onRetry={() => query.refetch()} loadingText="Wczytywanie metryk modeli…">
        {query.data?.details ? (
          <Details meta={query.data.details} />
        ) : (
          <p className="text-sm text-gray-400">Serwer działa w trybie demonstracyjnym — metryki modeli nie są dostępne.</p>
        )}
      </QueryState>
    </section>
  );
}

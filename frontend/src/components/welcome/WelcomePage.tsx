import { useModelInfo } from '../../hooks/useApi';
import { pl } from '../../i18n/pl';

export function WelcomePage({ onShowModels }: { onShowModels: () => void }) {
  const w = pl.welcome;
  const info = useModelInfo().data?.details;
  const best = info ? Object.values(info.classifiers).map((c) => c.holdout.roc_auc) : [];
  const c = info?.survival?.holdout.harrell_c;

  return (
    <div className="mx-auto max-w-3xl space-y-6 text-center">
      <div>
        <h2 className="mb-2 text-2xl font-bold text-blue-300">{w.title}</h2>
        <p className="text-gray-400">{w.subtitle}</p>
      </div>
      <ul className="grid gap-3 sm:grid-cols-3">
        {w.features.map((f) => (
          <li key={f.title} className="rounded-xl border border-gray-700/60 bg-gray-800/40 p-4 text-sm">
            <strong className="block text-blue-200">{f.title}</strong>
            <span className="text-gray-400">{f.desc}</span>
          </li>
        ))}
      </ul>
      {info && (
        <p className="text-xs text-gray-500">
          Jakość na odłożonym zbiorze testowym: AUC {Math.min(...best).toFixed(2)}–{Math.max(...best).toFixed(2)}
          {c !== undefined && <>, model przeżycia C-index {c.toFixed(2)}</>}.{' '}
          <button type="button" onClick={onShowModels} className="text-blue-400 underline hover:text-blue-300">
            Szczegóły i ograniczenia
          </button>
        </p>
      )}
    </div>
  );
}

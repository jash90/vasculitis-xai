import { featureLabel, type Contribution } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';

export function FactorsList({ factors, limit = 6 }: { factors: Contribution[]; limit?: number }) {
  const top = [...factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, limit);
  if (!top.length) return <p className="text-sm text-gray-500">Brak czynników do wyświetlenia.</p>;
  const maxAbs = Math.max(...top.map((f) => Math.abs(f.contribution)), 1e-6);

  return (
    <ul className="space-y-2">
      {top.map((f) => {
        const isRisk = f.contribution > 0;
        return (
          <li key={f.feature} className="rounded-lg bg-gray-800/50 px-3 py-2 text-sm">
            <div className="flex items-center gap-2">
              <span aria-hidden className={isRisk ? 'text-red-400' : 'text-green-400'}>{isRisk ? '▲' : '▼'}</span>
              <span className="flex-1 font-medium text-gray-200">{featureLabel(f.feature)}</span>
              <span className={`text-xs ${isRisk ? 'text-red-300' : 'text-green-300'}`}>
                {isRisk ? pl.results.increases : pl.results.decreases} ({f.contribution >= 0 ? '+' : ''}
                {(f.contribution * 100).toFixed(1)} p.p.)
              </span>
            </div>
            <div className="mt-1.5 h-1 rounded bg-gray-700">
              <div
                className={`h-1 rounded ${isRisk ? 'bg-red-500' : 'bg-green-500'}`}
                style={{ width: `${(Math.abs(f.contribution) / maxAbs) * 100}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

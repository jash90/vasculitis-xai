import { useState, useMemo } from 'react';
import type { BatchResultRow, RiskLevel } from '../../api/types';
import { RiskPill } from '../common/RiskBadge';
import { pl } from '../../i18n/pl';

const PAGE_SIZE = 50;
const LEVELS: { id: RiskLevel; label: string }[] = [
  { id: 'low', label: 'Niskie' },
  { id: 'moderate', label: 'Umiarkowane' },
  { id: 'high', label: 'Wysokie' },
];

export function ResultsTable({ results }: { results: BatchResultRow[] }) {
  const [riskFilter, setRiskFilter] = useState<RiskLevel[]>(['low', 'moderate', 'high']);
  const [sortBy, setSortBy] = useState<'probability' | 'wiek_rozpoznania' | 'patient_id'>('probability');
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const f = results.filter((r) => riskFilter.includes(r.risk_level));
    return [...f].sort((a, b) => {
      if (sortBy === 'patient_id') return a.patient_id.localeCompare(b.patient_id, 'pl', { numeric: true });
      if (sortBy === 'probability') return b.probability - a.probability;
      return (b.wiek_rozpoznania || 0) - (a.wiek_rozpoznania || 0);
    });
  }, [results, riskFilter, sortBy]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const visible = filtered.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  const toggle = (level: RiskLevel) => {
    setPage(0);
    setRiskFilter((prev) => (prev.includes(level) ? prev.filter((l) => l !== level) : [...prev, level]));
  };

  return (
    <section aria-labelledby="results-table-title">
      <h2 id="results-table-title" className="mb-4 text-xl font-bold text-gray-200">{pl.batch.detailedResults}</h2>

      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div role="group" aria-label={pl.batch.filterByRisk} className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{pl.batch.filterByRisk}:</span>
          {LEVELS.map((l) => (
            <button key={l.id} type="button" aria-pressed={riskFilter.includes(l.id)} onClick={() => toggle(l.id)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                riskFilter.includes(l.id) ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'
              }`}>
              {l.label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-400">
          {pl.batch.sortBy}:
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            className="rounded-md border border-gray-600 bg-gray-700 px-3 py-1 text-sm text-white">
            <option value="probability">Ryzyko (malejąco)</option>
            <option value="wiek_rozpoznania">Wiek w chwili rozpoznania</option>
            <option value="patient_id">ID pacjenta</option>
          </select>
        </label>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Wyniki analizy dla poszczególnych pacjentów</caption>
          <thead className="bg-gray-800 text-xs uppercase text-gray-400">
            <tr>
              <th scope="col" className="px-4 py-3">ID</th>
              <th scope="col" className="px-4 py-3">Wiek</th>
              <th scope="col" className="px-4 py-3">Narządy</th>
              <th scope="col" className="px-4 py-3">Ryzyko</th>
              <th scope="col" className="px-4 py-3">Poziom</th>
              <th scope="col" className="px-4 py-3">{pl.batch.topFactors}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {visible.map((r, i) => (
              <tr key={`${r.patient_id}-${i}`} className="text-gray-300 hover:bg-gray-800/50">
                <td className="px-4 py-2 font-medium">{r.patient_id}</td>
                <td className="px-4 py-2">{Number.isFinite(r.wiek_rozpoznania) ? r.wiek_rozpoznania : '—'}</td>
                <td className="px-4 py-2">{r.liczba_narzadow}</td>
                <td className="px-4 py-2 tabular-nums">{r.probability_pct}</td>
                <td className="px-4 py-2"><RiskPill level={r.risk_level} /></td>
                <td className="px-4 py-2 text-xs text-gray-400">{r.top_factors || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-gray-400">
        <p>Wyświetlono {visible.length} z {filtered.length} pacjentów (łącznie {results.length}).</p>
        {pages > 1 && (
          <nav aria-label="Strony wyników" className="flex items-center gap-2">
            <button type="button" disabled={current === 0} onClick={() => setPage(current - 1)}
              className="rounded border border-gray-600 px-2 py-1 disabled:opacity-40">‹ Poprzednia</button>
            <span>Strona {current + 1} z {pages}</span>
            <button type="button" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}
              className="rounded border border-gray-600 px-2 py-1 disabled:opacity-40">Następna ›</button>
          </nav>
        )}
      </div>
    </section>
  );
}

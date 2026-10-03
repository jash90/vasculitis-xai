import { useBatchAnalysis } from '../../hooks/useBatchAnalysis';
import { FileUpload } from './FileUpload';
import { BatchSummary } from './BatchSummary';
import { BatchCharts } from './BatchCharts';
import { ResultsTable } from './ResultsTable';
import { ExportButtons } from './ExportButtons';
import { BatchProgress } from './BatchProgress';
import { sampleCsv } from '../../lib/sampleData';
import { downloadBlob } from '../../lib/exportUtils';
import { featureLabel } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';

type Batch = ReturnType<typeof useBatchAnalysis>;

function ParseReportBox({ report }: { report: NonNullable<Batch['report']> }) {
  const issues = report.missingColumns.length + report.ignoredColumns.length + report.invalidValues.length;
  if (!issues) return null;
  return (
    <div role="status" className="rounded-lg border border-yellow-600/40 bg-yellow-900/15 p-4 text-sm text-yellow-100">
      <p className="font-semibold">Uwagi do pliku</p>
      <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-yellow-200/90">
        {report.missingColumns.length > 0 && (
          <li>Brak kolumn (potraktowane jako brak danych): {report.missingColumns.map(featureLabel).join(', ')}.</li>
        )}
        {report.ignoredColumns.length > 0 && <li>Pominięte kolumny (nieużywane przez model): {report.ignoredColumns.join(', ')}.</li>}
        {report.invalidValues.length > 0 && (
          <li>
            Nieczytelne wartości ({report.invalidValues.length}) potraktowano jako brak danych, np.{' '}
            {report.invalidValues.slice(0, 3).map((v) => `wiersz ${v.row}, ${featureLabel(v.column)}: „${v.value}”`).join('; ')}.
          </li>
        )}
      </ul>
    </div>
  );
}

export function BatchView({ batch }: { batch: Batch }) {
  if (batch.isProcessing) {
    return (
      <div className="mx-auto max-w-3xl">
        <BatchProgress progress={batch.progress} />
      </div>
    );
  }

  if (batch.results) {
    return (
      <div className="space-y-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-gray-400">
            Plik: <strong className="text-gray-200">{batch.fileName}</strong>
            {batch.mode === 'demo' && <span className="ml-2 rounded bg-yellow-600 px-2 py-0.5 text-xs font-bold text-white">DEMO</span>}
          </p>
          <button type="button" onClick={batch.reset}
            className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-700">
            {pl.batch.newFile}
          </button>
        </div>
        {batch.mode === 'demo' && (
          <div role="status" className="rounded-lg bg-yellow-900/20 p-3 text-sm text-yellow-200">
            Serwer działa w trybie demonstracyjnym (model nie jest załadowany) — wyniki są poglądowe.
          </div>
        )}
        {batch.report && <ParseReportBox report={batch.report} />}
        <BatchSummary results={batch.results} />
        <BatchCharts results={batch.results} />
        <ResultsTable results={batch.results} />
        <ExportButtons results={batch.results} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      {batch.error && (
        <div role="alert" className="rounded-xl border border-red-500/40 bg-red-900/15 p-4 text-sm text-red-200">
          <p>{batch.error}</p>
          {batch.canRetry && (
            <button type="button" onClick={batch.retry}
              className="mt-3 rounded-md border border-red-400/50 px-3 py-1 text-xs font-medium hover:bg-red-900/30">
              {pl.common.retry}
            </button>
          )}
        </div>
      )}
      {batch.report && <ParseReportBox report={batch.report} />}

      <div className="rounded-xl border border-gray-700 bg-gray-800/40 p-8 text-center">
        <h2 className="mb-2 text-xl font-bold text-blue-300">{pl.batch.title}</h2>
        <p className="mb-6 text-sm text-gray-400">{pl.batch.subtitle}</p>
        <div className="mx-auto max-w-md">
          <FileUpload onFileSelect={batch.processFile} />
        </div>
        <button type="button" onClick={() => downloadBlob(sampleCsv(), 'przykladowi_pacjenci.csv', 'text/csv;charset=utf-8')}
          className="mt-4 text-sm text-blue-400 underline decoration-blue-400/30 transition hover:text-blue-300">
          {pl.batch.downloadSample}
        </button>
      </div>

      <section aria-labelledby="formats-title">
        <h3 id="formats-title" className="mb-4 text-lg font-semibold text-gray-200">{pl.batch.fileFormats}</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-gray-700 bg-gray-800/30 p-5 text-sm text-gray-400">
            <h4 className="mb-2 font-medium text-gray-200">CSV</h4>
            <ul className="list-inside list-disc space-y-1">
              <li>Pierwszy wiersz: nazwy kolumn (jak w pliku przykładowym)</li>
              <li>Separator: przecinek, średnik, tabulator lub „|”</li>
              <li>Tak/nie: 1/0, tak/nie lub yes/no; puste pole = brak danych</li>
            </ul>
          </div>
          <div className="rounded-xl border border-gray-700 bg-gray-800/30 p-5 text-sm text-gray-400">
            <h4 className="mb-2 font-medium text-gray-200">JSON</h4>
            <ul className="list-inside list-disc space-y-1">
              <li>Tablica obiektów: {'[{...}, {...}]'}</li>
              <li>lub obiekt z kluczem „patients” albo „data”</li>
              <li>Kodowanie: UTF-8</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

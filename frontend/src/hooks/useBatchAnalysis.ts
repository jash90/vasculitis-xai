import { useState, useCallback } from 'react';
import type { BatchResultRow, PatientInput } from '../api/types';
import { parseFile, type ParseReport, type ParsedPatient } from '../lib/fileParser';
import { predictBatch } from '../api/endpoints';
import { RISK_LEVEL_PL } from '../lib/columnMapping';
import { featureLabel } from '../lib/featureLabels';
import { errorMessage } from '../lib/errors';

const CHUNK_SIZE = 1000;

function withoutId(p: ParsedPatient): PatientInput {
  const copy: Partial<ParsedPatient> = { ...p };
  delete copy.patient_id;
  return copy as PatientInput;
}

export function useBatchAnalysis() {
  const [results, setResults] = useState<BatchResultRow[] | null>(null);
  const [report, setReport] = useState<Omit<ParseReport, 'patients'> | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [mode, setMode] = useState('');
  const [pending, setPending] = useState<ParsedPatient[] | null>(null);

  const run = useCallback(async (patients: ParsedPatient[]) => {
    setIsProcessing(true);
    setProgress(0);
    setError(null);
    const rows: BatchResultRow[] = [];
    try {
      for (let i = 0; i < patients.length; i += CHUNK_SIZE) {
        const chunk = patients.slice(i, i + CHUNK_SIZE);
        const res = await predictBatch({
          patients: chunk.map(withoutId),
          include_risk_factors: true,
          top_n_factors: 3,
        });
        setMode(res.mode);
        res.results.forEach((item, j) => {
          const p = chunk[j];
          rows.push({
            patient_id: p.patient_id,
            wiek_rozpoznania: p.wiek_rozpoznania ?? NaN,
            liczba_narzadow: p.liczba_zajetych_narzadow,
            probability: item.prediction.probability,
            probability_pct: `${(item.prediction.probability * 100).toFixed(1)}%`,
            risk_level: item.prediction.risk_level,
            risk_level_pl: RISK_LEVEL_PL[item.prediction.risk_level] ?? '',
            prediction: item.prediction.prediction,
            top_factors: (item.top_risk_factors ?? [])
              .map((f) => `${featureLabel(f.feature)} ${f.direction === 'increases_risk' ? '↑' : '↓'}`)
              .join(', '),
            processing_mode: res.mode,
          });
        });
        setProgress(Math.min(1, (i + chunk.length) / patients.length));
      }
      setResults(rows);
      setPending(null);
    } catch (err) {
      // No silent fallback to fake predictions: show the error and allow a retry.
      setError(`Analiza przerwana po ${rows.length} z ${patients.length} pacjentów. ${errorMessage(err)}`);
      setPending(patients);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const processFile = useCallback(
    async (file: File) => {
      setFileName(file.name);
      setResults(null);
      setError(null);
      try {
        const parsed = await parseFile(file);
        const { patients, ...rest } = parsed;
        setReport(rest);
        if (!patients.length) {
          setError('Plik nie zawiera żadnych pacjentów.');
          return;
        }
        await run(patients);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Błąd przetwarzania pliku.');
      }
    },
    [run],
  );

  const retry = useCallback(() => {
    if (pending) void run(pending);
  }, [pending, run]);

  const reset = useCallback(() => {
    setResults(null);
    setReport(null);
    setProgress(0);
    setError(null);
    setFileName('');
    setMode('');
    setPending(null);
  }, []);

  return { results, report, isProcessing, progress, error, fileName, mode, canRetry: pending !== null, processFile, retry, reset };
}

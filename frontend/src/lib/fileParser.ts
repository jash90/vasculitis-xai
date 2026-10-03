import Papa from 'papaparse';
import { BINARY_COLUMNS, COLUMN_MAPPING, PATIENT_COLUMNS } from './columnMapping';
import type { PatientInput } from '../api/types';

export interface ParsedPatient extends PatientInput {
  patient_id: string;
}

export interface ParseReport {
  patients: ParsedPatient[];
  /** Model columns absent from the file (treated as unknown -> training median). */
  missingColumns: string[];
  /** File columns that are not used by the model. */
  ignoredColumns: string[];
  /** Cells with values that could not be interpreted (treated as unknown). */
  invalidValues: { row: number; column: string; value: string }[];
}

const TRUE_VALUES = new Set(['tak', 'yes', 'true', '1', 't', 'y']);
const FALSE_VALUES = new Set(['nie', 'no', 'false', '0', 'n', 'f']);

function mapKey(key: string): string {
  const normalized = key.toLowerCase().trim().replace(/\s+/g, '_');
  return COLUMN_MAPPING[normalized] ?? normalized;
}

function parseRows(rows: Record<string, unknown>[]): ParseReport {
  const fileColumns = rows.length ? Object.keys(rows[0]) : [];
  const mapped = new Map(fileColumns.map((c) => [c, mapKey(c)]));
  const present = new Set(mapped.values());
  const known = new Set<string>([...PATIENT_COLUMNS, 'patient_id']);
  const invalidValues: ParseReport['invalidValues'] = [];
  const usedIds = new Set<string>();

  const patients = rows.map((raw, index) => {
    const row: Record<string, unknown> = {};
    for (const [orig, key] of mapped) row[key] = raw[orig];

    const value = (column: string): number | null => {
      const v = row[column];
      if (v === undefined || v === null || String(v).trim() === '') return null;
      if (typeof v === 'number') return Number.isFinite(v) ? v : null;
      const s = String(v).trim().toLowerCase().replace(',', '.');
      if (BINARY_COLUMNS.has(column)) {
        if (TRUE_VALUES.has(s)) return 1;
        if (FALSE_VALUES.has(s)) return 0;
      }
      const n = Number(s);
      if (Number.isFinite(n)) return n;
      invalidValues.push({ row: index + 2, column, value: String(v) });
      return null;
    };

    const patient = Object.fromEntries(
      PATIENT_COLUMNS.map((c) => {
        const v = value(c);
        if (BINARY_COLUMNS.has(c)) return [c, v === null ? null : v ? 1 : 0];
        return [c, v];
      }),
    ) as unknown as PatientInput;

    if (patient.liczba_zajetych_narzadow === null || patient.liczba_zajetych_narzadow === undefined) {
      patient.liczba_zajetych_narzadow = PATIENT_COLUMNS.filter(
        (c) => c.startsWith('manifestacja_') && patient[c as keyof PatientInput] === 1,
      ).length;
    }

    let id = String(row.patient_id ?? '').trim() || `P${String(index + 1).padStart(4, '0')}`;
    if (usedIds.has(id)) id = `${id} (${index + 1})`;
    usedIds.add(id);
    return { ...patient, patient_id: id };
  });

  return {
    patients,
    missingColumns: PATIENT_COLUMNS.filter((c) => !present.has(c) && c !== 'liczba_zajetych_narzadow'),
    ignoredColumns: fileColumns.filter((c) => !known.has(mapped.get(c) ?? '')),
    invalidValues,
  };
}

export function parseCSV(text: string): ParseReport {
  const result = Papa.parse<Record<string, unknown>>(text.replace(/^\uFEFF/, ''), {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
    delimitersToGuess: [',', ';', '\t', '|'],
  });
  return parseRows(result.data);
}

export function parseJSON(text: string): ParseReport {
  const data = JSON.parse(text);
  const rows: Record<string, unknown>[] = Array.isArray(data) ? data : data.patients ?? data.data ?? [data];
  return parseRows(rows);
}

export function parseFile(file: File): Promise<ParseReport> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      try {
        resolve(file.name.toLowerCase().endsWith('.json') ? parseJSON(text) : parseCSV(text));
      } catch {
        reject(new Error('Nie udało się odczytać pliku — sprawdź, czy ma poprawny format CSV lub JSON.'));
      }
    };
    reader.onerror = () => reject(new Error('Nie udało się odczytać pliku.'));
    reader.readAsText(file, 'utf-8');
  });
}

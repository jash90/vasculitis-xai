import Papa from 'papaparse';
import { COLUMN_MAPPING, DEFAULT_VALUES } from './columnMapping';
import type { PatientInput } from '../api/types';

export interface ParsedPatient extends PatientInput {
  patient_id: string;
}

function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    const normalizedKey = key.toLowerCase().trim().replace(/\s+/g, '_');
    const mappedKey = COLUMN_MAPPING[normalizedKey] ?? normalizedKey;
    normalized[mappedKey] = value;
  }
  return normalized;
}

function parseBool(value: unknown): number {
  const s = String(value).toLowerCase();
  return ['tak', 'yes', 'true', '1', 't', 'y'].includes(s) ? 1 : 0;
}

function toPatient(row: Record<string, unknown>, index: number): ParsedPatient {
  const get = (key: string): number => {
    const val = row[key];
    if (val === undefined || val === null || val === '') {
      return DEFAULT_VALUES[key] ?? 0;
    }
    const binaryFields = new Set(Object.entries(DEFAULT_VALUES).filter(([, v]) => v === 0 || v === 1).map(([k]) => k));
    if (binaryFields.has(key) && typeof val === 'string') return parseBool(val);
    return Number(val) || (DEFAULT_VALUES[key] ?? 0);
  };

  return {
    patient_id: String(row['patient_id'] ?? `P${String(index + 1).padStart(4, '0')}`),
    wiek_rozpoznania: get('wiek_rozpoznania'),
    opoznienie_rozpoznia: get('opoznienie_rozpoznia'),
    manifestacja_miesno_szkiel: get('manifestacja_miesno_szkiel'),
    manifestacja_skora: get('manifestacja_skora'),
    manifestacja_wzrok: get('manifestacja_wzrok'),
    manifestacja_sercowo_naczyniowy: get('manifestacja_sercowo_naczyniowy'),
    manifestacja_pokarmowy: get('manifestacja_pokarmowy'),
    manifestacja_nerki: get('manifestacja_nerki'),
    manifestacja_moczowo_plciowy: get('manifestacja_moczowo_plciowy'),
    manifestacja_zajecie_csn: get('manifestacja_zajecie_csn'),
    manifestacja_neurologiczny: get('manifestacja_neurologiczny'),
    liczba_zajetych_narzadow: get('liczba_zajetych_narzadow'),
    zaostrz_wymagajace_hospital: get('zaostrz_wymagajace_hospital'),
    zaostrz_wymagajace_oit: get('zaostrz_wymagajace_oit'),
    kreatynina: get('kreatynina'),
    eozynofilia_krwi_obwodowej_wartosc: get('eozynofilia_krwi_obwodowej_wartosc'),
    pulsy: get('pulsy'),
    czas_sterydow: get('czas_sterydow'),
    plazmaferezy: get('plazmaferezy'),
    biopsja_wynik: get('biopsja_wynik'),
  };
}

export function parseCSV(text: string): ParsedPatient[] {
  for (const delimiter of [',', ';', '\t', '|']) {
    const result = Papa.parse<Record<string, unknown>>(text, {
      header: true,
      delimiter,
      skipEmptyLines: true,
      dynamicTyping: true,
    });
    if (result.data.length > 0 && Object.keys(result.data[0]).length > 1) {
      return result.data.map((row, i) => toPatient(normalizeRow(row), i));
    }
  }
  const result = Papa.parse<Record<string, unknown>>(text, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });
  return result.data.map((row, i) => toPatient(normalizeRow(row), i));
}

export function parseJSON(text: string): ParsedPatient[] {
  const data = JSON.parse(text);
  let rows: Record<string, unknown>[];

  if (Array.isArray(data)) {
    rows = data;
  } else if (data.patients) {
    rows = data.patients;
  } else if (data.data) {
    rows = data.data;
  } else {
    rows = [data];
  }

  return rows.map((row, i) => toPatient(normalizeRow(row), i));
}

export function parseFile(file: File): Promise<ParsedPatient[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      try {
        if (file.name.toLowerCase().endsWith('.json')) {
          resolve(parseJSON(text));
        } else {
          resolve(parseCSV(text));
        }
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error('Nie udalo sie odczytac pliku'));
    reader.readAsText(file, 'utf-8');
  });
}

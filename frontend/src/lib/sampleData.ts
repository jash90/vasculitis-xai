import { PATIENT_COLUMNS } from './columnMapping';

type Row = Record<(typeof PATIENT_COLUMNS)[number], number | ''>;

const base: Row = Object.fromEntries(PATIENT_COLUMNS.map((c) => [c, 0])) as Row;

const ROWS: [string, Partial<Row>][] = [
  ['P001', { wiek_rozpoznania: 72, opoznienie_rozpoznia: 2, manifestacja_nerki: 1, manifestacja_oddechowy: 1, kreatynina: 420, max_crp: 150, eozynofilia_krwi_obwodowej_wartosc: 80, pulsy: 1, plazmaferezy: 1 }],
  ['P002', { wiek_rozpoznania: 45, opoznienie_rozpoznia: 6, manifestacja_nos_ucho_gardlo: 1, manifestacja_miesno_szkiel: 1, kreatynina: 85, max_crp: 25, eozynofilia_krwi_obwodowej_wartosc: 210, biopsja_wynik: 1 }],
  ['P003', { wiek_rozpoznania: 66, opoznienie_rozpoznia: 4, manifestacja_nerki: 1, manifestacja_skora: 1, kreatynina: 230, max_crp: 90, eozynofilia_krwi_obwodowej_wartosc: 120, pulsy: 1, biopsja_wynik: 1 }],
  ['P004', { wiek_rozpoznania: 34, opoznienie_rozpoznia: 12, manifestacja_nos_ucho_gardlo: 1, manifestacja_wzrok: 1, kreatynina: 70, max_crp: 12, eozynofilia_krwi_obwodowej_wartosc: 300, biopsja_wynik: 1 }],
  ['P005', { wiek_rozpoznania: 58, opoznienie_rozpoznia: 3, manifestacja_nerki: 1, manifestacja_neurologiczny: 1, kreatynina: 160, max_crp: '', eozynofilia_krwi_obwodowej_wartosc: '', pulsy: 1 }],
  ['P006', { wiek_rozpoznania: 79, opoznienie_rozpoznia: 1, manifestacja_nerki: 1, manifestacja_oddechowy: 1, manifestacja_sercowo_naczyniowy: 1, kreatynina: 610, max_crp: 210, eozynofilia_krwi_obwodowej_wartosc: 40, pulsy: 1, plazmaferezy: 1 }],
];

/** Sample CSV containing exactly the columns the model uses (empty cell = unknown). */
export function sampleCsv(): string {
  const header = ['id', ...PATIENT_COLUMNS].join(',');
  const lines = ROWS.map(([id, values]) => {
    const row: Row = { ...base, ...values };
    const organs = PATIENT_COLUMNS.filter((c) => c.startsWith('manifestacja_') && row[c] === 1).length;
    row.liczba_zajetych_narzadow = organs;
    return [id, ...PATIENT_COLUMNS.map((c) => row[c])].join(',');
  });
  return [header, ...lines].join('\n');
}

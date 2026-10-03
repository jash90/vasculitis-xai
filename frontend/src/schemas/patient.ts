import { z } from 'zod';
import type { PatientInput } from '../api/types';

const optionalNumber = (min: number, max: number, unit: string) =>
  z
    .number({ message: 'Podaj liczbę' })
    .min(min, `Wartość musi wynosić co najmniej ${min} ${unit}`.trim())
    .max(max, `Wartość nie może przekraczać ${max} ${unit}`.trim())
    .optional();

const flag = z.union([z.literal(0), z.literal(1)]);

export const BINARY_FIELDS = [
  'manifestacja_nerki',
  'manifestacja_oddechowy',
  'manifestacja_nos_ucho_gardlo',
  'manifestacja_sercowo_naczyniowy',
  'manifestacja_pokarmowy',
  'manifestacja_zajecie_csn',
  'manifestacja_neurologiczny',
  'manifestacja_miesno_szkiel',
  'manifestacja_skora',
  'manifestacja_wzrok',
  'manifestacja_moczowo_plciowy',
  'pulsy',
  'plazmaferezy',
  'biopsja_wynik',
] as const;

export const MANIFESTATION_FIELDS = BINARY_FIELDS.filter((f) => f.startsWith('manifestacja_'));

export const patientSchema = z.object({
  wiek_rozpoznania: z
    .number({ message: 'Podaj wiek w chwili rozpoznania' })
    .min(1, 'Wiek musi być większy od 0')
    .max(120, 'Wiek nie może przekraczać 120 lat'),
  opoznienie_rozpoznia: optionalNumber(0, 600, 'mies.'),
  liczba_zajetych_narzadow: z.number().int().min(0).max(20),
  kreatynina: optionalNumber(10, 3000, 'μmol/L'),
  max_crp: optionalNumber(0, 500, 'mg/L'),
  eozynofilia_krwi_obwodowej_wartosc: optionalNumber(0, 50000, '/μL'),
  manifestacja_nerki: flag,
  manifestacja_oddechowy: flag,
  manifestacja_nos_ucho_gardlo: flag,
  manifestacja_sercowo_naczyniowy: flag,
  manifestacja_pokarmowy: flag,
  manifestacja_zajecie_csn: flag,
  manifestacja_neurologiczny: flag,
  manifestacja_miesno_szkiel: flag,
  manifestacja_skora: flag,
  manifestacja_wzrok: flag,
  manifestacja_moczowo_plciowy: flag,
  pulsy: flag,
  plazmaferezy: flag,
  biopsja_wynik: flag,
});

export type BinaryField = (typeof BINARY_FIELDS)[number];
export type PatientFormData = z.infer<typeof patientSchema>;

const NO_FLAGS = Object.fromEntries(BINARY_FIELDS.map((f) => [f, 0])) as Record<BinaryField, 0 | 1>;

export const emptyPatientValues: PatientFormData = {
  ...NO_FLAGS,
  wiek_rozpoznania: 55,
  opoznienie_rozpoznia: undefined,
  liczba_zajetych_narzadow: 0,
  kreatynina: undefined,
  max_crp: undefined,
  eozynofilia_krwi_obwodowej_wartosc: undefined,
};

/** A typical higher-risk presentation (renal + pulmonary involvement, older age). */
export const examplePatientValues: PatientFormData = {
  ...NO_FLAGS,
  wiek_rozpoznania: 71,
  opoznienie_rozpoznia: 2,
  manifestacja_nerki: 1,
  manifestacja_oddechowy: 1,
  manifestacja_sercowo_naczyniowy: 1,
  liczba_zajetych_narzadow: 3,
  kreatynina: 380,
  max_crp: 140,
  eozynofilia_krwi_obwodowej_wartosc: 90,
  pulsy: 1,
  plazmaferezy: 1,
  biopsja_wynik: 0,
};

export function toPatientInput(data: PatientFormData): PatientInput {
  const clean = <T,>(v: T | undefined) => (v === undefined || Number.isNaN(v) ? null : v);
  return {
    ...data,
    opoznienie_rozpoznia: clean(data.opoznienie_rozpoznia),
    kreatynina: clean(data.kreatynina),
    max_crp: clean(data.max_crp),
    eozynofilia_krwi_obwodowej_wartosc: clean(data.eozynofilia_krwi_obwodowej_wartosc),
  };
}

export function toFormValues(patient: PatientInput): PatientFormData {
  const num = (v: number | null | undefined) => (v === null || v === undefined ? undefined : v);
  const flags = Object.fromEntries(BINARY_FIELDS.map((f) => [f, patient[f] === 1 ? 1 : 0])) as Record<BinaryField, 0 | 1>;
  return {
    ...flags,
    wiek_rozpoznania: patient.wiek_rozpoznania ?? 55,
    opoznienie_rozpoznia: num(patient.opoznienie_rozpoznia),
    liczba_zajetych_narzadow: patient.liczba_zajetych_narzadow,
    kreatynina: num(patient.kreatynina),
    max_crp: num(patient.max_crp),
    eozynofilia_krwi_obwodowej_wartosc: num(patient.eozynofilia_krwi_obwodowej_wartosc),
  };
}

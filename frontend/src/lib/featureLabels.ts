/**
 * Single source of Polish labels for model features (backend names) and form fields.
 * Backend names use hyphens/slashes exactly as in models/saved/feature_names.json.
 */
export const FEATURE_LABELS: Record<string, string> = {
  Wiek_rozpoznania: 'Wiek w chwili rozpoznania',
  Opoznienie_Rozpoznia: 'Opóźnienie rozpoznania',
  'Manifestacja_Miesno-Szkiel': 'Mięśnie i stawy',
  Manifestacja_Skora: 'Skóra',
  Manifestacja_Wzrok: 'Narząd wzroku',
  'Manifestacja_Sercowo-Naczyniowy': 'Serce i naczynia',
  Manifestacja_Pokarmowy: 'Układ pokarmowy',
  Manifestacja_Nerki: 'Nerki',
  'Manifestacja_Moczowo-Plciowy': 'Układ moczowo-płciowy',
  Manifestacja_Zajecie_CSN: 'Ośrodkowy układ nerwowy',
  Manifestacja_Neurologiczny: 'Obwodowy układ nerwowy',
  Manifestacja_Oddechowy: 'Układ oddechowy',
  'Manifestacja_Nos/Ucho/Gardlo': 'Nos, ucho, gardło',
  Liczba_Zajetych_Narzadow: 'Liczba zajętych narządów',
  Kreatynina: 'Kreatynina',
  Max_CRP: 'CRP',
  Pulsy: 'Pulsy sterydowe IV',
  Plazmaferezy: 'Plazmaferezy',
  Eozynofilia_Krwi_Obwodowej_Wartosc: 'Eozynofilia',
  Biopsja_Wynik: 'Biopsja wykonana',
};

/** Form field (PatientInput key) -> backend feature name. */
export const FIELD_TO_FEATURE: Record<string, string> = {
  wiek_rozpoznania: 'Wiek_rozpoznania',
  opoznienie_rozpoznia: 'Opoznienie_Rozpoznia',
  manifestacja_miesno_szkiel: 'Manifestacja_Miesno-Szkiel',
  manifestacja_skora: 'Manifestacja_Skora',
  manifestacja_wzrok: 'Manifestacja_Wzrok',
  manifestacja_sercowo_naczyniowy: 'Manifestacja_Sercowo-Naczyniowy',
  manifestacja_pokarmowy: 'Manifestacja_Pokarmowy',
  manifestacja_nerki: 'Manifestacja_Nerki',
  manifestacja_moczowo_plciowy: 'Manifestacja_Moczowo-Plciowy',
  manifestacja_zajecie_csn: 'Manifestacja_Zajecie_CSN',
  manifestacja_neurologiczny: 'Manifestacja_Neurologiczny',
  manifestacja_oddechowy: 'Manifestacja_Oddechowy',
  manifestacja_nos_ucho_gardlo: 'Manifestacja_Nos/Ucho/Gardlo',
  liczba_zajetych_narzadow: 'Liczba_Zajetych_Narzadow',
  kreatynina: 'Kreatynina',
  max_crp: 'Max_CRP',
  pulsy: 'Pulsy',
  plazmaferezy: 'Plazmaferezy',
  eozynofilia_krwi_obwodowej_wartosc: 'Eozynofilia_Krwi_Obwodowej_Wartosc',
  biopsja_wynik: 'Biopsja_Wynik',
};

/** Label for a backend feature name or a form field name. */
export function featureLabel(name: string): string {
  return FEATURE_LABELS[name] ?? FEATURE_LABELS[FIELD_TO_FEATURE[name] ?? ''] ?? name.replace(/_/g, ' ');
}

/** Translate feature names inside a text such as a LIME condition ("Kreatynina > 120.00"). */
export function labelInText(text: string): string {
  // longest names first so "Manifestacja_Nerki" is not split by a shorter match
  const names = Object.keys(FEATURE_LABELS).sort((a, b) => b.length - a.length);
  let out = text;
  for (const n of names) {
    if (out.includes(n)) out = out.split(n).join(FEATURE_LABELS[n]);
  }
  return out;
}

export interface Contribution {
  feature: string;
  contribution: number;
}

const BINARY_FEATURES = new Set([
  'Manifestacja_Miesno-Szkiel', 'Manifestacja_Skora', 'Manifestacja_Wzrok', 'Manifestacja_Sercowo-Naczyniowy',
  'Manifestacja_Pokarmowy', 'Manifestacja_Nerki', 'Manifestacja_Moczowo-Plciowy', 'Manifestacja_Zajecie_CSN',
  'Manifestacja_Neurologiczny', 'Manifestacja_Oddechowy', 'Manifestacja_Nos/Ucho/Gardlo', 'Pulsy', 'Plazmaferezy',
  'Biopsja_Wynik',
]);

/** Readable LIME condition: yes/no features become "Nerki: tak", numbers keep their thresholds. */
export function prettyCondition(condition: string): string {
  for (const f of BINARY_FEATURES) {
    if (!condition.includes(f)) continue;
    const yes = /^0(\.0+)? < /.test(condition) || /> 0(\.0+)?$/.test(condition) || /(>=|=) 1(\.0+)?$/.test(condition);
    return `${FEATURE_LABELS[f]}: ${yes ? 'tak' : 'nie'}`;
  }
  return labelInText(condition).replace(/(\d+\.\d\d)\b/g, (m) => String(Number(m)));
}

const UNITS: Record<string, string> = {
  Wiek_rozpoznania: 'lat',
  Opoznienie_Rozpoznia: 'mies.',
  Kreatynina: 'μmol/L',
  Max_CRP: 'mg/L',
  Eozynofilia_Krwi_Obwodowej_Wartosc: '/μL',
};

/** "Kreatynina: 380 μmol/L", "Nerki: tak" — the patient's value next to the feature name. */
export function featureWithValue(feature: string, value: number | null | undefined): string {
  const label = featureLabel(feature);
  if (value === null || value === undefined || !Number.isFinite(value)) return label;
  if (BINARY_FEATURES.has(feature)) return `${label}: ${value >= 0.5 ? 'tak' : 'nie'}`;
  const rounded = Math.abs(value) >= 10 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${label}: ${rounded}${UNITS[feature] ? ` ${UNITS[feature]}` : ''}`;
}

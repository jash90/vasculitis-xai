import { useId, useState, type ReactNode } from 'react';
import { useForm, type UseFormRegister, type FieldError } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';
import {
  MANIFESTATION_FIELDS,
  emptyPatientValues,
  examplePatientValues,
  patientSchema,
  toPatientInput,
  type BinaryField,
  type PatientFormData,
} from '../../schemas/patient';

interface PatientFormProps {
  onSubmit: (patient: PatientInput) => void;
  isSubmitting: boolean;
  initialValues?: PatientFormData;
}

/* ---------- Section (collapsible) ---------- */
function Section({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(true);
  const id = useId();
  return (
    <section className="rounded-lg border border-gray-700/60 bg-gray-800/30">
      <h3>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-controls={id}
          className="flex w-full items-center justify-between rounded-lg px-4 py-3 text-sm font-semibold text-blue-300 transition hover:bg-gray-800/50"
        >
          {title}
          <span aria-hidden className={`text-xs text-gray-500 transition ${open ? 'rotate-180' : ''}`}>▾</span>
        </button>
      </h3>
      <div id={id} hidden={!open} className="space-y-4 px-4 pb-4">
        {children}
      </div>
    </section>
  );
}

/* ---------- Number input ---------- */
type NumericField = 'wiek_rozpoznania' | 'opoznienie_rozpoznia' | 'kreatynina' | 'max_crp' | 'eozynofilia_krwi_obwodowej_wartosc';

function NumberField({ label, register, name, unit, step, error, placeholder }: {
  label: string;
  register: UseFormRegister<PatientFormData>;
  name: NumericField;
  unit?: string;
  step?: number;
  error?: FieldError;
  placeholder?: string;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-xs font-medium text-gray-400">
        {label}
        {unit && <span className="text-gray-500"> ({unit})</span>}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        step={step ?? 'any'}
        placeholder={placeholder ?? 'brak danych'}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${id}-err` : undefined}
        {...register(name, { setValueAs: (v) => (v === '' || v === null ? undefined : Number(v)) })}
        className={`w-full rounded-lg border bg-gray-700/60 px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 ${
          error ? 'border-red-500 focus:ring-red-500/40' : 'border-gray-600/80 focus:border-blue-500 focus:ring-blue-500/30'
        }`}
      />
      {error && (
        <p id={`${id}-err`} className="mt-1 text-xs text-red-400">
          {error.message}
        </p>
      )}
    </div>
  );
}

/* ---------- Toggle (accessible checkbox switch) ---------- */
function ToggleField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-gray-700/40 bg-gray-800/20 px-3 py-2 transition hover:bg-gray-800/40 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-blue-500">
      <span className="text-sm text-gray-300">{label}</span>
      <span className="relative shrink-0">
        <input
          type="checkbox"
          role="switch"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span aria-hidden className="block h-5 w-9 rounded-full bg-gray-600 transition peer-checked:bg-blue-600" />
        <span aria-hidden className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" />
      </span>
    </label>
  );
}

const ORGANS: [BinaryField, string][] = [
  ['manifestacja_nerki', pl.form.fields.nerki],
  ['manifestacja_oddechowy', pl.form.fields.oddechowy],
  ['manifestacja_nos_ucho_gardlo', pl.form.fields.nos_ucho_gardlo],
  ['manifestacja_sercowo_naczyniowy', pl.form.fields.serce],
  ['manifestacja_pokarmowy', pl.form.fields.pokarmowy],
  ['manifestacja_zajecie_csn', pl.form.fields.csn],
  ['manifestacja_neurologiczny', pl.form.fields.neuro],
  ['manifestacja_miesno_szkiel', pl.form.fields.miesno_szkiel],
  ['manifestacja_skora', pl.form.fields.skora],
  ['manifestacja_wzrok', pl.form.fields.wzrok],
  ['manifestacja_moczowo_plciowy', pl.form.fields.moczowo_plciowy],
];

/* ---------- Main form ---------- */
export function PatientForm({ onSubmit, isSubmitting, initialValues }: PatientFormProps) {
  const form = useForm<PatientFormData>({
    resolver: zodResolver(patientSchema),
    defaultValues: initialValues ?? emptyPatientValues,
    mode: 'onBlur',
  });
  const { register, handleSubmit, setValue, getValues, watch, reset, formState } = form;
  const { errors } = formState;
  const t = pl.form;
  const organCountId = useId();

  const setFlag = (name: BinaryField, value: boolean) => {
    setValue(name, value ? 1 : 0, { shouldDirty: true });
    if ((MANIFESTATION_FIELDS as readonly string[]).includes(name)) {
      const count = MANIFESTATION_FIELDS.filter((f) => (f === name ? value : getValues(f) === 1)).length;
      setValue('liczba_zajetych_narzadow', count, { shouldDirty: true });
    }
  };

  return (
    <form onSubmit={handleSubmit((data) => onSubmit(toPatientInput(data)))} className="space-y-4" noValidate>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-blue-300">{t.title}</h2>
          <p className="mt-1 text-xs text-gray-500">{t.hint}</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => reset(examplePatientValues)}
            className="rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
          >
            {t.example}
          </button>
          <button
            type="button"
            onClick={() => reset(emptyPatientValues)}
            className="rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
          >
            {t.clear}
          </button>
        </div>
      </div>

      <Section title={t.sections.demographics}>
        <div className="grid gap-4 sm:grid-cols-2">
          <NumberField label={t.fields.wiek_rozpoznania} register={register} name="wiek_rozpoznania" unit="lata" step={1}
            error={errors.wiek_rozpoznania} placeholder="np. 60" />
          <NumberField label={t.fields.opoznienie_rozpoznia} register={register} name="opoznienie_rozpoznia" unit="miesiące"
            step={1} error={errors.opoznienie_rozpoznia} />
        </div>
      </Section>

      <Section title={t.sections.organs}>
        <div className="grid gap-2 sm:grid-cols-2">
          {ORGANS.map(([name, label]) => (
            <ToggleField key={name} label={label} checked={watch(name) === 1} onChange={(v) => setFlag(name, v)} />
          ))}
        </div>
        <div>
          <label htmlFor={organCountId} className="mb-1 block text-xs font-medium text-gray-400">
            {t.fields.liczba_narzadow}: <strong className="text-white">{watch('liczba_zajetych_narzadow')}</strong>
          </label>
          <input
            id={organCountId}
            type="range"
            min={0}
            max={12}
            {...register('liczba_zajetych_narzadow', { valueAsNumber: true })}
            className="w-full accent-blue-500"
          />
          <p className="mt-1 text-xs text-gray-500">{t.fields.liczba_narzadow_hint}</p>
        </div>
      </Section>

      <Section title={t.sections.labs}>
        <div className="grid gap-4 sm:grid-cols-3">
          <NumberField label={t.fields.kreatynina} register={register} name="kreatynina" unit="μmol/L" error={errors.kreatynina} />
          <NumberField label={t.fields.max_crp} register={register} name="max_crp" unit="mg/L" error={errors.max_crp} />
          <NumberField label={t.fields.eozynofilia} register={register} name="eozynofilia_krwi_obwodowej_wartosc" unit="/μL"
            step={1} error={errors.eozynofilia_krwi_obwodowej_wartosc} />
        </div>
      </Section>

      <Section title={t.sections.treatment}>
        <div className="grid gap-2 sm:grid-cols-2">
          <ToggleField label={t.fields.pulsy} checked={watch('pulsy') === 1} onChange={(v) => setFlag('pulsy', v)} />
          <ToggleField label={t.fields.plazmaferezy} checked={watch('plazmaferezy') === 1} onChange={(v) => setFlag('plazmaferezy', v)} />
        </div>
      </Section>

      <Section title={t.sections.diagnostics}>
        <ToggleField label={t.fields.biopsja} checked={watch('biopsja_wynik') === 1} onChange={(v) => setFlag('biopsja_wynik', v)} />
      </Section>

      {Object.keys(errors).length > 0 && (
        <p role="alert" className="text-sm text-red-400">Popraw zaznaczone pola, aby kontynuować.</p>
      )}

      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full rounded-xl bg-blue-600 py-3 text-sm font-bold text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-700 disabled:opacity-50"
      >
        {isSubmitting ? t.analyzing : t.analyze}
      </button>
    </form>
  );
}

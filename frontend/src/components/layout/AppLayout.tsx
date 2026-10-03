import { useState } from 'react';
import { Footer } from './Footer';
import { WelcomePage } from '../welcome/WelcomePage';
import { PatientResults } from '../patient/PatientResults';
import { PatientForm } from '../patient/PatientForm';
import { XaiTabs } from '../xai/XaiTabs';
import { AgentChatView } from '../xai/AgentChatView';
import { BatchView } from '../batch/BatchView';
import { ModelsInfoPanel } from '../about/ModelsInfoPanel';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { DemoModeIndicator } from '../common/DemoModeIndicator';
import { errorMessage } from '../../lib/errors';
import { Tabs } from '../common/Tabs';
import { usePredictAll } from '../../hooks/useApi';
import { useBatchAnalysis } from '../../hooks/useBatchAnalysis';
import { emptyPatientValues, toFormValues, type PatientFormData } from '../../schemas/patient';
import { pl } from '../../i18n/pl';
import type { MultiModelPredictionOutput, PatientInput } from '../../api/types';

type AppMode = 'single' | 'agent' | 'batch' | 'about';

const NAV_ITEMS = [
  { id: 'single', label: pl.nav.singlePatient },
  { id: 'agent', label: pl.nav.agent },
  { id: 'batch', label: pl.nav.batch },
  { id: 'about', label: pl.nav.about },
] as const;

export function AppLayout() {
  const [mode, setMode] = useState<AppMode>('single');
  const [patient, setPatient] = useState<PatientInput | null>(null);
  const [result, setResult] = useState<MultiModelPredictionOutput | null>(null);
  const [formValues, setFormValues] = useState<PatientFormData>(emptyPatientValues);
  const [formKey, setFormKey] = useState(0);
  const predict = usePredictAll();
  const batch = useBatchAnalysis();

  const analyze = async (p: PatientInput) => {
    setPatient(p);
    setFormValues(toFormValues(p));
    setResult(null);
    window.scrollTo({ top: 0 });
    try {
      setResult(await predict.mutateAsync(p));
    } catch {
      /* error shown from predict.error */
    }
  };

  const backToForm = (clear: boolean) => {
    predict.reset();
    setResult(null);
    if (clear) setFormValues(emptyPatientValues);
    setFormKey((k) => k + 1);
    window.scrollTo({ top: 0 });
  };

  const showForm = !result && !predict.isPending && !predict.isError;

  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-blue-600 focus:px-3 focus:py-2 focus:text-white">
        Przejdź do treści
      </a>
      <header className="sticky top-0 z-30 border-b border-gray-700/80 bg-gray-900/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-3 sm:px-6">
          <h1 className="text-base font-bold text-blue-300 sm:text-lg md:text-xl">{pl.app.title}</h1>
          <DemoModeIndicator />
        </div>
        <nav aria-label={pl.nav.label} className="mx-auto max-w-7xl px-2 sm:px-6">
          <Tabs items={NAV_ITEMS} value={mode} onChange={setMode} label={pl.nav.label} idPrefix="nav" />
        </nav>
      </header>

      <main id="main" className="flex-1">
        <div id="nav-panel" role="tabpanel" aria-labelledby={`nav-tab-${mode}`} className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
          {mode === 'single' && (
            <>
              {showForm && (
                <div className="space-y-10">
                  <WelcomePage onShowModels={() => setMode('about')} />
                  <div className="mx-auto max-w-2xl rounded-xl border border-gray-700 bg-gray-800/40 p-4 sm:p-6">
                    <PatientForm key={formKey} onSubmit={analyze} isSubmitting={predict.isPending} initialValues={formValues} />
                  </div>
                </div>
              )}

              {predict.isPending && <LoadingSkeleton />}

              {predict.isError && (
                <div role="alert" className="mx-auto max-w-2xl rounded-xl border border-red-500/40 bg-red-900/15 p-8 text-center">
                  <p className="text-lg font-semibold text-red-300">{pl.common.predictionError}</p>
                  <p className="mt-2 text-sm text-gray-400">{errorMessage(predict.error)}</p>
                  <div className="mt-5 flex justify-center gap-3">
                    <button type="button" onClick={() => patient && analyze(patient)}
                      className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
                      {pl.common.retry}
                    </button>
                    <button type="button" onClick={() => backToForm(false)}
                      className="rounded-md border border-gray-600 px-4 py-2 text-sm text-gray-200 hover:bg-gray-700">
                      {pl.common.back}
                    </button>
                  </div>
                </div>
              )}

              {result && patient && (
                <div className="space-y-10">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-2xl font-bold text-blue-300">{pl.results.title}</h2>
                    <div className="flex gap-2">
                      <button type="button" onClick={() => backToForm(false)}
                        className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-700">
                        ← {pl.results.editData}
                      </button>
                      <button type="button" onClick={() => backToForm(true)}
                        className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700">
                        {pl.results.newAnalysis}
                      </button>
                    </div>
                  </div>
                  <PatientResults patient={patient} result={result} />
                  <hr className="border-gray-700/50" />
                  <XaiTabs patient={patient} />
                </div>
              )}
            </>
          )}

          {/* Kept mounted so the conversation survives switching tabs */}
          <div hidden={mode !== 'agent'}>
            <AgentChatView />
          </div>

          {mode === 'batch' && <BatchView batch={batch} />}

          {mode === 'about' && <ModelsInfoPanel />}
        </div>
      </main>

      <Footer />
    </div>
  );
}

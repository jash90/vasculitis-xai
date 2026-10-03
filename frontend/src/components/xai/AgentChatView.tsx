import { useId, useRef, useEffect, useState } from 'react';
import { useAgentChat } from '../../hooks/useApi';
import { GaugeChart } from '../charts/GaugeChart';
import { ContributionChart } from '../charts/ContributionChart';
import { RiskBadge } from '../common/RiskBadge';
import { MarkdownMessage } from '../common/MarkdownMessage';
import { featureLabel } from '../../lib/featureLabels';
import type { AgentFieldMeta, ChatPredictionData } from '../../api/types';
import type { AgentChatPayload } from '../../api/endpoints';

type Phase = AgentChatPayload['phase'];
type Collected = Record<string, number | string | null>;

interface Message {
  role: 'user' | 'assistant';
  content: string;
  prediction_data?: ChatPredictionData | null;
}

const BOOL_FIELDS = new Set([
  'manifestacja_nerki', 'manifestacja_oddechowy', 'manifestacja_nos_ucho_gardlo', 'manifestacja_sercowo_naczyniowy',
  'manifestacja_pokarmowy', 'manifestacja_zajecie_csn', 'manifestacja_neurologiczny', 'manifestacja_skora',
  'pulsy', 'plazmaferezy', 'biopsja_wynik',
]);

function formatValue(key: string, val: number | string | null): string {
  if (val === null || val === undefined) return 'nie wiem';
  if (BOOL_FIELDS.has(key)) return val ? 'tak' : 'nie';
  return String(val);
}

function CollectedSummary({ data }: { data: Collected }) {
  const entries = Object.entries(data);
  if (!entries.length) return null;
  return (
    <details className="rounded-lg border border-gray-600 bg-gray-800/80 p-3" open>
      <summary className="cursor-pointer text-xs font-semibold text-blue-300">Zebrane dane ({entries.length})</summary>
      <ul className="mt-2 flex flex-wrap gap-2">
        {entries.map(([key, val]) => (
          <li key={key} className="rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-300">
            {featureLabel(key)}: <strong>{formatValue(key, val)}</strong>
          </li>
        ))}
      </ul>
    </details>
  );
}

function SliderWidget({ meta, onSend }: { meta: AgentFieldMeta; onSend: (val: string) => void }) {
  const min = meta.min ?? 0;
  const max = meta.max ?? 100;
  const [val, setVal] = useState(meta.default ?? Math.round((min + max) / 2));
  const id = useId();
  const unit = meta.unit ? ` ${meta.unit}` : '';
  return (
    <div className="rounded-lg border border-blue-500/30 bg-blue-900/10 p-3">
      <label htmlFor={id} className="mb-2 flex items-center justify-between text-xs text-gray-400">
        <span>{min}</span>
        <span className="rounded bg-gray-800 px-3 py-1 text-lg font-bold text-white tabular-nums">{val}{unit}</span>
        <span>{max}</span>
      </label>
      <input id={id} type="range" min={min} max={max} step={meta.step ?? 1} value={val}
        onChange={(e) => setVal(Number(e.target.value))} className="w-full accent-blue-500" />
      <div className="mt-2 flex gap-2">
        <button type="button" onClick={() => onSend(String(val))}
          className="flex-1 rounded-lg bg-blue-600 py-2 text-sm font-medium text-white transition hover:bg-blue-700">
          Potwierdź: {val}{unit}
        </button>
        {meta.skippable && (
          <button type="button" onClick={() => onSend('nie wiem')}
            className="rounded-lg border border-gray-600 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700">
            Nie wiem
          </button>
        )}
      </div>
    </div>
  );
}

function ChoiceButtons({ options, onSend }: { options: string[]; onSend: (val: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => (
        <button key={opt} type="button" onClick={() => onSend(opt)}
          className="rounded-full border border-blue-500/50 bg-blue-900/20 px-4 py-1.5 text-sm text-blue-200 transition hover:bg-blue-900/40">
          {opt}
        </button>
      ))}
    </div>
  );
}

export function AgentChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [collected, setCollected] = useState<Collected>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [phase, setPhase] = useState<Phase>('collecting');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [fieldMeta, setFieldMeta] = useState<AgentFieldMeta | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const agent = useAgentChat();
  const listRef = useRef<HTMLDivElement>(null);
  const inputId = useId();
  const started = messages.length > 0 || agent.isPending || failed !== null;

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, fieldMeta, agent.isPending]);

  const send = async (msg: string, fresh = false) => {
    setFailed(null);
    setFieldMeta(null);
    if (!fresh) setMessages((prev) => [...prev, { role: 'user', content: msg }]);
    setInput('');
    try {
      const res = await agent.mutateAsync({
        message: msg,
        conversation_history: fresh ? [] : messages.slice(-20).map((m) => ({ role: m.role, content: m.content })),
        collected_data: fresh ? {} : collected,
        current_step: fresh ? 0 : currentStep,
        phase: fresh ? 'collecting' : phase,
      });
      setMessages((prev) => [...(fresh ? [] : prev), { role: 'assistant', content: res.response, prediction_data: res.prediction_data }]);
      setCollected(res.collected_data);
      setCurrentStep(res.current_step);
      setPhase(res.phase);
      setSuggestions(res.follow_up_suggestions ?? []);
      setFieldMeta(res.field_meta ?? null);
    } catch {
      setFailed(msg);
    }
  };

  const restart = () => {
    setMessages([]);
    setCollected({});
    setCurrentStep(0);
    setPhase('collecting');
    setSuggestions([]);
    void send('start', true);
  };

  if (!started) {
    return (
      <div className="mx-auto max-w-lg py-12 text-center">
        <h2 className="mb-3 text-2xl font-bold text-blue-300">Asystent AI — rozmowa o pacjencie</h2>
        <p className="mb-6 text-gray-400">
          Asystent zada 16 krótkich pytań o stan pacjenta w chwili rozpoznania, obliczy ryzyko i wyjaśni wynik.
          Na każde pytanie możesz odpowiedzieć „nie wiem”.
        </p>
        <button type="button" onClick={restart}
          className="rounded-lg bg-blue-600 px-8 py-3 text-lg font-bold text-white transition hover:bg-blue-700">
          Rozpocznij rozmowę
        </button>
      </div>
    );
  }

  const showWidget = phase === 'collecting' && fieldMeta && !agent.isPending && !failed;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-blue-300">Asystent AI</h2>
        <button type="button" onClick={restart} disabled={agent.isPending}
          className="rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700 disabled:opacity-50">
          Zacznij od nowa
        </button>
      </div>

      <CollectedSummary data={collected} />

      <div ref={listRef} className="h-[30rem] space-y-4 overflow-y-auto rounded-lg border border-gray-700 bg-gray-900/40 p-4" aria-live="polite">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === 'user' ? 'ml-auto w-fit max-w-[80%]' : 'mr-auto max-w-[95%]'}>
            <div className={`rounded-lg px-4 py-3 text-sm leading-relaxed ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200'}`}>
              {msg.role === 'user' ? <p>{msg.content}</p> : <MarkdownMessage content={msg.content} />}
            </div>
            {msg.role === 'assistant' && msg.prediction_data && (
              <div className="mt-3 grid gap-4 rounded-lg border border-gray-600 bg-gray-800 p-4 md:grid-cols-2">
                <div className="text-center">
                  <GaugeChart probability={msg.prediction_data.prediction.probability} title="Ryzyko zgonu" />
                  <RiskBadge level={msg.prediction_data.prediction.risk_level} compact />
                </div>
                <ContributionChart
                  factors={msg.prediction_data.factors.map((f) => ({ feature: featureLabel(f.feature), contribution: f.contribution }))}
                  maxItems={8}
                />
              </div>
            )}
          </div>
        ))}
        {agent.isPending && (
          <div role="status" className="max-w-[80%] rounded-lg bg-gray-700 px-4 py-3 text-sm text-gray-400">Piszę odpowiedź…</div>
        )}
        {failed !== null && (
          <div role="alert" className="rounded-lg border border-red-500/40 bg-red-900/15 p-3 text-sm text-red-200">
            Nie udało się połączyć z serwerem.
            <button type="button" onClick={() => (messages.length ? send(failed) : restart())}
              className="ml-3 rounded border border-red-400/50 px-2 py-0.5 text-xs hover:bg-red-900/30">
              Spróbuj ponownie
            </button>
          </div>
        )}
      </div>

      <div>
        {showWidget && fieldMeta.widget === 'slider' && (
          <SliderWidget key={`${currentStep}-${fieldMeta.field}`} meta={fieldMeta} onSend={send} />
        )}
        {showWidget && fieldMeta.widget === 'buttons' && fieldMeta.options && (
          <ChoiceButtons options={fieldMeta.options} onSend={send} />
        )}
        {phase === 'discussion' && suggestions.length > 0 && !agent.isPending && (
          <div className="mb-2">
            <ChoiceButtons options={suggestions} onSend={send} />
          </div>
        )}
        {!showWidget && (
          <form className="mt-2 flex" onSubmit={(e) => { e.preventDefault(); if (input.trim()) send(input.trim()); }}>
            <label htmlFor={inputId} className="sr-only">Wiadomość do asystenta</label>
            <input id={inputId} type="text" value={input} onChange={(e) => setInput(e.target.value)}
              placeholder="Zadaj pytanie…" maxLength={2000} disabled={agent.isPending}
              className="flex-1 rounded-l-lg border border-gray-600 bg-gray-700 px-4 py-3 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none" />
            <button type="submit" disabled={agent.isPending || !input.trim()}
              className="rounded-r-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">
              Wyślij
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

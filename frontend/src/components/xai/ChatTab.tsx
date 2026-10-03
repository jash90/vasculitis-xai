import { useId, useRef, useEffect, useState } from 'react';
import { useChat } from '../../hooks/useApi';
import { GaugeChart } from '../charts/GaugeChart';
import { ContributionChart } from '../charts/ContributionChart';
import { RiskBadge } from '../common/RiskBadge';
import { MarkdownMessage } from '../common/MarkdownMessage';
import { featureLabel } from '../../lib/featureLabels';
import { pl } from '../../i18n/pl';
import type { PatientInput, ChatPredictionData } from '../../api/types';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  prediction_data?: ChatPredictionData | null;
}

const SUGGESTIONS = ['Jakie czynniki najbardziej wpływają na ryzyko?', 'Co oznacza ten wynik?', 'Na co zwrócić uwagę?'];

export function ChatTab({ patient }: { patient: PatientInput }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const chatMutation = useChat();
  const listRef = useRef<HTMLDivElement>(null);
  const inputId = useId();

  // Scroll inside the message list only (never the whole page).
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, chatMutation.isPending]);

  const send = async (text: string) => {
    const prompt = text.trim();
    if (!prompt || chatMutation.isPending) return;
    setMessages((prev) => [...prev, { role: 'user', content: prompt }]);
    setInput('');
    try {
      const res = await chatMutation.mutateAsync({
        message: prompt,
        patient,
        health_literacy: 'clinician',
        conversation_history: messages.slice(-20).map((m) => ({ role: m.role, content: m.content })),
      });
      setMessages((prev) => [...prev, { role: 'assistant', content: res.response, prediction_data: res.prediction_data }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Nie udało się uzyskać odpowiedzi z serwera. Sprawdź połączenie z API i spróbuj ponownie.' },
      ]);
    }
  };

  return (
    <section aria-labelledby="chat-title">
      <h3 id="chat-title" className="mb-3 text-lg font-semibold text-blue-300">{pl.xai.chatTitle}</h3>
      <div className="flex h-[28rem] flex-col rounded-lg border border-gray-700 bg-gray-800/50">
        <div ref={listRef} className="flex-1 space-y-3 overflow-y-auto p-4" aria-live="polite">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <p className="text-sm text-gray-400">Zadaj pytanie o wynik analizy lub wybierz podpowiedź:</p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} type="button" onClick={() => send(s)}
                    className="rounded-full border border-blue-500/50 bg-blue-900/20 px-3 py-1 text-xs text-blue-200 hover:bg-blue-900/40">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={msg.role === 'user' ? 'ml-auto w-fit max-w-[80%]' : 'mr-auto max-w-[95%]'}>
              <div className={`rounded-lg px-3 py-2 text-sm leading-relaxed ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200'}`}>
                {msg.role === 'user' ? <p>{msg.content}</p> : <MarkdownMessage content={msg.content} />}
              </div>
              {msg.role === 'assistant' && msg.prediction_data && (
                <div className="mt-2 grid gap-4 rounded-lg border border-gray-600 bg-gray-800 p-3 md:grid-cols-2">
                  <div className="text-center">
                    <GaugeChart probability={msg.prediction_data.prediction.probability} height={180} />
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
          {chatMutation.isPending && (
            <div role="status" className="max-w-[80%] rounded-lg bg-gray-700 px-3 py-2 text-sm text-gray-400">Piszę odpowiedź…</div>
          )}
        </div>
        <form
          className="flex border-t border-gray-700 p-2"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <label htmlFor={inputId} className="sr-only">Pytanie do asystenta</label>
          <input
            id={inputId}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={pl.xai.chatPlaceholder}
            maxLength={2000}
            className="flex-1 rounded-l-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            disabled={chatMutation.isPending}
          />
          <button
            type="submit"
            disabled={chatMutation.isPending || !input.trim()}
            className="rounded-r-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Wyślij
          </button>
        </form>
      </div>
    </section>
  );
}

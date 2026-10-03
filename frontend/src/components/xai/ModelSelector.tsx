interface ModelSelectorProps {
  value: string;
  onChange: (key: string) => void;
}

const MODELS = [
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'random_forest', label: 'Random Forest' },
  { key: 'lightgbm', label: 'LightGBM' },
] as const;

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-2">
      <span id="model-selector-label" className="text-xs text-gray-400">Wyjaśniany model:</span>
      <div role="group" aria-labelledby="model-selector-label" className="flex overflow-hidden rounded-lg border border-gray-600">
        {MODELS.map((m) => (
          <button
            key={m.key}
            type="button"
            aria-pressed={value === m.key}
            onClick={() => onChange(m.key)}
            className={`px-3 py-1.5 text-xs font-medium transition ${
              value === m.key ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}

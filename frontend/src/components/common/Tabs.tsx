import { useRef, type KeyboardEvent } from 'react';

export interface TabItem<T extends string> {
  id: T;
  label: string;
}

interface TabsProps<T extends string> {
  items: readonly TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  idPrefix: string;
  size?: 'md' | 'sm';
}

/** Accessible tab list (role=tablist, arrow-key navigation, aria-selected). */
export function Tabs<T extends string>({ items, value, onChange, label, idPrefix, size = 'md' }: TabsProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const delta = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!delta) return;
    e.preventDefault();
    const next = (index + delta + items.length) % items.length;
    onChange(items[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div role="tablist" aria-label={label} className="flex overflow-x-auto">
      {items.map((item, i) => {
        const selected = item.id === value;
        return (
          <button
            key={item.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            id={`${idPrefix}-tab-${item.id}`}
            role="tab"
            type="button"
            aria-selected={selected}
            aria-controls={`${idPrefix}-panel`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={`relative shrink-0 whitespace-nowrap font-medium transition ${
              size === 'sm' ? 'px-3 py-2 text-sm sm:px-4' : 'px-3 py-2.5 text-sm sm:px-5'
            } ${selected ? 'text-blue-400' : 'text-gray-400 hover:text-gray-200'}`}
          >
            {item.label}
            {selected && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500" />}
          </button>
        );
      })}
    </div>
  );
}

import { isAxiosError } from 'axios';

/** Human-readable (Polish) message for an API error. */
export function errorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    if (!error.response) return 'Brak połączenia z serwerem API.';
    if (error.code === 'ECONNABORTED') return 'Serwer nie odpowiedział w wyznaczonym czasie.';
    const detail = (error.response.data as { detail?: unknown })?.detail;
    if (typeof detail === 'string') return detail;
    return `Serwer zwrócił błąd (${error.response.status}).`;
  }
  return 'Wystąpił nieoczekiwany błąd.';
}

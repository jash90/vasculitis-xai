"""
Szablony promptów dla agenta konwersacyjnego.

Zawiera prompty systemowe i szablony wyjaśnień dla różnych
poziomów health literacy (pacjenci, klinicyści).
"""

# ============================================================================
# PROMPTY SYSTEMOWE
# ============================================================================

SYSTEM_PROMPT_CLINICIAN = """
Jesteś asystentem medycznym wspierającym lekarzy w interpretacji wyników
analizy ryzyka śmiertelności u pacjentów z zapaleniem naczyń.

TWOJE KOMPETENCJE:
- Wyjaśnianie wyników modeli predykcyjnych ML
- Interpretacja wartości SHAP i LIME
- Omówienie czynników ryzyka i ochronnych
- Kontekstualizacja wyników w świetle wiedzy medycznej

ZASADY KOMUNIKACJI:
1. Używaj terminologii medycznej odpowiedniej dla lekarzy specjalistów
2. Podawaj konkretne wartości liczbowe i przedziały ufności gdy dostępne
3. Odniesienie do literatury medycznej jeśli to możliwe
4. Zachowaj obiektywizm - przedstaw zarówno ograniczenia jak i mocne strony analizy
5. NIGDY nie używaj emoji w odpowiedziach

STRUKTURA ODPOWIEDZI:
1. Podsumowanie predykcji (ryzyko, prawdopodobieństwo)
2. Kluczowe czynniki wpływające na wynik
3. Kontekst kliniczny
4. Ograniczenia analizy
5. Sugestie dalszego postępowania (jeśli stosowne)

ZAWSZE PAMIĘTAJ:
- Model jest narzędziem wspierającym, nie zastępuje oceny klinicznej
- Podkreślaj niepewność predykcji
- Nie formułuj bezpośrednich zaleceń terapeutycznych
- NIE UŻYWAJ EMOJI -- komunikacja wyłącznie tekstowa
"""

SYSTEM_PROMPT_PATIENT_BASIC = """
Jesteś pomocnym asystentem wyjaśniającym wyniki analizy zdrowotnej.
Rozmawiasz z pacjentem, który może nie mieć wykształcenia medycznego.

ZASADY KOMUNIKACJI:
1. Używaj prostego języka zrozumiałego dla ucznia 8 klasy podstawowej
2. NIGDY nie używaj terminów technicznych jak "SHAP", "model", "algorytm", "predykcja"
3. Mów o "czynnikach" i "wskaźnikach zdrowia" zamiast "features" czy "zmiennych"
4. Używaj porównań do codziennych sytuacji
5. Bądź cierpliwy i empatyczny
6. Powtarzaj ważne informacje różnymi słowami
7. NIGDY nie używaj emoji w odpowiedziach

ZAMIEŃ TERMINOLOGIĘ:
- "prawdopodobieństwo" → "jak bardzo prawdopodobne"
- "czynniki ryzyka" → "rzeczy wymagające uwagi"
- "czynniki ochronne" → "rzeczy działające na Twoją korzyść"
- "predykcja modelu" → "wyniki analizy"
- "wysoki wskaźnik" → "podwyższona wartość"

STRUKTURA ODPOWIEDZI:
1. Najpierw wyjaśnij co oznacza ogólny wynik (spokojnie, bez paniki)
2. Wymień najważniejsze rzeczy wymagające uwagi (max 3)
3. Wymień pozytywne aspekty (max 3)
4. Zachęć do rozmowy z lekarzem
5. Zawsze kończ disclaimerem

NIGDY NIE:
- Diagnozuj chorób
- Zalecaj konkretnych leków
- Podważaj decyzji lekarza
- Podawaj prognoz liczbowych ("50% szans na...")
- Strasz pacjenta
- Używaj emoji
"""

SYSTEM_PROMPT_PATIENT_ADVANCED = """
Jesteś asystentem medycznym wyjaśniającym wyniki analizy ryzyka.
Rozmawiasz z wykształconym pacjentem, który rozumie podstawową terminologię.

ZASADY KOMUNIKACJI:
1. Możesz używać podstawowej terminologii medycznej
2. Wyjaśniaj koncepcje statystyczne w przystępny sposób
3. Bądź precyzyjny, ale unikaj żargonu ML
4. Odpowiadaj na pytania szczegółowo
5. NIGDY nie używaj emoji w odpowiedziach

DOZWOLONE TERMINY:
- czynniki ryzyka, czynniki ochronne
- prawdopodobieństwo, ryzyko względne
- wskaźniki laboratoryjne (CRP, kreatynina)
- nazwy narządów i układów

UNIKAJ:
- wartości SHAP, LIME
- dokładnych wartości prawdopodobieństwa
- terminologii ML (model, trening, walidacja)
- emoji

ZAWSZE:
- Podkreślaj że to narzędzie wspierające
- Zachęcaj do konsultacji z lekarzem
- Dodaj disclaimer na końcu
- Komunikuj się wyłącznie tekstem, bez emoji
"""

# ============================================================================
# SZABLONY WYJAŚNIEŃ
# ============================================================================

EXPLANATION_TEMPLATE_PATIENT = """
Na podstawie analizy Twoich wyników zdrowotnych, przygotowaliśmy podsumowanie.

OGÓLNA OCENA:
{overall_assessment}

CZYNNIKI WYMAGAJĄCE UWAGI:
{risk_factors}

CZYNNIKI POZYTYWNE:
{protective_factors}

ZALECENIA:
{recommendations}

{disclaimer}
"""

EXPLANATION_TEMPLATE_CLINICIAN = """
## Analiza ryzyka śmiertelności

### Predykcja modelu
- **Prawdopodobieństwo zgonu**: {probability:.1%}
- **Poziom ryzyka**: {risk_level}
- **95% CI**: {confidence_interval}

### Główne czynniki wpływające na predykcję

#### Czynniki zwiększające ryzyko:
{risk_factors}

#### Czynniki zmniejszające ryzyko:
{protective_factors}

### Metodologia
- Model: {model_type}
- Metoda XAI: {xai_method}
- Bazowe ryzyko (intercept): {base_value:.3f}

### Uwagi
{notes}

---
*Ten raport został wygenerowany automatycznie i służy wyłącznie jako narzędzie wspierające decyzje kliniczne.*
"""

# ============================================================================
# GUARDRAILS
# ============================================================================

GUARDRAILS = """
BEZWZGLĘDNE ZAKAZY - NIGDY NIE:

1. PROGNOZY LICZBOWE
   - "Ma Pan/Pani 50% szans na..."
   - "Rokowanie wynosi X lat"
   - "Przeżywalność to Y%"

2. ZALECENIA TERAPEUTYCZNE
   - "Powinien Pan/Pani zażywać lek X"
   - "Zalecam zwiększenie dawki"
   - "Proszę odstawić to leczenie"

3. DIAGNOZY
   - "Ma Pan/Pani chorobę X"
   - "To oznacza że cierpi Pan/Pani na..."
   - "Wyniki wskazują na diagnozę..."

4. KWESTIONOWANIE LEKARZA
   - "Lekarz się myli"
   - "To leczenie jest niewłaściwe"
   - "Powinien Pan/Pani zmienić lekarza"

5. INTERPRETACJA BADAŃ LABORATORYJNYCH
   - "Twoja kreatynina 150 oznacza niewydolność nerek"
   - "CRP 80 to stan zapalny"
   (Można wspomnieć że wartość jest podwyższona, ale nie diagnozować)

OBOWIĄZKOWE REAKCJE:

1. PRZY PYTANIACH O SAMOBÓJSTWO/SAMOOKALECZENIE:
   "Rozumiem, że możesz przechodzić trudny czas. To są poważne myśli
   i bardzo ważne jest, żebyś porozmawiał z kimś, kto może pomóc.
   Proszę skontaktuj się z:
   - Telefon zaufania: 116 123
   - Centrum Wsparcia: 800 70 2222
   - Lub udaj się na najbliższy oddział ratunkowy"

2. PRZY PROŚBIE O DIAGNOZĘ:
   "Nie jestem w stanie postawić diagnozy - to wymaga bezpośredniego
   badania przez lekarza. Zachęcam do umówienia wizyty."

3. PRZY PROŚBIE O ZMIANĘ LECZENIA:
   "Decyzje o leczeniu powinny być podejmowane wspólnie z Twoim
   lekarzem prowadzącym, który zna pełną historię Twojego zdrowia."
"""

# ============================================================================
# DISCLAIMER
# ============================================================================

DISCLAIMER_PATIENT = """
---
**Ważne**: To narzędzie ma charakter wyłącznie informacyjny.
Nie zastępuje profesjonalnej diagnozy medycznej ani porady lekarza.
Jeśli masz pytania dotyczące swojego zdrowia, skonsultuj się
z lekarzem prowadzącym.
"""

DISCLAIMER_CLINICIAN = """
---
*Uwaga: Ten system jest narzędziem wspierającym decyzje kliniczne (Clinical Decision Support).
Wyniki predykcji opierają się na modelach statystycznych i powinny być interpretowane
w kontekście pełnego obrazu klinicznego pacjenta. Wszystkie decyzje terapeutyczne
pozostają w gestii lekarza prowadzącego.*
"""

# ============================================================================
# POMOCNICZE SZABLONY
# ============================================================================

FEATURE_TRANSLATIONS = {
    'Wiek': 'Twój wiek',
    'Plec': 'Płeć',
    'Wiek_rozpoznania': 'Wiek w momencie rozpoznania choroby',
    'Opoznienie_Rozpoznia': 'Czas do postawienia diagnozy',
    'Liczba_Zajetych_Narzadow': 'Liczba narządów objętych chorobą',
    'Manifestacja_Sercowo-Naczyniowy': 'Stan układu sercowo-naczyniowego',
    'Manifestacja_Nerki': 'Stan nerek',
    'Manifestacja_Pokarmowy': 'Stan układu pokarmowego',
    'Manifestacja_Zajecie_CSN': 'Stan ośrodkowego układu nerwowego',
    'Manifestacja_Neurologiczny': 'Objawy neurologiczne',
    'Zaostrz_Wymagajace_OIT': 'Historia pobytu na intensywnej terapii',
    'Kreatynina': 'Poziom wskaźnika czynności nerek',
    'Manifestacja_Oddechowy': 'Stan układu oddechowego',
    'Manifestacja_Nos/Ucho/Gardlo': 'Objawy ze strony nosa, ucha lub gardła',
    'Max_CRP': 'Poziom stanu zapalnego w organizmie',
    'Plazmaferezy': 'Przebyte zabiegi oczyszczania krwi',
    'Dializa': 'Historia leczenia nerkozastępczego',
    'Sterydy_Dawka_g': 'Dawka stosowanych sterydów',
    'Czas_Sterydow': 'Czas trwania leczenia sterydami',
    'Powiklania_Serce/pluca': 'Powikłania sercowo-płucne',
    'Powiklania_Infekcja': 'Przebyte infekcje jako powikłania'
}

RISK_LEVEL_DESCRIPTIONS = {
    'low': {
        'patient': 'Analiza wskazuje na niskie ryzyko. To dobra wiadomość, ale ważne jest regularne monitorowanie stanu zdrowia.',
        'clinician': 'Niskie ryzyko zgonu (prawdopodobieństwo <30%). Zalecany standardowy schemat follow-up.'
    },
    'moderate': {
        'patient': 'Analiza wskazuje na umiarkowane ryzyko. Warto zwrócić uwagę na kilka czynników i regularnie konsultować się z lekarzem.',
        'clinician': 'Umiarkowane ryzyko zgonu (prawdopodobieństwo 30-70%). Zalecana intensyfikacja monitorowania.'
    },
    'high': {
        'patient': 'Analiza wskazuje na podwyższone ryzyko. Ważne jest, aby być pod stałą opieką specjalisty i przestrzegać zaleceń.',
        'clinician': 'Wysokie ryzyko zgonu (prawdopodobieństwo >70%). Zalecana intensywna opieka i rozważenie eskalacji leczenia.'
    }
}


def get_risk_level(probability: float) -> str:
    """Określ poziom ryzyka na podstawie prawdopodobieństwa."""
    if probability < 0.3:
        return 'low'
    elif probability < 0.7:
        return 'moderate'
    else:
        return 'high'


def translate_feature(feature_name: str) -> str:
    """Przetłumacz nazwę cechy na język przyjazny dla pacjenta."""
    return FEATURE_TRANSLATIONS.get(feature_name, feature_name)


def format_risk_factors_patient(risk_factors: list, max_items: int = 5) -> str:
    """Sformatuj czynniki ryzyka dla pacjenta."""
    if not risk_factors:
        return "Nie zidentyfikowano znaczących czynników ryzyka."

    lines = []
    for i, factor in enumerate(risk_factors[:max_items], 1):
        if isinstance(factor, dict):
            feature = factor.get('feature', factor.get('variable', ''))
        elif isinstance(factor, tuple):
            feature = factor[0]
        else:
            feature = str(factor)

        translated = translate_feature(feature.split()[0] if ' ' in feature else feature)
        lines.append(f"{i}. {translated}")

    return "\n".join(lines)


def format_risk_factors_clinician(risk_factors: list, max_items: int = 10) -> str:
    """Sformatuj czynniki ryzyka dla klinicysty."""
    if not risk_factors:
        return "Brak znaczących czynników ryzyka."

    lines = []
    for factor in risk_factors[:max_items]:
        if isinstance(factor, dict):
            feature = factor.get('feature', factor.get('variable', ''))
            value = factor.get('shap_value', factor.get('contribution', factor.get('score', 0)))
            lines.append(f"- **{feature}**: wpływ {value:+.3f}")
        elif isinstance(factor, tuple):
            feature, value = factor[0], factor[1]
            lines.append(f"- **{feature}**: wpływ {value:+.3f}")
        else:
            lines.append(f"- {factor}")

    return "\n".join(lines)


def generate_patient_explanation(
    probability: float,
    risk_factors: list,
    protective_factors: list,
    health_literacy: str = 'basic'
) -> str:
    """
    Wygeneruj pełne wyjaśnienie dla pacjenta.

    Args:
        probability: Prawdopodobieństwo zgonu
        risk_factors: Lista czynników ryzyka
        protective_factors: Lista czynników ochronnych
        health_literacy: Poziom ('basic' lub 'advanced')

    Returns:
        Sformatowane wyjaśnienie
    """
    risk_level = get_risk_level(probability)
    level_desc = RISK_LEVEL_DESCRIPTIONS[risk_level]['patient']

    risk_text = format_risk_factors_patient(risk_factors)
    protective_text = format_risk_factors_patient(protective_factors)

    explanation = EXPLANATION_TEMPLATE_PATIENT.format(
        overall_assessment=level_desc,
        risk_factors=risk_text if risk_text else "Brak szczególnych czynników ryzyka.",
        protective_factors=protective_text if protective_text else "Analiza trwa.",
        recommendations="Zalecamy omówienie tych wyników z lekarzem prowadzącym podczas następnej wizyty.",
        disclaimer=DISCLAIMER_PATIENT
    )

    return explanation


def generate_clinician_explanation(
    probability: float,
    risk_factors: list,
    protective_factors: list,
    model_type: str = 'XGBoost',
    xai_method: str = 'SHAP',
    base_value: float = 0.0,
    confidence_interval: str = 'N/A',
    notes: str = ''
) -> str:
    """
    Wygeneruj pełne wyjaśnienie dla klinicysty.

    Args:
        probability: Prawdopodobieństwo zgonu
        risk_factors: Lista czynników ryzyka
        protective_factors: Lista czynników ochronnych
        model_type: Typ modelu
        xai_method: Metoda XAI
        base_value: Wartość bazowa
        confidence_interval: Przedział ufności
        notes: Dodatkowe uwagi

    Returns:
        Sformatowane wyjaśnienie (Markdown)
    """
    risk_level = get_risk_level(probability)
    level_desc = RISK_LEVEL_DESCRIPTIONS[risk_level]['clinician']

    risk_text = format_risk_factors_clinician(risk_factors)
    protective_text = format_risk_factors_clinician(protective_factors)

    explanation = EXPLANATION_TEMPLATE_CLINICIAN.format(
        probability=probability,
        risk_level=f"{risk_level.upper()} - {level_desc}",
        confidence_interval=confidence_interval,
        risk_factors=risk_text,
        protective_factors=protective_text,
        model_type=model_type,
        xai_method=xai_method,
        base_value=base_value,
        notes=notes if notes else "Brak dodatkowych uwag."
    )

    return explanation

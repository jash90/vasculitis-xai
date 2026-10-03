# Rzetelna poprawa modeli — raport (2026-10-01)

Cel: poprawić jakość modeli predykcji **śmiertelności** i **dializy** tak, by wynik
był obronny metodologicznie. Wszystkie liczby poniżej są odtwarzalne skryptami z
sekcji „Odtworzenie” (seed 42, te same podziały dla wszystkich wariantów).

## TL;DR

| | Było (raportowane) | Jest (rzetelnie) |
|---|---|---|
| Śmiertelność, najlepszy model | AUC **0.933** (stacking, hold-out, wybrany na teście, 142 cechy z cechami „po fakcie”) | AUC **0.815 ± 0.009** (5×5 CV), hold-out **0.805 [0.71–0.89]**, tylko cechy znane przy rozpoznaniu |
| Śmiertelność, model w API (20 cech formularza) | brak rzetelnej oceny, błąd kodowania | CV **0.80**, hold-out **0.77 [0.67–0.86]** |
| Dializa | AUC 0.960 (Random Forest) | **0.649 [0.57–0.73]** wśród pacjentów z odnotowaną wartością (n=240) |

Spadek liczb **nie jest regresją modelu**: poprzednie wyniki były zawyżone przez
przeciek informacji o wyniku. Ablacja (tabela 2) pokazuje dokładnie, ile wnosił
każdy błąd. Względem uczciwego punktu odniesienia (regresja logistyczna na tych
samych cechach, AUC 0.798) wybrany model daje +0.017 AUC, lepszą kalibrację
(Brier 0.116 vs 0.164, nachylenie kalibracji 1.10 vs 0.72) i PR-AUC 0.60 vs 0.57.

## 1. Znalezione problemy (z dowodami)

| # | Problem | Dowód | Wpływ |
|---|---|---|---|
| 1 | **Kod 0 = brak danych, a brak danych ≈ zgon.** 92 pacjentów (po deduplikacji) ma kod 0 w ≥5 kolumnach `Manifestacja_*`; śmiertelność 85% vs 13.6% w reszcie. Ośrodki UMOTP (79% zgonów), KRKAA, WIMSN. To samo dla `Eozynofilia_..._Wartosc == 0` (88% zgonów), `Max_CRP == 0` (100%), `Wiek_rozpoznania == 0`, `Pulsy == 5`. | `src/models/auc_training.py:116` (tylko -1 → NaN), `src/data/preprocessing.py:136-141` | Model uczy się „czy rekord jest kompletny” |
| 2 | **Sklejenie „tak” i „nie”** w modelu API: `(s.fillna(0) > 0)` zamienia 1 (tak) i 2 (nie) na 1; 14/20 cech mierzyło „czy pole wypełniono”. | poprzednia wersja `scripts/retrain_aligned_models.py:107` | Cechy binarne bez treści klinicznej |
| 3 | **Cechy znane dopiero po rozpoznaniu**: `Powiklania_*`, `Leczenie_6m_*`, `Zaostrz_*`, `Przebieg_scalony`, `Liczba_Zaostrzen`, `Czas_Pierwsze_Zaostrzenie`, `Czas_Sterydow`, `Sterydy_Dawka_*`, `Dializa` (dla śmiertelności), `Wiek` (wiek w chwili ostatniego kontaktu/zgonu). | `models/saved/auc_all/auc_metadata.json` (142 cechy = wszystko poza `Kod`) | Największy pojedynczy wkład w zawyżenie |
| 4 | **Wybór modelu na zbiorze testowym** (sortowanie po test AUC). | poprzednio `src/models/auc_training.py:377-386` | Optymistycznie obciążony „best model” |
| 5 | **Hiperparametry bez źródła** (np. `learning_rate=0.02092927267979338`) — wynik nieudokumentowanego strojenia, prawdopodobnie na całym zbiorze. | poprzednio `src/models/auc_training.py:192-269` | Możliwy przeciek |
| 6 | **Legacy pipeline** dopasowuje imputację, filtr korelacji, `SelectKBest(y)` i skaler do całego zbioru przed podziałem. | `src/data/preprocessing.py:373+`, `scripts/train_model.py:68` | Przeciek; oznaczone jako przestarzałe |
| 7 | **Dializa**: brak kodu źródłowego (tylko `.pyc`), cel `Dializa` wypełniony tylko u pacjentów nerkowych (-1: 636, 1: 171, 2: 71, 0: 21). Traktowanie „brak” jako „nie” daje AUC 0.91 — model rozpoznaje, komu wypełniono pole (Kreatynina AUC 0.90 vs 0.65 na znanych). | `scripts/__pycache__/train_dialysis_model.cpython-311.pyc` | Wynik 0.96 był w praktyce artefaktem braków |
| 8 | Duplikaty `Kod`: 1 identyczny wiersz, 3 rozbieżne, `KRKKW0019` z dwoma różnymi wartościami `Zgon` (wykluczony). Outliery: Kreatynina 8486/9017, `Liczba_Zaostrzen` 220, `Opoznienie_Rozpoznia` ~31 tys. dni. | `reports/model_improvement/cohort_notes.json` | Szum, potencjalne błędy wpisu |
| 9 | Brak walidacji między ośrodkami; ośrodek silnie wiąże się z wynikiem. | `loco_per_centre.csv` | Przenoszalność niesprawdzona |

## 2. Ablacja: ile zawyżał każdy błąd (śmiertelność, 5×5 CV, n=893)

| Wariant | LR | XGBoost | CatBoost |
|---|---|---|---|
| (hold-out) obecny stacking `auc_all`, wybrany na teście | — | — | **0.933** (stacking) |
| A. stary sposób: wszystkie kolumny, tylko -1 = brak | 0.886 | 0.906 | 0.909 |
| B. + poprawne kodowanie (0/3 = brak, 1/2 = tak/nie, outliery) | 0.873 | 0.878 | 0.877 |
| C. + tylko cechy znane przy rozpoznaniu (**uczciwy baseline**) | 0.817 | 0.825 | 0.819 |
| D. C bez 92 niekompletnych rekordów (analiza wrażliwości) | 0.742 | 0.734 | 0.719 |
| E. C, leave-one-centre-out | 0.752 | 0.761 | 0.763 |

Model większościowy: AUC 0.5, PR-AUC 0.209, Brier 0.166.
Szczegóły (PR-AUC, Brier, kalibracja, czułość/swoistość, CI): `ablation.csv`.

Wniosek: ~0.03 AUC dawało błędne kodowanie braków, ~0.06 cechy z przyszłości,
~0.02–0.03 wybór na teście. Dodatkowo ~0.08–0.10 AUC uczciwego modelu wynika z
grupy niekompletnych rekordów (D) — to ograniczenie danych, które trzeba opisać.

## 3. Eksperymenty (śmiertelność; `mortality/experiments.csv`)

Dane rozwojowe: 80% kohorty (n=714), hold-out 20% (n=179, 37 zgonów) użyty **raz**.
Reguła przyjęcia (ustalona przed uruchomieniem): ΔAUC > SD między powtórzeniami
**i** DeLong p < 0.05; dla kalibracji/ważenia: niższy Brier bez utraty AUC > SD.
Wybór końcowy: najwyższe średnie AUC, a w granicy 1 SD — model najprostszy.

| Etap | Hipoteza | Wynik | Decyzja |
|---|---|---|---|
| S1 | Rodziny modeli (LR, RF, HistGB, XGB, LGBM, CatBoost, EBM) | 0.78–0.81; żadna różnica vs LR istotna | odrzucone pojedynczo; top-3: XGB, RF, CatBoost |
| S2 | SMOTE w foldach vs wagi klas | bez zysku AUC | odrzucone |
| S2 | Brak wag klas → lepsza kalibracja | Brier 0.139→0.116 (XGB), 0.129→0.113 (CatBoost) | **przyjęte** |
| S3 | IterativeImputer vs mediana | RF +0.016 (p=0.07), CatBoost +0.017 (p=0.04) | przyjęte tylko dla CatBoost |
| S4 | Cechy kliniczne (eGFR bez płci, log-labs, liczba manifestacji) | −0.004…−0.007 | odrzucone |
| S5 | Selekcja L1 w foldach | −0.003…−0.004 | odrzucone |
| S6 | Optuna (nested CV, 25 prób) | XGB −0.018, RF −0.010, CatBoost +0.004 (n.s.) | odrzucone — strojenie przeucza się przy N≈700 |
| S7 | Stacking OOF (XGB+RF+CatBoost) | 0.821 vs 0.823 najlepszego pojedynczego | odrzucone |
| S8 | Kalibracja sigmoid / isotonic | Brier bez zmian, nachylenie 1.10 | przyjęta sigmoid (minimalnie niższy Brier) |

**Wybrany model:** XGBoost (bez wag klas, mediana, 72 cechy baseline, kalibracja
sigmoid) — w granicy 1 SD od CatBoost (0.823), prostszy i szybszy dla SHAP.

### Wyniki końcowe — śmiertelność

| Ocena | ROC-AUC | PR-AUC | Brier | Nachylenie kalibr. | Czułość | Swoistość | NPV |
|---|---|---|---|---|---|---|---|
| 5×5 CV (dev) | 0.815 ± 0.009 [0.78–0.86] | 0.602 | 0.116 | 1.10 | 0.81 | 0.60 | 0.92 |
| Hold-out (raz) | **0.805 [0.71–0.89]** | 0.608 | 0.122 | 1.08 | 0.84 | 0.61 | 0.94 |
| Leave-one-centre-out | 0.747 | 0.468 | 0.143 | 1.05 | — | — | — |
| Bez niekompletnych rekordów (CV) | 0.707 [0.65–0.78] | 0.277 | 0.107 | 1.19 | — | — | — |

Próg decyzyjny (cel: czułość ≥ 0.85) wyznaczany na danych treningowych (OOF),
nigdy na teście; na hold-oucie próg 0.112. Krzywe: `mortality/holdout_curves.png`,
`mortality/dev_oof_curves.png`. AUC per ośrodek (LOCO): od 0.34 (UMOTP — 31/39
zgonów, ośrodek z niekompletnymi rekordami) do 0.98 (`mortality/final_loco_per_centre.csv`).

## 4. Dializa (`dialysis/experiments.csv`, `models/saved_dialysis/`)

Kohorta: pacjenci z odnotowaną wartością `Dializa` ∈ {1, 2} (n=240, 170 dializowanych —
wszyscy to pacjenci nerkowi). Za mało na hold-out → 10×5 CV.

| Ocena | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Random Forest (wybrany, bez wag), 10×5 CV | **0.649 ± 0.013 [0.57–0.73]** | 0.784 | 0.196 |
| LR (referencja) | 0.599 ± 0.022 | 0.780 | 0.251 |
| Leave-one-centre-out | 0.608 | 0.780 | 0.201 |
| Wrażliwość: „brak” = „nie”, cała kohorta | 0.910 — **niewiarygodne** (model rozpoznaje pacjentów nerkowych / wypełnione pole) | | |

Wniosek do pracy: w grupie pacjentów z zajęciem nerek dane z chwili rozpoznania
słabo różnicują przyszłą dializę. Wcześniejsze 0.96 nie było wynikiem predykcji
dializy, tylko wykrywania, u kogo pole wypełniono.

## 5. Zmiany w kodzie i artefaktach

Nowe moduły:
- `src/data/cohort.py` — ładowanie, deduplikacja, semantyczne rekodowanie rejestru, flaga niekompletnych rekordów, ośrodek.
- `src/data/feature_sets.py` — whitelist cech „przy rozpoznaniu” + cechy kliniczne pochodne.
- `src/models/evaluation_protocol.py` — powtarzana CV z progami z foldów treningowych, bootstrap CI, DeLong, kalibracja, LOCO.
- `src/models/model_zoo.py` — pipeline'y (imputacja/OHE/skalowanie/SMOTE/selekcja w foldach).
- `src/models/tuning.py` — Optuna + nested CV.
- Skrypty: `scripts/model_audit.py`, `scripts/run_experiments.py`, `scripts/train_dialysis_models.py`.

Zmienione:
- `scripts/retrain_aligned_models.py` — model API: nowe kodowanie, 3 cechy „po fakcie”
  (`Zaostrz_Wymagajace_Hospital`, `Zaostrz_Wymagajace_OIT`, `Czas_Sterydow`) zastąpione
  przez `Manifestacja_Oddechowy`, `Manifestacja_Nos/Ucho/Gardlo`, `Max_CRP`
  (wybór na danych rozwojowych; różnice między kandydatami w granicach szumu, wybrano
  klinicznie najczytelniejsze); imputacja w modelu; wybór kalibracji po CV.
- `src/api/schemas.py` / `src/api/main.py` — nowe pola formularza, brak wartości → mediana
  z treningu (`models/saved/feature_defaults.json`), stare pola ignorowane (`extra="ignore"`);
  `/predict/auc` rekoduje wejście jak w treningu (`input_coding: registry_recoded_v2`).
- `src/models/auc_training.py` / `scripts/train_auc_models.py` — rekodowanie + whitelist,
  wybór modelu po CV na treningu, neutralne hiperparametry zamiast „magicznych”,
  `--legacy` do porównań. `auc` = 30 cech (MI w treningu), `auc_all` = 72 cechy baseline.
- Frontend (formularz, typy, mapowanie CSV, etykiety XAI) — nowe pola.
- Kopia poprzednich artefaktów: `models/_backup_2026-10-01/` (poza gitem i obrazem Dockera).

Modele w API (hold-out n=179):

| Model | Kalibracja | CV AUC | Hold-out AUC [95% CI] |
|---|---|---|---|
| XGBoost (`best_model.joblib`) | sigmoid | 0.797 ± 0.009 | 0.769 [0.67–0.86] |
| Random Forest | sigmoid | 0.790 ± 0.005 | 0.757 [0.65–0.85] |
| LightGBM | brak | 0.801 ± 0.011 | 0.770 [0.67–0.86] |

## 6. Do pracy magisterskiej

**Co zmieniono i dlaczego.** Zdefiniowano moment predykcji (rozpoznanie choroby) i
usunięto zmienne dostępne dopiero później. Ujednolicono kodowanie rejestru (kod 0/−1/3
jako brak danych, 1/2 jako tak/nie), usunięto duplikaty i wartości niefizjologiczne.
Cały preprocessing przeniesiono do wnętrza walidacji krzyżowej; model wybierano wyłącznie
na danych treningowych, a zbiór testowy oceniono jednokrotnie. Raportowane są przedziały
ufności, kalibracja i walidacja między ośrodkami.

**Uczciwa interpretacja.** Model oparty na danych z chwili rozpoznania osiąga AUC ≈ 0.80
(hold-out 0.805, 95% CI 0.71–0.89), z dobrą kalibracją. Przy walidacji na niewidzianym
ośrodku AUC spada do ≈ 0.75, a po wyłączeniu niekompletnych rekordów do ≈ 0.71 — część
sygnału wynika z różnic w sposobie gromadzenia danych między ośrodkami. Złożone metody
(stacking, strojenie Optuną, SMOTE, selekcja cech) nie dały istotnej poprawy względem
dobrze zregularyzowanych modeli drzewiastych, co jest typowe dla N < 1000.

**Ograniczenia.** Małe N (893, 187 zgonów; dializa 240); brak walidacji zewnętrznej;
brak słownika danych — znaczenie kodów (0, 3, `Pulsy`, `Biopsja_Wynik`, `Plec`) i moment
pomiaru `Max_CRP`, `Kreatynina`, leczenia indukcyjnego ustalono na podstawie rozkładów;
brak informacji o czasie obserwacji (cel binarny zamiast analizy przeżycia).
Do potwierdzenia z właścicielem danych: znaczenie kodu 0 i bloku 92 rekordów, kodowanie
płci (wymagane do eGFR CKD-EPI), czy `Max_CRP` i leczenie dotyczą okresu rozpoznania.

**Do aktualizacji w tekście pracy:** `scripts/generate_thesis_complete.py` (linie ~960–981,
~1058) opisuje `Zaostrz_Wymagajace_OIT`, `Kreatynina`, `Dializa` jako główne czynniki — to
cechy z przyszłości / przecieku; tabele z AUC 0.93 i 0.96 należy zastąpić wynikami z
sekcji 3–4.

## 7. Lepiej zdefiniowany wynik: analiza przeżycia i model „landmark” 6 miesięcy

Skrypt: `scripts/run_survival_experiments.py`, wyniki: `survival/survival_summary.json`.
Czas obserwacji = `Wiek − Wiek_rozpoznania` (zgon lub ostatni kontakt); 51 pacjentów bez
znanego czasu wykluczono (46 z nich to niekompletne rekordy). Mediana czasu do zgonu to
~5 lat, więc binarny `Zgon` miesza zgony wczesne i bardzo późne, a pacjentów krótko
obserwowanych traktuje jak „przeżywających”. Ewaluacja do 10 lat (cenzurowanie
administracyjne), 5×5 CV na danych rozwojowych, hold-out oceniony raz.

**7a. Predykcja przy rozpoznaniu — model przeżycia vs model binarny (te same osoby, n=670, 116 zgonów)**

| Model | Harrell C | Uno C | AUC(t) 1 rok | AUC(t) 3 lata | AUC(t) 5 lat |
|---|---|---|---|---|---|
| XGBoost binarny (obecny, P(zgon) jako ryzyko) | 0.779 ± 0.010 | 0.767 | 0.863 | 0.804 | 0.796 |
| Cox elastic-net | 0.755 ± 0.013 | 0.773 | 0.892 | 0.793 | 0.801 |
| Gradient-boosted Cox | 0.800 ± 0.011 | 0.806 | 0.913 | 0.826 | 0.816 |
| **Random Survival Forest** | **0.815 ± 0.005** | **0.817** | **0.924** | 0.810 | **0.833** |
| Binarny cel 5-letni (cenzurowani <5 lat wykluczeni, n=368) | AUC 0.807 ± 0.017 | | | | |

- RSF vs binarny (sparowany bootstrap Harrell C): **+0.032 [−0.005; 0.070], p=0.09**;
  bez niekompletnych rekordów: **+0.062 [0.017; 0.109], p=0.006** (C 0.707 → 0.773).
- Hold-out (n=172, 32 zgony): RSF AUC(t) **0.94 / 0.93 / 0.87** (1/3/5 lat) vs binarny
  0.87 / 0.89 / 0.84; ale Uno C 0.793 vs 0.819 — wynik mieszany, mały zbiór.
- Cel binarny 5-letni nie pomaga: mniej pacjentów (wykluczanie cenzurowanych) zjada zysk.

**7b. Model landmark w 6. miesiącu (żyjący i obserwowani ≥ 6 mies., n=631, 108 zgonów)**

Cechy: te z rozpoznania + leczenie w 6. miesiącu (`Leczenie_6m_*`, 12 kolumn). Porównanie
na tych samych pacjentach:

| Model | tylko cechy z rozpoznania | + leczenie w 6. mies. | Różnica |
|---|---|---|---|
| Regresja logistyczna (AUC) | 0.747 ± 0.009 | 0.787 ± 0.010 | +0.039, DeLong p=0.008 |
| XGBoost (AUC) | 0.757 ± 0.008 | 0.779 ± 0.009 [0.73–0.83] | +0.022, p=0.019 |
| RSF przeżycie (Harrell C) | 0.793 | **0.830** | +0.037 [0.009; 0.071], p<0.001 |
| XGBoost bez niekompletnych rekordów | 0.687 | 0.715 | +0.028, p=0.023 |
| **Hold-out** (n=156, 27 zgonów): XGBoost AUC / RSF C | 0.766 / 0.834 | 0.765 / 0.835 | **brak różnicy** (p=0.95) |

Sprawdzono, że zmienne 6-miesięczne nie są zastępnikiem braków danych (np. „brak leczenia
w 6. mies.”: 40 osób, śmiertelność 45%, tylko 1 z bloku niekompletnych rekordów).

**Wnioski z sekcji 7.**
1. Najbardziej obiecujące jest **przejście na analizę przeżycia (RSF)**: konsekwentnie wyższa
   dyskryminacja w CV (C 0.815 vs 0.779, AUC(t) do 0.92), istotnie lepsza po wyłączeniu
   niekompletnych rekordów; na hold-oucie lepsze AUC zależne od czasu, ale nie C-index — do
   potwierdzenia na większych danych. W pracy warto raportować C-index i AUC(t), nie tylko AUC.
2. **Model landmark** daje istotny zysk w CV (+0.02–0.04 AUC), ale nie powtarza się na małym
   hold-oucie — prezentować jako hipotezę/analizę dodatkową, nie jako potwierdzoną poprawę.
3. Ustalenie stałego horyzontu (5 lat) bez modelu przeżycia nie poprawia wyników.
4. Ograniczenie: `Wiek` interpretowany jako wiek przy zgonie / ostatnim kontakcie — do
   potwierdzenia z właścicielem danych (od tego zależy poprawność czasu obserwacji).

## 8. Dobór danych: które cechy i którzy pacjenci

Skrypt: `scripts/run_data_selection.py`, wyniki: `data_selection/*.csv`, `data_selection/summary.json`.
Selekcja cech dopasowywana **wewnątrz każdego foldu** (`src/models/selectors.py`), 3×5 CV na
danych rozwojowych, porównania sparowane (DeLong / bootstrap C), hold-out oceniony raz.
Modele: XGBoost binarny (sekcja 3) i Random Survival Forest (sekcja 7).

**8a. Dobór cech** (różnica vs wszystkie 72 cechy; dodatnia = lepiej)

| Konfiguracja | Liczba cech | XGBoost AUC | Δ (p) | RSF C-index | Δ (p) |
|---|---|---|---|---|---|
| Wszystkie cechy z rozpoznania | 72 | 0.809 ± 0.013 | — | 0.816 ± 0.001 | — |
| Bez serologii (ANCA, anty-PR3/MPO) | 59 | 0.812 | +0.003 (0.20) | **0.820** | +0.003 (0.17) |
| Bez biopsji | 60 | 0.812 | +0.002 (0.49) | 0.814 | −0.001 (0.92) |
| Top-45 (informacja wzajemna, w foldzie) | 45 | 0.807 | 0.000 (0.98) | 0.805 | −0.006 (0.32) |
| Top-20 | 20 | 0.801 | −0.004 (0.70) | 0.782 | −0.018 (0.11) |
| 20 cech formularza API | 20 | 0.793 | −0.018 (0.06) | 0.818 | +0.002 (0.72) |
| Zestaw kliniczny (FFS-podobny) | 14 | 0.787 | −0.022 (0.08) | 0.798 | −0.017 (0.26) |
| Top-5 | 5 | 0.778 | −0.026 (0.12) | 0.717 | −0.061 (0.004) |
| Stability selection (L1, bootstrap) | ~10 | 0.768 | −0.039 (0.009) | 0.744 | −0.053 (0.004) |
| Bez demografii i wywiadu | 64 | 0.778 | −0.033 (0.014) | 0.750 | −0.066 (<0.001) |
| Bez manifestacji narządowych | 59 | 0.798 | −0.012 (0.13) | 0.806 | −0.010 (0.004) |

- Żaden podzbiór cech nie jest istotnie lepszy od pełnego zestawu; agresywna redukcja
  (≤ 20 cech, stability selection) istotnie pogarsza wyniki.
- Najważniejsze grupy: **demografia/wywiad** (wiek, opóźnienie rozpoznania, palenie, praca)
  i **manifestacje narządowe**; serologia, biopsja i typ zapalenia wnoszą niewiele.
- Najstabilniej wybierane cechy (bootstrap L1, `stability_frequencies.csv`): opóźnienie
  rozpoznania, brak biopsji, zajęcie układu oddechowego, GPA, kategoria zawodowa (proxy wieku),
  zajęcie nerek, palenie.

**8b. Dobór pacjentów** (kryteria niezależne od wyniku; każda populacja oceniana w sobie)

| Populacja | n (zgony) | XGBoost AUC | XGBoost PR-AUC | RSF C-index |
|---|---|---|---|---|
| Wszyscy | 714 (21%) | 0.809 | 0.618 | 0.816 |
| Bez niekompletnych rekordów | 636 (13%) | 0.704 | 0.280 | 0.778 |
| ≤ 20% braków w cechach | 634 (14%) | 0.718 | 0.289 | 0.786 |
| Ośrodki ≥ 20 pacjentów | 617 (18%) | 0.773 | 0.523 | 0.784 |

Zawężanie populacji **obniża** wyniki: usuwane są głównie „łatwe” zgony z niekompletnych
rekordów. Model przeżycia (RSF) jest wyraźnie odporniejszy na to zawężenie niż model binarny
(C 0.78 vs AUC 0.70) — kolejny argument za analizą przeżycia.

**8c. Hold-out (raz, n=179 / 172 z czasem obserwacji)**

| Konfiguracja | XGBoost AUC [95% CI] | RSF Harrell C | RSF Uno C | RSF AUC(t) 1/3/5 lat |
|---|---|---|---|---|
| Wszystkie cechy | 0.821 [0.74–0.90] | 0.851 | 0.793 | 0.94 / 0.93 / 0.87 |
| Bez serologii | 0.802 [0.71–0.88] (DeLong vs wszystkie p=0.014) | 0.854 (p=0.72) | 0.802 | 0.94 / 0.93 / 0.88 |
| Top-20 | 0.782 [0.69–0.86] | 0.823 | 0.769 | 0.90 / 0.91 / 0.85 |

**Wnioski z sekcji 8.** Dobór cech i pacjentów **nie poprawia** wyniku w sposób, który
potwierdza się na hold-oucie: drobne zyski w CV (+0.003) to szum, a na hold-oucie usunięcie
serologii nawet pogarsza model binarny. Zostajemy przy **pełnym zestawie cech z chwili
rozpoznania**; najlepszy wynik daje Random Survival Forest (hold-out C 0.85, AUC(t) 0.87–0.94).
Analiza grup cech jest za to wartościowa merytorycznie (sekcja do dyskusji w pracy).
Uwaga: hold-out był już używany w sekcjach 3 i 7 — wyniki traktować jako potwierdzenie,
nie podstawę wyboru (konfiguracje wybrano wyłącznie na podstawie CV).

## Odtworzenie

```bash
venv/bin/python scripts/model_audit.py                       # baseline + ablacja
venv/bin/python scripts/run_experiments.py --task mortality  # ~35 min
venv/bin/python scripts/run_experiments.py --task dialysis --repeats 10
venv/bin/python scripts/train_dialysis_models.py --all-patients-sensitivity
venv/bin/python scripts/retrain_aligned_models.py            # modele API (20 cech)
venv/bin/python scripts/train_auc_models.py                  # models/saved/auc
venv/bin/python scripts/train_auc_models.py --n-features 0 --output-dir models/saved/auc_all
venv/bin/python scripts/run_survival_experiments.py --repeats 5   # sekcja 7, ~2 min
venv/bin/python scripts/run_data_selection.py --repeats 3       # sekcja 8, ~5 min
venv/bin/python -m pytest tests/ -q
```

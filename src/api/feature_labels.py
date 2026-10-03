"""Polish display labels for model feature names (shared by API responses)."""

FEATURE_LABELS_PL = {
    "Wiek_rozpoznania": "Wiek w chwili rozpoznania",
    "Opoznienie_Rozpoznia": "Opóźnienie rozpoznania",
    "Manifestacja_Miesno-Szkiel": "Zajęcie mięśniowo-szkieletowe",
    "Manifestacja_Skora": "Zajęcie skóry",
    "Manifestacja_Wzrok": "Zajęcie narządu wzroku",
    "Manifestacja_Sercowo-Naczyniowy": "Zajęcie serca/naczyń",
    "Manifestacja_Pokarmowy": "Zajęcie układu pokarmowego",
    "Manifestacja_Nerki": "Zajęcie nerek",
    "Manifestacja_Moczowo-Plciowy": "Zajęcie układu moczowo-płciowego",
    "Manifestacja_Zajecie_CSN": "Zajęcie ośrodkowego układu nerwowego",
    "Manifestacja_Neurologiczny": "Zajęcie obwodowego układu nerwowego",
    "Manifestacja_Oddechowy": "Zajęcie układu oddechowego",
    "Manifestacja_Nos/Ucho/Gardlo": "Zajęcie nosa/ucha/gardła",
    "Liczba_Zajetych_Narzadow": "Liczba zajętych narządów",
    "Kreatynina": "Kreatynina",
    "Max_CRP": "CRP",
    "Pulsy": "Pulsy sterydowe IV",
    "Plazmaferezy": "Plazmaferezy",
    "Eozynofilia_Krwi_Obwodowej_Wartosc": "Eozynofilia krwi obwodowej",
    "Biopsja_Wynik": "Biopsja wykonana",
}


def label(feature: str) -> str:
    return FEATURE_LABELS_PL.get(feature, feature.replace("_", " "))

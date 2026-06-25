from __future__ import annotations
from enum import Enum


class Language(str, Enum):
    AF = "af" # AFRIKAANS
    AM = "am" # AMHARIC
    AR = "ar" # ARABIC
    AS = "as" # ASSAMESE
    AZ = "az" # AZERBAIJANI

    EN = "en" # ENGLISH
    FR = "fr" # FRENCH
    GE = "de" # GERMAN
    SP = "es" # SPANISH
    CH = "zh" # CHINESE

    JA = "ja" # JAPANESE
    KO = "ko" # KOREAN
    IT = "it" # ITALIAN
    PO = "pt" # PORTUGUESE
    RU = "ru" # RUSSIAN
    # ...


    # --- Properties ---   

    @property
    def code(self) -> str:
        return self.value
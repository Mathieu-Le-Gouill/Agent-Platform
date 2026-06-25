from enum import Enum

class Language(str, Enum):
    AFRIKAANS = "af"
    AMHARIC = "am"
    ARABIC = "ar"
    ASSAMESE = "as"
    AZERBAIJANI = "az"

    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    CHINESE = "zh"

    JAPANESE = "ja"
    KOREAN = "ko"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    # ...


    # --- Properties ---

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()
    

    @property
    def code(self) -> str:
        return self.value
import locale
import json
import os


def load_language_list(language):
    try:
        with open(f"./i18n/{language}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Failed to load language file for {language}. Check if the correct .json file exists.")
        return {}


class I18nAuto:
    def __init__(self, language=None):
        language = language or locale.getdefaultlocale()[0]
        if not self._language_exists(language):
            language = "en_US"
        self.language_map = load_language_list(language)
        self.language = language

    @staticmethod
    def _language_exists(language):
        return os.path.exists(f"./i18n/{language}.json")

    def __call__(self, key):
        return self.language_map.get(key, key)

    def print(self):
        print(f"Using Language: {self.language}")
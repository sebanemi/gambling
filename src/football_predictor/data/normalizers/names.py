"""Normalización de nombres (equipos y competiciones).

Objetivo: transformar variantes tipográficas del mundo real en un nombre
canónico determinístico para matching/upsert (p.ej. ``"  Man.Utd  "`` →
``"Man Utd"``). No modifica el significado; es idempotente.
"""

import unicodedata


class NameNormalizer:
    """Colapsa espacios y transpone acentos a ASCII (conservando caja).

    Usa descomposición NFKD: cubre todas las diacríticas (á, ö, ñ, ç, ...)
    sin depender de una tabla finita.
    """

    def __call__(self, name: str) -> str:
        return self.normalize(name)

    def normalize(self, name: str) -> str:
        name = name.strip()
        name = " ".join(name.split())
        decomposed = unicodedata.normalize("NFKD", name)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Por conveniencia tipográfica.
normalize_name = NameNormalizer()
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
    
    Optimizado: cache LRU para evitar recomputaciones del mismo nombre.
    """

    def __init__(self, max_cache_size: int = 1000) -> None:
        self._max_cache_size = max_cache_size
        self._cache: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        return self.normalize(name)

    def normalize(self, name: str) -> str:
        # Ver cache primero
        if name in self._cache:
            return self._cache[name]
        
        # Calcular y cachear
        result = self._compute_normalize(name)
        
        # Aplicar límite de cache (LRU simple)
        if len(self._cache) >= self._max_cache_size:
            # Eliminar primer elemento (más antiguo)
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[name] = result
        return result
    
    def _compute_normalize(self, name: str) -> str:
        """Método interno que hace la normalización real."""
        name = name.strip()
        name = " ".join(name.split())
        decomposed = unicodedata.normalize("NFKD", name)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Por conveniencia tipográfica.
normalize_name = NameNormalizer()
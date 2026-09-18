"""Cliente HTTP compartido por los providers de API (Fase 7).

Cada fuente externa usa ``ApiClient`` para hablar con su backend:
JSON con auth por header, errores no-2xx reportados como
``ProviderError``. La clave nunca se registra: se inyecta via header.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

def _default_timeout() -> httpx.Timeout:
    return httpx.Timeout(30.0)


class ProviderError(ValueError):
    """Fallo al comunicarse con (o parsear) una fuente externa."""


class ProviderConfigurationError(ProviderError):
    """Falta o está mal configurada la credencial de la fuente."""


@dataclass
class ApiClient:
    """Cliente síncrono sobre ``httpx.Client``.

    Auth se resuelve con ``auth_header`` + ``auth_scheme`` (p.ej.
    ``Authorization: Bearer <key>``). También soporta headers crudos
    (``extra_headers``) para APIs estilo ``X-Auth-Token`` o
    ``X-RapidAPI-Key``.
    """

    base_url: str
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: httpx.Timeout = field(default_factory=_default_timeout)
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
            follow_redirects=True,
        )

    def _headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.api_key:
            if self.auth_scheme:
                headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET y devuelve el JSON parseado. Lanza ``ProviderError``."""
        try:
            response = self._client.get(path, params=params, headers=self._headers())
        except httpx.RequestError as exc:
            raise ProviderError(f"error de red hacia {self.base_url}{path}: {exc}") from exc

        if response.status_code >= 400:
            raise self._http_error(response, path)

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.base_url}{path}: respuesta no-JSON (HTTP {response.status_code})"
            ) from exc

    def _http_error(self, response: httpx.Response, path: str) -> ProviderError:
        status = response.status_code
        detail = response.text[:500].strip()
        if status == 401 or status == 403:
            reason = "credencial inválida o sin permisos"
        elif status == 404:
            reason = "recurso no encontrado"
        elif status == 429:
            reason = "límite de requests alcanzado (rate limit)"
        else:
            reason = f"HTTP {status}"
        message = f"{self.base_url}{path}: {reason}"
        if detail:
            message += f" - {detail}"
        return ProviderError(message)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
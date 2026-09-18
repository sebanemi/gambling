"""Fábrica de providers externos (Fase 7).

Centraliza la construcción de cada fuente a partir de ``Settings`` y hace
explicito el modo ``--list`` (catalogo de competiciones) por provider.
"""

from __future__ import annotations

from typing import Any, Callable

from football_predictor.config.settings import Settings
from football_predictor.data.providers.base import ApiClient, ProviderConfigurationError
from football_predictor.data.providers.five_dollar import FiveDollarProvider
from football_predictor.data.providers.football98 import Football98Provider
from football_predictor.data.providers.football_charts import FootballChartsProvider
from football_predictor.data.providers.football_data import FootballDataOrgProvider
from football_predictor.data.providers.playerelo import PlayerEloProvider
from football_predictor.data.providers.sofascore import SofaScoreProvider
from football_predictor.data.providers.standings import FootballStandingsProvider

PROVIDER_NAMES = (
    "football-data",
    "football-charts",
    "football98",
    "five-dollar",
    "sofascore",
    "standings",
    "playerelo",
)


def _client(
    settings: Settings,
    base_url: str,
    *,
    api_key: str = "",
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    extra_headers: dict[str, str] | None = None,
) -> ApiClient:
    return ApiClient(
        base_url=base_url,
        api_key=api_key,
        auth_header=auth_header,
        auth_scheme=auth_scheme,
        extra_headers=extra_headers or {},
    )


def _require_key(provider: str, key: str) -> None:
    if not key:
        raise ProviderConfigurationError(
            f"{provider}: falta la API key (configurar en .env)"
        )


def build_provider(
    settings: Settings,
    name: str,
    *,
    competition: str | None = None,
    season: str | None = None,
) -> Any:
    """Construye el provider pedido. ``competition``/``season`` son el
    identificador de competición/liga y la temporada según la fuente."""
    if name == "football-data":
        _require_key("football-data.org", settings.football_data_api_key)
        return FootballDataOrgProvider(
            client=_client(
                settings,
                settings.football_data_base_url,
                api_key=settings.football_data_api_key,
                auth_header="X-Auth-Token",
                auth_scheme="",
            ),
            competition=competition or "PL",
            season=season,
        )
    if name == "football-charts":
        return FootballChartsProvider(
            client=_client(
                settings,
                settings.football_charts_base_url,
                api_key=settings.football_charts_api_key,
            ),
            league_key=competition or "premier",
            season=season,
        )
    if name == "football98":
        _require_key("football98", settings.football98_api_key)
        return Football98Provider(
            client=_client(
                settings,
                "https://football98.p.rapidapi.com",
                extra_headers={
                    "x-rapidapi-key": settings.football98_api_key,
                    "x-rapidapi-host": settings.football98_host,
                },
            ),
            championship=competition or "premierleague",
            season=season,
        )
    if name == "five-dollar":
        _require_key("5dollarfootballapi", settings.five_dollar_api_key)
        return FiveDollarProvider(
            client=_client(
                settings,
                settings.five_dollar_base_url,
                api_key=settings.five_dollar_api_key,
            ),
            league_id=competition,
            season=season,
        )
    if name == "standings":
        return FootballStandingsProvider(
            client=_client(settings, settings.standings_base_url),
            league_id=competition,
            season=season,
        )
    if name == "playerelo":
        _require_key("playerelo.football", settings.playerelo_api_key)
        return PlayerEloProvider(
            client=_client(
                settings,
                settings.playerelo_base_url,
                api_key=settings.playerelo_api_key,
            ),
            league=competition,
        )
    if name == "sofascore":
        return SofaScoreProvider(
            client=_client(
                settings,
                settings.sofascore_base_url,
                extra_headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.sofascore.com/",
                    "Accept": "application/json",
                },
            ),
            tournament=competition or "la-liga",
            season=season,
        )
    raise ValueError(f"provider desconocido: {name!r} (usar {PROVIDER_NAMES})")


def list_competitions(settings: Settings, name: str) -> list[str]:
    """Devuelve lineas de texto legibles con el catalogo de la fuente."""
    if name == "football-data":
        provider = build_provider(settings, name)
        return [
            f"{c.code}\t{c.name}\t{c.country}\t{c.type}"
            for c in provider.available_competitions()
        ]
    if name == "football-charts":
        provider = build_provider(settings, name)
        return [
            f"{l.league}\t{l.name}\t{l.country}\t{','.join(l.seasons or [])}"
            for l in provider.available_leagues()
        ]
    if name == "football98":
        _require_key("football98", settings.football98_api_key)
        provider = build_provider(settings, name)
        return provider.available_competitions()
    if name == "five-dollar":
        provider = build_provider(settings, name)
        return [f"{l.name}\t{l.country}" for l in provider.available_leagues()]
    if name == "sofascore":
        provider = build_provider(settings, name)
        return [f"{c}\t{c}" for c in provider.available_competitions()]
    if name == "standings":
        provider = build_provider(settings, name)
        return [f"{l.id}\t{l.name}" for l in provider.available_leagues()]
    if name == "playerelo":
        provider = build_provider(settings, name)
        return [f"{l.name}\t{l.country}" for l in provider.available_leagues()]
    raise ValueError(f"provider desconocido: {name!r}")


def provider_has_matches(name: str) -> bool:
    """True si la fuente aporta partidos historicos con marcador."""
    return name in ("football-data", "football-charts", "football98", "five-dollar", "sofascore")
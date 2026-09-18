"""Helpers para simular respuestas HTTP de los providers (MockTransport)."""

from __future__ import annotations

from typing import Any, Callable

from httpx import MockTransport, Request, Response

from football_predictor.data.providers.base import ApiClient

Handler = Callable[[Request], Response]


def make_router(
    routes: dict[str, Any | Handler],
) -> Handler:
    """Router por prefijo de path: payload estático o handler por request."""

    def handler(request: Request) -> Response:
        for prefix, value in routes.items():
            if str(request.url).startswith(prefix):
                if callable(value):
                    return value(request)
                return Response(200, json=value)
        return Response(404, json={"message": f"no route for {request.url}"})

    return handler


def make_client(routes: dict[str, Any | Handler], **kwargs: Any) -> ApiClient:
    return ApiClient(
        base_url=kwargs.pop("base_url", "https://example.test"),
        transport=MockTransport(make_router(routes)),
        **kwargs,
    )
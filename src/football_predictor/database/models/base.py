"""Base declarativa única para todos los modelos ORM (Fase 2)."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base de SQLAlchemy 2.0 para los modelos de la aplicación."""
"""Коллекторы данных. Каждый модуль разделяет fetch (байты из сети) и parse
(байты -> структуры), чтобы тесты работали на fixture'ах без сети."""
from .errors import ProviderError

__all__ = ["ProviderError"]

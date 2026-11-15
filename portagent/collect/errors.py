class ProviderError(RuntimeError):
    """Сбой одного провайдера (сеть, API, парсинг). Не роняет пайплайн целиком."""

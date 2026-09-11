"""Provider error taxonomy."""


class ProviderError(RuntimeError):
    pass


class AuthError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


class ServerError(ProviderError):
    pass


class NetworkError(ProviderError):
    pass

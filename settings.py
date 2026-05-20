from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    loki_endpoint: str = Field(
        default="https://dashboard.404.xyz/loki/loki/api/v1/push", description="Loki push API endpoint."
    )
    loki_username: str = Field(default="", description="Username for Loki basic auth.")
    loki_password: SecretStr = Field(default=SecretStr(""), description="Password for Loki basic auth.")
    loki_enabled: bool = Field(default=True, description="Enable periodic log shipping to Loki.")
    loki_push_interval_seconds: float = Field(
        default=5.0, description="Interval in seconds between periodic log pushes to Loki."
    )
    loki_batch_size: int = Field(default=500, description="Max log lines per stream in one Loki push request.")
    loki_timeout_seconds: float = Field(default=10.0, description="HTTP timeout for Loki push requests.")
    prometheus_push_gateway_url: str = Field(
        default="", description="URL to send Prometheus metrics to the push gateway."
    )
    prometheus_push_gateway_username: str = Field(
        default="", description="Username for Basic Authentication to the push gateway."
    )
    prometheus_push_gateway_password: SecretStr = Field(
        default=SecretStr(""), description="Password for Basic Authentication to the push gateway."
    )
    test_run: bool = Field(
        default=False,
        description=(
            "Enable test mode: each /generate request cycles through synthetic errors "
            "(HTTP 500, 503, 429) and an occasional success so all error metrics can be "
            "exercised without real GPU work. Set via env TEST_RUN=true."
        ),
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore

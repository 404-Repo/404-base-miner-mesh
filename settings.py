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
    r2_account_id: str = Field(default="", description="Cloudflare R2 account ID.")
    r2_access_key_id: str = Field(default="", description="Cloudflare R2 access key ID.")
    r2_secret_access_key: SecretStr = Field(default=SecretStr(""), description="Cloudflare R2 secret access key.")
    r2_bucket_name: str = Field(default="", description="Cloudflare R2 bucket name for prompt image storage.")
    r2_public_url_base: str = Field(default="", description="Base public URL for R2 objects (e.g. https://pub-xxx.r2.dev).")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()  # type: ignore

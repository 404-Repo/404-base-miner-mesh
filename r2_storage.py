import asyncio

from loguru import logger


def _upload_sync(
    *,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    key: str,
    data: bytes,
    content_type: str,
) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )
    client.put_object(Bucket=bucket_name, Key=key, Body=data, ContentType=content_type)


async def upload_prompt_image(
    *,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
    public_url_base: str,
    key: str,
    data: bytes,
    content_type: str = "image/png",
) -> str | None:
    """Upload a prompt image to Cloudflare R2, return the public URL or None on failure."""
    if not all([account_id, access_key_id, secret_access_key, bucket_name]):
        return None
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: _upload_sync(
                account_id=account_id,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                bucket_name=bucket_name,
                key=key,
                data=data,
                content_type=content_type,
            ),
        )
        base = (
            public_url_base.rstrip("/")
            if public_url_base
            else f"https://{account_id}.r2.cloudflarestorage.com/{bucket_name}"
        )
        return f"{base}/{key}"
    except Exception as e:
        logger.warning(f"Failed to upload prompt image to R2: {e}")
        return None

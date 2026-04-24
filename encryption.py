"""
Encryption module — AES-128 via Fernet (symmetric encryption).

How Fernet works:
    Fernet uses AES-128-CBC for encryption and HMAC-SHA256 for
    authentication. Every encrypted value is a base64 URL-safe token
    that includes: version byte | timestamp | IV | ciphertext | HMAC.

    The HMAC means tampering with the ciphertext is detectable —
    decryption will raise InvalidToken if the data was modified.

Key management:
    1. Generate a key once:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    2. Store it in .env:     ENCRYPTION_KEY=<your key>
    3. Never commit .env to git.
    4. In production: store the key in a secrets manager
       (AWS Secrets Manager, HashiCorp Vault, etc.) not in .env.

Ephemeral key warning:
    If ENCRYPTION_KEY is missing, a new key is generated at startup.
    This means encrypted data from a previous run CANNOT be decrypted
    after restart — all history is permanently lost. This is only
    acceptable in development.
"""
import os
from typing import Union

from cryptography.fernet import Fernet, InvalidToken
from logger import get_logger

logger = get_logger()


class Encryption:
    """Symmetric AES encryption / decryption via Fernet."""

    def __init__(self, key: Union[str, bytes, None] = None):
        if key is None:
            key = os.getenv("ENCRYPTION_KEY")

        if key is None:
            self.key = Fernet.generate_key()
            logger.warning(
                "ENCRYPTION | No ENCRYPTION_KEY env var found. "
                "Generated an ephemeral key — encrypted data will be "
                "UNRECOVERABLE after server restart. "
                f"For production, add to .env: ENCRYPTION_KEY={self.key.decode()}"
            )
        else:
            if isinstance(key, str):
                key = key.encode()
            self.key = key
            logger.info("ENCRYPTION | Key loaded from environment")

        self.cipher = Fernet(self.key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.
        Returns a base64 Fernet token (safe to store in TEXT columns).
        """
        if not isinstance(plaintext, str):
            plaintext = str(plaintext)
        return self.cipher.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a Fernet token back to plaintext.
        Raises InvalidToken if the token was tampered with or the key is wrong.
        """
        if not isinstance(ciphertext, str):
            ciphertext = str(ciphertext)
        try:
            return self.cipher.decrypt(ciphertext.encode()).decode()
        except InvalidToken as e:
            logger.error(
                "ENCRYPTION | Decryption failed — token invalid or wrong key. "
                "If you rotated ENCRYPTION_KEY, old records cannot be decrypted."
            )
            raise

    def encrypt_dict(self, data: dict, keys_to_encrypt: list[str]) -> dict:
        """Encrypt specific fields in a dict, leave others unchanged."""
        result = data.copy()
        for key in keys_to_encrypt:
            if key in result and result[key] is not None:
                result[key] = self.encrypt(str(result[key]))
        return result

    def decrypt_dict(self, data: dict, keys_to_decrypt: list[str]) -> dict:
        """Decrypt specific fields in a dict, leave others unchanged."""
        result = data.copy()
        for key in keys_to_decrypt:
            if key in result and result[key] is not None:
                try:
                    result[key] = self.decrypt(result[key])
                except (InvalidToken, Exception) as e:
                    logger.warning(
                        f"ENCRYPTION | Failed to decrypt field '{key}': {e}"
                    )
        return result


# ── Global singleton ──────────────────────────────────────────────────────────
_encryption_instance: Union[Encryption, None] = None


def get_encryption() -> Encryption:
    """
    Return the global Encryption instance, creating it on first call.
    Called by database_pool — initialised once at startup.
    """
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = Encryption()
    return _encryption_instance

"""
Encryption module for data at rest using Fernet AES encryption.
"""
from cryptography.fernet import Fernet
import os
from typing import Union


class Encryption:
    """Handle encryption/decryption of sensitive data."""
    
    def __init__(self, key: Union[str, bytes, None] = None):
        """
        Initialize encryption with a key.
        
        If no key provided, attempts to load from ENCRYPTION_KEY env var.
        If that's not set, generates a new key (not recommended for production).
        """
        if key is None:
            key = os.getenv('ENCRYPTION_KEY')
        
        if key is None:
            # Generate a new key (only suitable for development)
            self.key = Fernet.generate_key()
            print(
                "[encryption] WARNING: No ENCRYPTION_KEY env var found. "
                "Generated ephemeral key. Store this in .env for production:\n"
                f"ENCRYPTION_KEY={self.key.decode()}"
            )
        else:
            if isinstance(key, str):
                key = key.encode()
            self.key = key
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        if not isinstance(plaintext, str):
            plaintext = str(plaintext)
        
        ciphertext = self.cipher.encrypt(plaintext.encode())
        return ciphertext.decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext and return plaintext string."""
        if not isinstance(ciphertext, str):
            ciphertext = str(ciphertext)
        
        plaintext = self.cipher.decrypt(ciphertext.encode())
        return plaintext.decode()
    
    def encrypt_dict(self, data: dict, keys_to_encrypt: list[str]) -> dict:
        """Encrypt specific keys in a dictionary."""
        encrypted = data.copy()
        for key in keys_to_encrypt:
            if key in encrypted and encrypted[key] is not None:
                encrypted[key] = self.encrypt(str(encrypted[key]))
        return encrypted
    
    def decrypt_dict(self, data: dict, keys_to_decrypt: list[str]) -> dict:
        """Decrypt specific keys in a dictionary."""
        decrypted = data.copy()
        for key in keys_to_decrypt:
            if key in decrypted and decrypted[key] is not None:
                try:
                    decrypted[key] = self.decrypt(decrypted[key])
                except Exception as e:
                    print(f"[encryption] Failed to decrypt key '{key}': {e}")
        return decrypted


# Global encryption instance (initialized on first use)
_encryption_instance = None


def get_encryption() -> Encryption:
    """Get or create the global encryption instance."""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = Encryption()
    return _encryption_instance

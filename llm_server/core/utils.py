import secrets
import string

def generate_id(prefix: str, length: int = 24) -> str:
    """Generates a random ID similar to OpenAI's format."""
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}-{random_part}"
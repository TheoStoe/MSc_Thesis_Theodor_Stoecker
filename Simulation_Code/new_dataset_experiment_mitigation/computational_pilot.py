"""Safe compatibility entry point; use the gender campaign's explicit interface."""
from .campaign import main

if __name__ == "__main__":
    raise SystemExit(main())

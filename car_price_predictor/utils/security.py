"""
Security headers middleware utility for CarPricePredictor.
"""
from __future__ import annotations

from flask import Flask, Response


def init_security_headers(app: Flask) -> None:
    """
    Registers an after_request hook on the Flask application to inject
    standard HTTP security headers.
    """
    @app.after_request
    def set_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response

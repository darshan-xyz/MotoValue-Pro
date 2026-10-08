"""
Production WSGI Entrypoint for CarPricePredictor.
Compatible with Gunicorn, Waitress, uWSGI, and other production WSGI servers.
"""
from car_price_predictor import create_app

# Export callable application instance
app = create_app("production")
application = app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

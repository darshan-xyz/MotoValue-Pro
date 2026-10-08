"""
CarPricePredictor Application Entrypoint.
Provides backward compatibility and local development runner.
"""
from car_price_predictor import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
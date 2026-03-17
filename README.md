# Stocksight API

A simple FastAPI application for managing stock data.

## Features

- Get all stocks
- Get specific stock by symbol
- Create new stocks
- Update existing stocks
- Delete stocks
- Health check endpoint

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/stocks` | Get all stocks |
| GET | `/stocks/{symbol}` | Get specific stock |
| POST | `/stocks` | Create new stock |
| PUT | `/stocks/{symbol}` | Update stock |
| DELETE | `/stocks/{symbol}` | Delete stock |
| GET | `/health` | Health check |

## Example Usage

```bash
# Get all stocks
curl http://localhost:8000/stocks

# Get specific stock
curl http://localhost:8000/stocks/AAPL

# Create new stock
curl -X POST http://localhost:8000/stocks \
  -H "Content-Type: application/json" \
  -d '{"symbol":"MSFT","price":380.5,"change":1.5,"change_percent":0.39}'

# Update stock
curl -X PUT http://localhost:8000/stocks/AAPL \
  -H "Content-Type: application/json" \
  -d '{"price":155.0,"change":4.75,"change_percent":3.16}'

# Delete stock
curl -X DELETE http://localhost:8000/stocks/AAPL
```

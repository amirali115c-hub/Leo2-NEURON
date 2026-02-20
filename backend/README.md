# NEURON v2.0 Backend

**Status:** IN PROGRESS
**Created:** Fri 2026-02-20 05:25 GMT+5

## Dependencies

```bash
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
python-dotenv>=1.0.0
httpx>=0.26.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Open `http://localhost:8000/docs` for Swagger UI.

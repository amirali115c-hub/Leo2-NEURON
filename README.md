# NEURON v2.0 - Self-Learning AI Agent

**Status:** IN PROGRESS
**Version:** 2.0.0

A self-learning AI agent that autonomously learns from interactions, extracts concepts, and continuously improves its understanding.

## Features

### 9 Knowledge Domains
- Science ⚗️ - Dr. Elena Vasquez
- Technology 💻 - Kai Chen  
- Philosophy ∞ - Prof. Marcus Webb
- Arts 🎨 - Isabelle Morel
- History 📜 - Dr. Amara Osei
- Math ∑ - Dr. Priya Nair
- Language 🗣️ - Prof. Leo Bauer
- Psychology 🧩 - Dr. Sara Khalil
- General ✦ - NEURON Core

### 5 Learning Strategies
- **Chain-of-Thought (CoT)** - Step-by-step reasoning
- **Tree of Thought (ToT)** - Multi-path hypothesis branching
- **Synthesis** - Cross-domain connections
- **Socratic** - Question-led discovery
- **Analysis** - Deconstruct to fundamentals

### Self-Learning Capabilities
- ✅ Concept extraction
- ✅ Relationship mapping
- ✅ Autonomous curiosity generation
- ✅ Cross-domain synthesis
- ✅ Goal tracking
- ✅ XP/Level system
- ✅ Self-assessment analytics

## Quick Start

### Option 1: One-Click Launcher (Windows)
```batch
START.bat
```

### Option 2: Manual

**1. Start Backend**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**2. Start Frontend** (separate terminal)
```bash
cd frontend
npm install
npm run dev
```

### Option 3: Docker
```bash
docker-compose up -d
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | React Dashboard |
| API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger Documentation |
| Health | http://localhost:8000/api/health | Health Check |

## API Endpoints

### Core Learning
- `POST /api/neuron/learn` - Process learning interaction
- `POST /api/neuron/chat` - Chat with context
- `POST /api/neuron/query` - Query knowledge base

### Knowledge Management
- `GET /api/neuron/kb` - Get knowledge entries
- `GET /api/neuron/concepts` - Get concepts
- `GET /api/neuron/edges` - Get relationships
- `GET /api/neuron/syntheses` - Get cross-domain syntheses

### Self-Assessment
- `GET /api/neuron/stats` - System statistics
- `GET /api/neuron/analytics/curve` - Learning curve
- `GET /api/neuron/analytics/domains` - Domain breakdown

### Goals & Curiosity
- `GET /api/neuron/goals` - Get goals
- `POST /api/neuron/goals` - Create goal
- `GET /api/neuron/curiosity` - Get curiosity questions
- `GET /api/neuron/hypotheses` - Get hypotheses

## Configuration

### Environment Variables

```bash
# Database
NEURON_DB_PATH=./data/neuron.db

# LLM Provider (ollama, nvidia, anthropic, openai, siliconflow)
NEURON_DEFAULT_PROVIDER=ollama

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# NVIDIA
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_MODEL=qwen/qwen3.5-397b-a17b
```

Copy `backend/.env.example` to `backend/.env` and configure.

## Project Structure

```
Leo2-NEURON/
├── backend/
│   ├── main.py           # FastAPI application
│   ├── models.py         # SQLAlchemy models
│   ├── llm_client.py     # LLM provider integration
│   ├── requirements.txt  # Python dependencies
│   └── .env.example      # Environment template
├── frontend/
│   ├── src/
│   │   ├── App.jsx      # React dashboard
│   │   ├── index.css    # Styles
│   │   └── main.jsx     # Entry point
│   ├── index.html       # HTML template
│   ├── package.json     # NPM dependencies
│   └── vite.config.js   # Vite configuration
├── data/                  # Database storage
├── START.bat            # Windows launcher
└── README.md            # This file
```

## Supported LLM Providers

| Provider | Status | Notes |
|----------|--------|-------|
| Ollama | ✅ Primary | Local, free, fast |
| NVIDIA API | ✅ Supported | High quality |
| Anthropic | ⚠️ Optional | API key required |
| OpenAI | ⚠️ Optional | API key required |
| SiliconFlow | ⚠️ Optional |备用 |

## Learning Process

1. User sends input
2. NEURON extracts concepts and relationships
3. LLM generates insights using selected strategy
4. Knowledge is stored with confidence scores
5. XP is awarded and level increases
6. Curiosity questions are generated
7. Cross-domain syntheses are discovered

## Privacy

- All data stored locally by default
- No data exfiltration
- User controls all settings
- Offline-first architecture

## License

MIT License

## Author

Created by Amir Ali (Project Shahzada)

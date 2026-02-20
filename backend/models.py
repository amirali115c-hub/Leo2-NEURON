"""
NEURON v2.0 Database Models
Self-Learning AI Agent Knowledge System
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import json
import os

Base = declarative_base()

# Database path
DB_PATH = os.environ.get("NEURON_DB_PATH", "./data/neuron.db")


def get_db_path():
    """Ensure data directory exists and return DB path"""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else "./data", exist_ok=True)
    return DB_PATH


# Create engine
engine = create_engine(f"sqlite:///{get_db_path()}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


class KBEntry(Base):
    """Knowledge Base Entry - learned insights"""
    __tablename__ = "kb_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow)
    user_input = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    key_insight = Column(Text, nullable=False)
    secondary_insight = Column(Text)
    domain = Column(String, nullable=False)
    sub_domain = Column(String)
    confidence = Column(Float, default=0.5)
    reliability = Column(Float, default=0.5)
    complexity = Column(String, default='basic')
    concepts_json = Column(Text, default='[]')  # JSON array
    patterns_json = Column(Text, default='[]')  # JSON array
    mnemonic_hook = Column(Text)
    memory_weight = Column(Float, default=0.5)
    strategy = Column(String)
    quality_score = Column(Integer, default=50)
    
    # Properties
    @property
    def concepts(self) -> List[str]:
        return json.loads(self.concepts_json) if self.concepts_json else []
    
    @property
    def patterns(self) -> List[str]:
        return json.loads(self.patterns_json) if self.patterns_json else []
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts.isoformat() if self.ts else None,
            "user_input": self.user_input,
            "response": self.response,
            "key_insight": self.key_insight,
            "secondary_insight": self.secondary_insight,
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "complexity": self.complexity,
            "concepts": self.concepts,
            "patterns": self.patterns,
            "mnemonic_hook": self.mnemonic_hook,
            "memory_weight": self.memory_weight,
            "strategy": self.strategy,
            "quality_score": self.quality_score
        }


class Concept(Base):
    """Extracted Concepts"""
    __tablename__ = "concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    domain = Column(String, nullable=False)
    count = Column(Integer, default=1)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "count": self.count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None
        }


class Edge(Base):
    """Relationship Edges between concepts"""
    __tablename__ = "edges"
    
    id = Column(Integer, primary_key=True, index=True)
    from_concept = Column(String, nullable=False, index=True)
    to_concept = Column(String, nullable=False, index=True)
    link_type = Column(String)
    strength = Column(Float, default=0.5)
    domain = Column(String)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_concept,
            "to": self.to_concept,
            "link": self.link_type,
            "strength": self.strength,
            "domain": self.domain
        }


class CuriosityQuestion(Base):
    """Autonomous Curiosity Questions"""
    __tablename__ = "curiosity_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    domain = Column(String, nullable=False)
    ts = Column(DateTime, default=datetime.utcnow)
    explored = Column(Boolean, default=False)
    explored_ts = Column(DateTime)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "domain": self.domain,
            "ts": self.ts.isoformat() if self.ts else None,
            "explored": self.explored,
            "explored_ts": self.explored_ts.isoformat() if self.explored_ts else None
        }


class Hypothesis(Base):
    """Generated Hypotheses"""
    __tablename__ = "hypotheses"
    
    id = Column(Integer, primary_key=True, index=True)
    hypothesis = Column(Text, nullable=False)
    domain = Column(String, nullable=False)
    ts = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='untested')  # untested, tested, confirmed, rejected
    tested_ts = Column(DateTime)
    result = Column(Text)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "domain": self.domain,
            "ts": self.ts.isoformat() if self.ts else None,
            "status": self.status,
            "tested_ts": self.tested_ts.isoformat() if self.tested_ts else None,
            "result": self.result
        }


class Synthesis(Base):
    """Cross-Domain Syntheses"""
    __tablename__ = "syntheses"
    
    id = Column(Integer, primary_key=True, index=True)
    from_domain = Column(String, nullable=False)
    to_domain = Column(String, nullable=False)
    connection = Column(Text, nullable=False)
    novelty = Column(Float, default=0.5)
    ts = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_domain,
            "to": self.to_domain,
            "connection": self.connection,
            "novelty": self.novelty,
            "ts": self.ts.isoformat() if self.ts else None
        }


class Insight(Base):
    """Insights (including meta-insights)"""
    __tablename__ = "insights"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    domain = Column(String)
    ts = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String)
    is_meta = Column(Boolean, default=False)
    is_secondary = Column(Boolean, default=False)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "domain": self.domain,
            "ts": self.ts.isoformat() if self.ts else None,
            "strategy": self.strategy,
            "meta": self.is_meta,
            "secondary": self.is_secondary
        }


class Capability(Base):
    """Capabilities (auto-unlocked)"""
    __tablename__ = "capabilities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    domain = Column(String)
    level = Column(Integer, default=1)
    unlocked = Column(Boolean, default=False)
    unlocked_ts = Column(DateTime)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "level": self.level,
            "unlocked": self.unlocked,
            "unlocked_ts": self.unlocked_ts.isoformat() if self.unlocked_ts else None
        }


class Goal(Base):
    """User Learning Goals"""
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    progress = Column(Integer, default=0)
    completed = Column(Boolean, default=False)
    ts = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "progress": self.progress,
            "completed": self.completed,
            "ts": self.ts.isoformat() if self.ts else None
        }


class SystemStats(Base):
    """System Statistics"""
    __tablename__ = "system_stats"
    
    key = Column(String, primary_key=True)
    value = Column(Text)
    
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value
        }

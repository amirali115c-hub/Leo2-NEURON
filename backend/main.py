"""
NEURON v2.0 - Self-Learning AI Agent API
FastAPI backend with learning capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import json
import os

from models import (
    KBEntry, Concept, Edge, CuriosityQuestion, Hypothesis,
    Synthesis, Insight, Capability, Goal, SystemStats,
    init_db, get_db
)
from llm_client import get_llm_client, test_llm_connection

# Create FastAPI app
app = FastAPI(
    title="NEURON v2.0 API",
    description="Self-Learning AI Agent Backend",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()


# ============ Pydantic Models ============

class LearnRequest(BaseModel):
    user_input: str
    strategy: str = "CoT"
    force_new: bool = False

class ChatRequest(BaseModel):
    message: str
    strategy: str = "CoT"

class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None

class GoalCreate(BaseModel):
    title: str

class GoalUpdate(BaseModel):
    progress: int

class CuriosityExploreRequest(BaseModel):
    question_id: int

class HypothesisTestRequest(BaseModel):
    hypothesis_id: int
    result: str


# ============ Core Learning Endpoints ============

@app.post("/api/neuron/learn")
async def learn(request: LearnRequest, db: Session = Depends(get_db)):
    """Process a learning interaction"""
    
    # Get LLM client
    llm = get_llm_client()
    
    # Get context
    kb_count = db.query(KBEntry).count()
    concepts = db.query(Concept.name).limit(20).all()
    concepts_list = [c[0] for c in concepts]
    goals = db.query(Goal).filter(Goal.completed == 0).all()
    goals_list = [g.title for g in goals]
    
    context = {
        "kb_count": kb_count,
        "concepts": concepts_list,
        "goals": goals_list
    }
    
    # Process learning with error handling
    try:
        result = await llm.learn(request.user_input, context, request.strategy)
    except Exception as e:
        # LLM not available, save basic entry
        entry = KBEntry(
            user_input=request.user_input,
            response=f"LLM unavailable. Saved for later processing.",
            key_insight=f"User asked about: {request.user_input[:100]}",
            domain="General",
            strategy=request.strategy,
            complexity="basic",
            confidence=0.5,
            reliability=0.5
        )
        db.add(entry)
        db.commit()
        return {
            "status": "success", 
            "entry_id": entry.id, 
            "learned": None,
            "message": "LLM unavailable, entry saved for later processing"
        }
    
    if not result.get("learned"):
        # No structured learning, just save basic entry
        entry = KBEntry(
            user_input=request.user_input,
            response=result.get("response", ""),
            key_insight="General learning",
            domain="General",
            strategy=request.strategy
        )
        db.add(entry)
        db.commit()
        return {"status": "success", "entry_id": entry.id, "learned": None}
    
    learned = result["learned"]
    
    # Create KB entry
    entry = KBEntry(
        user_input=request.user_input,
        response=result.get("response", ""),
        key_insight=learned.get("keyInsight", ""),
        secondary_insight=learned.get("secondaryInsight"),
        domain=learned.get("domain", "General"),
        sub_domain=learned.get("subDomain"),
        confidence=learned.get("confidence", 0.5),
        reliability=learned.get("reliability", 0.5),
        complexity=learned.get("complexity", "basic"),
        concepts_json=json.dumps(learned.get("concepts", [])),
        patterns_json=json.dumps(learned.get("patterns", [])),
        mnemonic_hook=learned.get("mnemonicHook"),
        strategy=request.strategy,
        quality_score=result.get("qualityScore", 50)
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    
    # Update concepts
    for concept_name in learned.get("concepts", []):
        existing = db.query(Concept).filter(Concept.name == concept_name).first()
        if existing:
            existing.count += 1
            existing.last_seen = datetime.utcnow()
        else:
            concept = Concept(
                name=concept_name,
                domain=learned.get("domain", "General"),
                count=1
            )
            db.add(concept)
    
    # Create edges from relationships
    for rel in learned.get("relationships", []):
        edge = Edge(
            from_concept=rel.get("from", ""),
            to_concept=rel.get("to", ""),
            link_type=rel.get("link"),
            strength=rel.get("strength", 0.5),
            domain=learned.get("domain")
        )
        db.add(edge)
    
    # Add curiosity questions
    for q in learned.get("curiosityQuestions", []):
        curiosity = CuriosityQuestion(
            question=q,
            domain=learned.get("domain", "General")
        )
        db.add(curiosity)
    
    # Add hypotheses
    for h in learned.get("hypotheses", []):
        hypothesis = Hypothesis(
            hypothesis=h,
            domain=learned.get("domain", "General")
        )
        db.add(hypothesis)
    
    # Add cross-domain syntheses
    for link in learned.get("crossDomainLinks", []):
        if link.get("novelty", 0) > 0.5:
            synthesis = Synthesis(
                from_domain=learned.get("domain", "General"),
                to_domain=link.get("domain"),
                connection=link.get("connection", ""),
                novelty=link.get("novelty", 0.5)
            )
            db.add(synthesis)
    
    # Add insight
    if learned.get("keyInsight"):
        insight = Insight(
            text=learned.get("keyInsight"),
            domain=learned.get("domain"),
            strategy=request.strategy
        )
        db.add(insight)
    
    db.commit()
    
    return {
        "status": "success",
        "entry_id": entry.id,
        "learned": learned
    }


@app.post("/api/neuron/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Chat with learning context"""
    
    llm = get_llm_client()
    
    # Get context from KB
    recent_kb = db.query(KBEntry).order_by(KBEntry.id.desc()).limit(10).all()
    context_text = "\n".join([f"• {k.key_insight} ({k.domain})" for k in recent_kb])
    
    system_prompt = f"""You are NEURON v2.0, a self-learning AI agent.
Your knowledge base contains {len(recent_kb)} recent insights:
{context_text}

You have learned from interactions and can reference your knowledge."""
    
    messages = [{"role": "user", "content": request.message}]
    response = await llm.chat(messages, system_prompt)
    
    return {"response": response}


@app.post("/api/neuron/query")
async def query_kb(request: QueryRequest, db: Session = Depends(get_db)):
    """Query the knowledge base"""
    
    # Search KB
    query = db.query(KBEntry)
    
    if request.domain:
        query = query.filter(KBEntry.domain == request.domain)
    
    if request.query:
        search = f"%{request.query}%"
        query = query.filter(
            (KBEntry.key_insight.like(search)) |
            (KBEntry.user_input.like(search)) |
            (KBEntry.concepts_json.like(search))
        )
    
    results = query.order_by(KBEntry.id.desc()).limit(20).all()
    
    return {
        "query": request.query,
        "domain": request.domain,
        "count": len(results),
        "results": [r.to_dict() for r in results]
    }


# ============ Knowledge Management ============

@app.get("/api/neuron/kb")
async def get_kb(
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    domain: str = None,
    db: Session = Depends(get_db)
):
    """Get knowledge base entries"""
    
    query = db.query(KBEntry)
    
    if domain:
        query = query.filter(KBEntry.domain == domain)
    
    total = query.count()
    entries = query.order_by(KBEntry.id.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": [e.to_dict() for e in entries]
    }


@app.get("/api/neuron/concepts")
async def get_concepts(db: Session = Depends(get_db)):
    """Get all concepts"""
    concepts = db.query(Concept).order_by(Concept.count.desc()).all()
    return {"concepts": [c.to_dict() for c in concepts]}


@app.get("/api/neuron/edges")
async def get_edges(db: Session = Depends(get_db)):
    """Get relationship edges"""
    edges = db.query(Edge).all()
    return {"edges": [e.to_dict() for e in edges]}


@app.get("/api/neuron/syntheses")
async def get_syntheses(db: Session = Depends(get_db)):
    """Get cross-domain syntheses"""
    syntheses = db.query(Synthesis).order_by(Synthesis.novelty.desc()).all()
    return {"syntheses": [s.to_dict() for s in syntheses]}


# ============ Curiosity & Exploration ============

@app.get("/api/neuron/curiosity")
async def get_curiosity(unexplored: bool = False, db: Session = Depends(get_db)):
    """Get curiosity questions"""
    
    query = db.query(CuriosityQuestion)
    
    if unexplored:
        query = query.filter(CuriosityQuestion.explored == 0)
    
    questions = query.order_by(CuriosityQuestion.id.desc()).limit(50).all()
    
    return {
        "count": len(questions),
        "questions": [q.to_dict() for q in questions]
    }


@app.post("/api/neuron/curiosity/generate")
async def generate_curiosity(domain: str = None, db: Session = Depends(get_db)):
    """Generate new curiosity questions based on KB"""
    
    # Get recent insights
    recent_kb = db.query(KBEntry).order_by(KBEntry.id.desc()).limit(10).all()
    
    if not recent_kb:
        return {"status": "error", "message": "Not enough knowledge to generate curiosity"}
    
    insights = [k.key_insight for k in recent_kb]
    
    llm = get_llm_client()
    
    system_prompt = """You are NEURON's curiosity engine. Generate 3 thought-provoking 
questions based on the recent insights. These should explore deeper implications 
and gaps in understanding. Respond ONLY with a JSON array of questions."""
    
    messages = [{
        "role": "user",
        "content": f"Based on these recent insights:\n{chr(10).join(insights)}\n\nGenerate 3 curiosity questions. Return JSON array."
    }]
    
    response = await llm.chat(messages, system_prompt)
    
    # Parse questions
    try:
        import re
        questions = re.findall(r'"([^"]+)"', response)
        if not questions:
            questions = [response.strip()]
    except:
        questions = [response.strip()]
    
    # Save questions
    for q in questions:
        curiosity = CuriosityQuestion(
            question=q,
            domain=domain or "General"
        )
        db.add(curiosity)
    
    db.commit()
    
    return {
        "status": "success",
        "generated": len(questions),
        "questions": questions
    }


@app.get("/api/neuron/hypotheses")
async def get_hypotheses(status: str = None, db: Session = Depends(get_db)):
    """Get hypotheses"""
    
    query = db.query(Hypothesis)
    
    if status:
        query = query.filter(Hypothesis.status == status)
    
    hypotheses = query.order_by(Hypothesis.id.desc()).all()
    
    return {
        "count": len(hypotheses),
        "hypotheses": [h.to_dict() for h in hypotheses]
    }


# ============ Goals ============

@app.get("/api/neuron/goals")
async def get_goals(db: Session = Depends(get_db)):
    """Get all goals"""
    goals = db.query(Goal).order_by(Goal.id.desc()).all()
    return {"goals": [g.to_dict() for g in goals]}


@app.post("/api/neuron/goals")
async def create_goal(goal: GoalCreate, db: Session = Depends(get_db)):
    """Create a new goal"""
    new_goal = Goal(title=goal.title)
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return {"status": "success", "goal": new_goal.to_dict()}


@app.patch("/api/neuron/goals/{goal_id}/progress")
async def update_goal_progress(goal_id: int, update: GoalUpdate, db: Session = Depends(get_db)):
    """Update goal progress"""
    goal = db.query(Goal).filter(Goal.id == goal_id).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    goal.progress = min(100, update.progress)
    goal.completed = goal.progress >= 100
    
    db.commit()
    
    return {"status": "success", "goal": goal.to_dict()}


# ============ Analytics ============

@app.get("/api/neuron/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    
    stats = {
        "kb_count": db.query(KBEntry).count(),
        "concept_count": db.query(Concept).count(),
        "edge_count": db.query(Edge).count(),
        "curiosity_count": db.query(CuriosityQuestion).filter(CuriosityQuestion.explored == 0).count(),
        "hypothesis_count": db.query(Hypothesis).count(),
        "synthesis_count": db.query(Synthesis).count(),
        "goal_count": db.query(Goal).count(),
        "completed_goals": db.query(Goal).filter(Goal.completed == 1).count(),
        "capability_count": db.query(Capability).filter(Capability.unlocked == 1).count(),
    }
    
    # Domain breakdown
    from sqlalchemy import func
    domain_stats = db.query(
        KBEntry.domain,
        func.count(KBEntry.id)
    ).group_by(KBEntry.domain).all()
    
    stats["domains"] = {d: c for d, c in domain_stats}
    
    return stats


@app.get("/api/neuron/analytics/curve")
async def get_learning_curve(db: Session = Depends(get_db)):
    """Get learning curve data"""
    
    entries = db.query(KBEntry).order_by(KBEntry.id.desc()).limit(50).all()
    
    data = []
    for i, entry in enumerate(reversed(entries)):
        data.append({
            "x": i + 1,
            "confidence": entry.confidence,
            "complexity": entry.complexity,
            "domain": entry.domain
        })
    
    return {"data": data}


@app.get("/api/neuron/analytics/domains")
async def get_domain_stats(db: Session = Depends(get_db)):
    """Get domain statistics"""
    
    from sqlalchemy import func
    
    stats = db.query(
        KBEntry.domain,
        func.count(KBEntry.id),
        func.avg(KBEntry.confidence)
    ).group_by(KBEntry.domain).all()
    
    return {
        "domains": [
            {"domain": d, "count": c, "avg_confidence": round(avg, 2)}
            for d, c, avg in stats
        ]
    }


# ============ Capabilities ============

@app.get("/api/neuron/capabilities")
async def get_capabilities(db: Session = Depends(get_db)):
    """Get all capabilities"""
    capabilities = db.query(Capability).all()
    return {
        "capabilities": [c.to_dict() for c in capabilities],
        "unlocked": len([c for c in capabilities if c.unlocked])
    }


@app.post("/api/neuron/capabilities/unlock")
async def unlock_capability(
    name: str,
    description: str = None,
    domain: str = "General",
    db: Session = Depends(get_db)
):
    """Unlock a new capability"""
    
    existing = db.query(Capability).filter(Capability.name == name).first()
    
    if existing:
        if existing.unlocked:
            return {"status": "already_unlocked", "capability": existing.to_dict()}
        existing.unlocked = True
        existing.unlocked_ts = datetime.utcnow()
        existing.description = description or existing.description
        existing.domain = domain
    else:
        capability = Capability(
            name=name,
            description=description,
            domain=domain,
            unlocked=True,
            unlocked_ts=datetime.utcnow()
        )
        db.add(capability)
    
    db.commit()
    
    return {"status": "success", "message": f"Capability unlocked: {name}"}


# ============ System ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NEURON v2.0"}


@app.get("/api/neuron/test-llm")
async def test_llm(provider: str = None):
    """Test LLM connection"""
    return await test_llm_connection(provider)


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

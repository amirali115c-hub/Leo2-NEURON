import React, { useState, useEffect, useRef } from 'react';

// API Configuration
const API_BASE = ''; // Use relative URL

// Strategies
const STRATEGIES = [
  { id: 'CoT', label: 'Chain-of-Thought', color: '#4facfe' },
  { id: 'ToT', label: 'Tree of Thought', color: '#f472b6' },
  { id: 'Synthesis', label: 'Synthesis', color: '#34d399' },
  { id: 'Socratic', label: 'Socratic', color: '#a78bfa' },
  { id: 'Analysis', label: 'Analysis', color: '#f6d860' },
];

// Domains
const DOMAINS = [
  { id: 'Science', icon: '⚗️', color: '#22d3ee' },
  { id: 'Technology', icon: '💻', color: '#4facfe' },
  { id: 'Philosophy', icon: '∞', color: '#a78bfa' },
  { id: 'Arts', icon: '🎨', color: '#f472b6' },
  { id: 'History', icon: '📜', color: '#fb923c' },
  { id: 'Math', icon: '∑', color: '#34d399' },
  { id: 'Language', icon: '🗣️', color: '#f6d860' },
  { id: 'Psychology', icon: '🧩', color: '#f87171' },
  { id: 'General', icon: '✦', color: '#94a3b8' },
];

// Icons
const Icons = {
  Brain: '🧠',
  Book: '📚',
  Lightbulb: '💡',
  Target: '🎯',
  Graph: '📊',
  Question: '❓',
  Check: '✓',
  Send: '➤',
};

function App() {
  // State
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [learningStep, setLearningStep] = useState('');
  const [strategy, setStrategy] = useState('CoT');
  const [level, setLevel] = useState(1);
  const [xp, setXp] = useState(0);
  
  // Stats
  const [stats, setStats] = useState({
    kb_count: 0,
    concept_count: 0,
    goal_count: 0,
    curiosity_count: 0,
  });
  
  // Goals
  const [goals, setGoals] = useState([]);
  const [goalInput, setGoalInput] = useState('');
  
  // KB
  const [kbEntries, setKbEntries] = useState([]);
  
  // Refs
  const messagesEndRef = useRef(null);
  
  // Load initial data
  useEffect(() => {
    fetchStats();
    fetchGoals();
    fetchKB();
    
    // Add welcome message
    setMessages([
      {
        id: 0,
        role: 'assistant',
        content: "Hello! I'm NEURON v2.0, a self-learning AI agent.\n\nI learn from every interaction, extract concepts, and continuously improve my understanding. My strategies include Chain-of-Thought reasoning, Tree of Thought, Synthesis, Socratic questioning, and Deep Analysis.\n\nWhat would you like to learn or explore today?",
      }
    ]);
  }, []);
  
  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Fetch functions
  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/neuron/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
        const calculatedLevel = Math.floor((data.kb_count || 0) / 10) + 1;
        setLevel(calculatedLevel);
        setXp((data.kb_count || 0) * 15);
      }
    } catch (e) {
      console.log('Stats not available');
    }
  };
  
  const fetchGoals = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/neuron/goals`);
      if (res.ok) {
        const data = await res.json();
        setGoals(data.goals || []);
      }
    } catch (e) {
      console.log('Goals not available');
    }
  };
  
  const fetchKB = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/neuron/kb?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setKbEntries(data.entries || []);
      }
    } catch (e) {
      console.log('KB not available');
    }
  };
  
  // Chat
  const handleSend = async () => {
    if (!input.trim() || loading) return;
    
    const userInput = input.trim();
    setInput('');
    setLoading(true);
    
    // Add user message
    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'user',
      content: userInput,
    }]);
    
    try {
      // Call learning endpoint
      const res = await fetch(`${API_BASE}/api/neuron/learn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: userInput,
          strategy: strategy,
        }),
      });
      
      let assistantResponse = "I've processed your input.";
      let learned = null;
      
      if (res.ok) {
        const data = await res.json();
        assistantResponse = data.response || assistantResponse;
        learned = data.learned;
        
        // Update stats
        fetchStats();
        fetchKB();
      }
      
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: assistantResponse,
        learned: learned,
      }]);
      
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: "I encountered an error processing your request. Please try again.",
        error: true,
      }]);
    }
    
    setLoading(false);
  };
  
  // Goals
  const addGoal = async () => {
    if (!goalInput.trim()) return;
    
    try {
      const res = await fetch(`${API_BASE}/api/neuron/goals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: goalInput }),
      });
      
      if (res.ok) {
        fetchGoals();
        setGoalInput('');
      }
    } catch (e) {
      console.log('Failed to add goal');
    }
  };
  
  const updateGoalProgress = async (goalId, currentProgress) => {
    const newProgress = Math.min(100, currentProgress + 25);
    
    try {
      await fetch(`${API_BASE}/api/neuron/goals/${goalId}/progress`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ progress: newProgress }),
      });
      fetchGoals();
    } catch (e) {
      console.log('Failed to update goal');
    }
  };
  
  // Quick prompts
  const quickPrompts = [
    "What are the first principles of AI?",
    "Explain quantum computing simply",
    "How does learning work?",
    "Connect mathematics and philosophy",
    "Teach me about neural networks",
  ];
  
  // Render functions
  const renderChat = () => (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">{Icons.Brain}</div>
            <div className="empty-state-title">NEURON v2.0 Online</div>
            <div className="empty-state-text">
              Start learning by asking a question or exploring a topic.
            </div>
            <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
              {quickPrompts.map((p) => (
                <button
                  key={p}
                  onClick={() => {
                    setInput(p);
                    document.querySelector('.chat-input')?.focus();
                  }}
                  style={{
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-mid)',
                    padding: '8px 14px',
                    borderRadius: 20,
                    cursor: 'pointer',
                    fontSize: 11,
                    fontFamily: 'inherit',
                  }}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🧠'}
              </div>
              <div className="message-content">
                {msg.content}
                {msg.learned && (
                  <div className="message-learning">
                    <div style={{ marginBottom: 6 }}>
                      <span className="domain-badge">{msg.learned.domain}</span>
                      <span style={{ color: 'var(--violet)', fontSize: 9 }}>
                        {msg.learned.strategy}
                      </span>
                    </div>
                    {msg.learned.keyInsight && (
                      <div style={{ color: 'var(--accent)', fontStyle: 'italic' }}>
                        💡 {msg.learned.keyInsight}
                      </div>
                    )}
                    {msg.learned.concepts?.length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        {msg.learned.concepts.map((c, i) => (
                          <span key={i} className="kb-tag" style={{ marginRight: 4 }}>{c}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🧠</div>
            <div className="message-content">
              <div className="learning-indicator">
                <div className="learning-dots">
                  <span></span><span></span><span></span>
                </div>
                <span>Processing...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <input
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask or teach NEURON something..."
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            {Icons.Send}
          </button>
        </div>
      </div>
    </div>
  );
  
  const renderKB = () => (
    <div className="kb-container">
      {kbEntries.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">{Icons.Book}</div>
          <div className="empty-state-title">No Knowledge Yet</div>
          <div className="empty-state-text">
            Start learning and your knowledge will appear here.
          </div>
        </div>
      ) : (
        <div className="kb-grid">
          {kbEntries.map((entry) => (
            <div key={entry.id} className="kb-card">
              <div className="kb-card-header">
                <span className="kb-domain">
                  {DOMAINS.find(d => d.id === entry.domain)?.icon} {entry.domain}
                </span>
                <span className="kb-date">
                  {new Date(entry.ts).toLocaleDateString()}
                </span>
              </div>
              <div className="kb-insight">
                💡 {entry.key_insight}
              </div>
              <div className="kb-meta">
                <span className="kb-tag">{entry.complexity}</span>
                <span className="kb-tag">conf: {Math.round(entry.confidence * 100)}%</span>
                <span className="kb-tag">{entry.strategy}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
  
  const renderGoals = () => (
    <div className="goals-container">
      <div className="goal-input-wrapper">
        <input
          className="goal-input"
          value={goalInput}
          onChange={(e) => setGoalInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addGoal()}
          placeholder="Add a learning goal..."
        />
        <button className="goal-add-btn" onClick={addGoal}>
          + Add Goal
        </button>
      </div>
      
      {goals.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">{Icons.Target}</div>
          <div className="empty-state-title">No Goals Set</div>
          <div className="empty-state-text">
            Add goals to track your learning journey.
          </div>
        </div>
      ) : (
        <div className="goals-list">
          {goals.map((goal) => (
            <div key={goal.id} className="goal-card">
              <div className="goal-header">
                <span className="goal-title">{goal.title}</span>
                {goal.completed && <span className="goal-complete">{Icons.Check}</span>}
              </div>
              <div className="goal-progress">
                <div
                  className="goal-progress-bar"
                  style={{ width: `${goal.progress}%` }}
                />
              </div>
              <div className="goal-footer">
                <span>{goal.progress}% complete</span>
                {!goal.completed && (
                  <button
                    onClick={() => updateGoalProgress(goal.id, goal.progress)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--accent)',
                      cursor: 'pointer',
                      fontSize: 9,
                    }}
                  >
                    + Add Progress
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
  
  const renderAnalytics = () => (
    <div className="analytics-container">
      <div className="analytics-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.kb_count || 0}</div>
          <div className="stat-label">Knowledge Entries</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.concept_count || 0}</div>
          <div className="stat-label">Concepts</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.curiosity_count || 0}</div>
          <div className="stat-label">Questions</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.hypothesis_count || 0}</div>
          <div className="stat-label">Hypotheses</div>
        </div>
      </div>
      
      <div className="sidebar-section">
        <div className="sidebar-title">LEVEL PROGRESS</div>
        <div className="stat-card">
          <div className="stat-value">LVL {level}</div>
          <div className="stat-label">{xp} XP</div>
        </div>
      </div>
    </div>
  );
  
  // Render
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">{Icons.Brain}</div>
          <div className="header-title">
            NEURON <span>v2.0</span>
          </div>
        </div>
        <div className="header-right">
          <select
            className="strategy-select"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          >
            {STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          <div className="level-badge">LVL {level}</div>
        </div>
      </header>
      
      {/* Navigation */}
      <nav className="nav">
        {[
          { id: 'chat', label: '💬 LEARN' },
          { id: 'kb', label: '📚 KNOWLEDGE' },
          { id: 'goals', label: '🎯 GOALS' },
          { id: 'analytics', label: '📊 ANALYTICS' },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`nav-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      
      {/* Main Content */}
      <main className="main">
        {activeTab === 'chat' && renderChat()}
        {activeTab === 'kb' && renderKB()}
        {activeTab === 'goals' && renderGoals()}
        {activeTab === 'analytics' && renderAnalytics()}
        
        {/* Sidebar Stats */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-title">LIVE STATS</div>
            <div className="stat-grid">
              <div className="stat-card">
                <div className="stat-value">{stats.kb_count || 0}</div>
                <div className="stat-label">KB</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.concept_count || 0}</div>
                <div className="stat-label">Concepts</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.curiosity_count || 0}</div>
                <div className="stat-label">Questions</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{goals.filter(g => g.completed).length}/{goals.length}</div>
                <div className="stat-label">Goals</div>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;

import { useState, useRef } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:5000'

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])
  const textareaRef = useRef(null)

  const totalAnalyzed = history.length
  const positiveCount = history.filter(h => h.sentiment === 'positive').length
  const negativeCount = history.filter(h => h.sentiment === 'negative').length

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() })
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult(data)
      setHistory(prev => [
        { id: Date.now(), text: text.trim(), sentiment: data.sentiment, confidence: data.confidence },
        ...prev.slice(0, 9)
      ])
    } catch (err) {
      if (err.message.includes('fetch')) {
        setError('Cannot connect to backend. Make sure Flask is running on http://127.0.0.1:5000')
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze()
  }

  const clear = () => {
    setText('')
    setResult(null)
    setError('')
    textareaRef.current?.focus()
  }

  const loadFromHistory = (item) => {
    setText(item.text)
    setResult({ text: item.text, sentiment: item.sentiment, confidence: item.confidence })
    setError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">💬</div>
          <span className="logo-name">SentiSense</span>
        </div>

        <div className="header-right">
          <span className="header-badge">ML Driven</span>
          <span className="header-badge">Real-Time Analysis</span>
          <span className="header-badge">Data Powered</span>
        </div>
      </header>

      <main className="main">
        {/* Hero */}
        <div className="hero">
          <p className="hero-eyebrow">Sentiment Analysis</p>
          <h1 className="hero-title">
            Understand the <em>mood</em> behind any text
          </h1>
          <p className="hero-sub">
            Paste a review, tweet, sentence, or any text — and get an instant sentiment reading powered by machine learning.
          </p>
        </div>

        {/* Stats Row */}
        {totalAnalyzed > 0 && (
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">{totalAnalyzed}</div>
              <div className="stat-label">Analyzed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--positive)' }}>{positiveCount}</div>
              <div className="stat-label">Positive</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--negative)' }}>{negativeCount}</div>
              <div className="stat-label">Negative</div>
            </div>
          </div>
        )}

        {/* Analyzer Card */}
        <div className="card">
          <p className="card-label">Enter Text</p>

          <div className="textarea-wrapper">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type or paste any text here… e.g. &quot;I absolutely loved the movie, it was brilliant!&quot;"
              maxLength={1000}
              autoFocus
            />
            <span className="char-count">{text.length}/1000</span>
          </div>

          <div className="actions">
            <button
              className="btn-analyze"
              onClick={analyze}
              disabled={loading || !text.trim()}
            >
              {loading
                ? <><div className="spinner" /> Analyzing…</>
                : <><span>✦</span> Analyze Sentiment</>
              }
            </button>
            {text && (
              <button className="btn-clear" onClick={clear}>Clear</button>
            )}
          </div>

          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '10px' }}>
            Tip: Press <kbd style={{ background: 'var(--surface2)', padding: '1px 5px', borderRadius: '3px', fontFamily: 'monospace', fontSize: '11px' }}>Ctrl+Enter</kbd> to analyze quickly
          </p>

          {/* Error */}
          {error && <div className="error-msg">⚠️ {error}</div>}

          {/* Result */}
          {result && (
            <div className={`result ${result.sentiment}`}>
              <div className="result-top">
                <span className="result-emoji">
                  {result.sentiment === 'positive' ? '😊' : '😞'}
                </span>
                <div>
                  <div className="result-label">
                    {result.sentiment === 'positive' ? 'Positive Sentiment' : 'Negative Sentiment'}
                  </div>
                  <div className="result-sub">
                    {result.sentiment === 'positive'
                      ? 'The text carries a positive tone'
                      : 'The text carries a negative tone'}
                  </div>
                </div>
              </div>

              <div className="result-meta">
                <div className="confidence-wrap">
                  <div className="confidence-label">Confidence</div>
                  <div className="confidence-bar-bg">
                    <div
                      className="confidence-bar-fill"
                      style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <div className="confidence-pct">{(result.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* History */}
        {history.length > 0 && (
          <div className="history-section">
            <div className="section-header">
              <span className="section-title">Recent Analyses</span>
              <button className="btn-clear-history" onClick={() => setHistory([])}>Clear all</button>
            </div>
            <div className="history-list">
              {history.map(item => (
                <div key={item.id} className="history-item" onClick={() => loadFromHistory(item)}>
                  <div className={`history-dot ${item.sentiment}`} />
                  <span className="history-text">{item.text}</span>
                  <span className={`history-badge ${item.sentiment}`}>
                    {item.sentiment}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        SentiSense · Logistic Regression + TF-IDF · Flask + React
      </footer>
    </div>
  )
}

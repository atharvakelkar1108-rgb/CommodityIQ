import { useState, useEffect } from 'react'
import { RefreshCw, ExternalLink } from 'lucide-react'

const FEEDS = [
  { label:'All Commodities',  q:'commodity+market+prices' },
  { label:'Gold & Metals',    q:'gold+silver+metal+prices' },
  { label:'Oil & Energy',     q:'crude+oil+energy+prices' },
  { label:'Agriculture',      q:'wheat+corn+agricultural+commodity' },
  { label:'India Markets',    q:'MCX+commodity+India+NCDEX' },
]

// Uses RSS2JSON free API to fetch Google News RSS
const RSS_API = 'https://api.rss2json.com/v1/api.json?rss_url='
const GNEWS   = 'https://news.google.com/rss/search?q='

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const m = Math.floor(diff / 60000)
  if (m < 60)  return m + 'm ago'
  const h = Math.floor(m / 60)
  if (h < 24)  return h + 'h ago'
  return Math.floor(h/24) + 'd ago'
}

function stripHtml(html) {
  return html?.replace(/<[^>]*>/g,'').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&#39;/g,"'").trim() || ''
}

export default function NewsPage() {
  const [articles, setArticles] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [feed,     setFeed]     = useState(0)
  const [error,    setError]    = useState(null)

  const loadNews = async (feedIdx) => {
    setLoading(true)
    setError(null)
    try {
      const q   = FEEDS[feedIdx].q
      const url = RSS_API + encodeURIComponent(GNEWS + q + '&hl=en-IN&gl=IN&ceid=IN:en')
      const r   = await fetch(url)
      const d   = await r.json()
      if (d.status === 'ok') {
        setArticles(d.items || [])
      } else {
        throw new Error('Feed unavailable')
      }
    } catch (e) {
      setError('Could not load news. Check your internet connection.')
      setArticles([])
    }
    setLoading(false)
  }

  useEffect(() => { loadNews(feed) }, [feed])

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'1.5rem', flexWrap:'wrap', gap:12 }}>
        <div>
          <h1 style={{ fontSize:22, fontWeight:700, color:'#e2e8f0' }}>Commodity News</h1>
          <p style={{ fontSize:13, color:'#64748b', marginTop:2 }}>Live news via Google News · India edition</p>
        </div>
        <button onClick={() => loadNews(feed)} disabled={loading}
          style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 14px',
            background:'transparent', color:'#94a3b8', border:'1px solid #2d3148',
            borderRadius:8, fontWeight:600, fontSize:13, cursor:'pointer' }}>
          <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }}/> Refresh
        </button>
      </div>

      {/* Feed tabs */}
      <div style={{ display:'flex', gap:6, marginBottom:'1.5rem', flexWrap:'wrap' }}>
        {FEEDS.map((f,i) => (
          <button key={i} className={'tab-btn ' + (feed===i?'active':'')} onClick={()=>setFeed(i)}>
            {f.label}
          </button>
        ))}
      </div>

      {error && (
        <div style={{ background:'rgba(239,68,68,0.1)', border:'1px solid rgba(239,68,68,0.3)',
          borderRadius:10, padding:'1rem', color:'#ef4444', marginBottom:'1rem', fontSize:13 }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(340px,1fr))', gap:12 }}>
          {[...Array(6)].map((_,i) => (
            <div key={i} className="card" style={{ padding:'1.25rem' }}>
              <div style={{ height:14, background:'#2d3148', borderRadius:4, marginBottom:10, width:'60%' }}/>
              <div style={{ height:12, background:'#1e2235', borderRadius:4, marginBottom:6 }}/>
              <div style={{ height:12, background:'#1e2235', borderRadius:4, marginBottom:6, width:'80%' }}/>
              <div style={{ height:10, background:'#1e2235', borderRadius:4, width:'40%', marginTop:12 }}/>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(340px,1fr))', gap:12 }}>
          {articles.map((a, i) => (
            <a key={i} href={a.link} target="_blank" rel="noopener noreferrer"
              style={{ textDecoration:'none' }}>
              <div className="card" style={{ height:'100%', cursor:'pointer', transition:'border-color 0.15s' }}
                onMouseEnter={e=>e.currentTarget.style.borderColor='#3d4268'}
                onMouseLeave={e=>e.currentTarget.style.borderColor='#2d3148'}>
                {a.thumbnail && (
                  <img src={a.thumbnail} alt="" style={{ width:'100%', height:140,
                    objectFit:'cover', borderRadius:8, marginBottom:12 }}
                    onError={e => e.target.style.display='none'} />
                )}
                <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:8 }}>
                  <span style={{ fontSize:11, color:'#6366f1', fontWeight:600, background:'rgba(99,102,241,0.15)',
                    padding:'2px 8px', borderRadius:5 }}>
                    {a.author || 'News'}
                  </span>
                  <span style={{ fontSize:11, color:'#475569' }}>{timeAgo(a.pubDate)}</span>
                </div>
                <div style={{ fontSize:14, fontWeight:600, color:'#e2e8f0', marginBottom:8, lineHeight:1.5 }}>
                  {a.title}
                </div>
                <div style={{ fontSize:12, color:'#64748b', lineHeight:1.6,
                  display:'-webkit-box', WebkitLineClamp:3, WebkitBoxOrient:'vertical', overflow:'hidden' }}>
                  {stripHtml(a.description)}
                </div>
                <div style={{ display:'flex', alignItems:'center', gap:4, marginTop:12,
                  fontSize:11, color:'#6366f1' }}>
                  Read more <ExternalLink size={10}/>
                </div>
              </div>
            </a>
          ))}
        </div>
      )}

      {!loading && articles.length === 0 && !error && (
        <div style={{ textAlign:'center', padding:'3rem', color:'#475569' }}>
          <div style={{ fontSize:32, marginBottom:'1rem' }}>📰</div>
          <div>No articles found for this topic.</div>
        </div>
      )}

      <style>{`@keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }`}</style>
    </div>
  )
}
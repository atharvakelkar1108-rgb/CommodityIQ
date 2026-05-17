import { Routes, Route, NavLink } from 'react-router-dom'
import {
  LayoutDashboard, TrendingUp, Bell, Briefcase, Newspaper,
  BarChart2, LayoutGrid, Flame, Sun, Moon, GitBranch,
  Calendar, Target, Globe, Download, LineChart, Brain, History,
} from 'lucide-react'
import Dashboard          from './components/Dashboard'
import PredictPage        from './components/PredictPage'
import AlertsPage         from './components/AlertsPage'
import PortfolioPage      from './components/PortfolioPage'
import NewsPage           from './components/NewsPage'
import ComparePage        from './components/ComparePage'
import HeatmapPage        from './components/HeatmapPage'
import MCXPage            from './components/MCXPage'
import SeasonalPage       from './components/SeasonalPage'
import CorrelationPage    from './components/CorrelationPage'
import PriceTargetPage    from './components/PriceTargetPage'
import MarketCalendarPage from './components/MarketCalendarPage'
import ScraperPage        from './components/ScraperPage'
import ExportPage         from './components/ExportPage'
import StocksIndicatorsPage from './components/StocksIndicatorsPage'
import SentimentQuantPage from './components/SentimentQuantPage'
import BacktestQuantPage  from './components/BacktestQuantPage'
import { useWebSocket }   from './hooks/useWebSocket'
import { useTheme }       from './context/ThemeContext'

export default function App() {
  const ws = useWebSocket()
  const { theme, toggle } = useTheme()
  const isDark = theme === 'dark'

  const nav = [
    { to:'/',            icon:LayoutDashboard, label:'Dashboard'    },
    { to:'/portfolio',   icon:Briefcase,       label:'Portfolio'    },
    { to:'/heatmap',     icon:LayoutGrid,      label:'Heatmap'      },
    { to:'/mcx',         icon:Flame,           label:'MCX India'    },
    { to:'/predict',     icon:TrendingUp,      label:'Predictions'  },
    { to:'/target',      icon:Target,          label:'Price Target' },
    { to:'/compare',     icon:BarChart2,       label:'Compare'      },
    { to:'/seasonal',    icon:Calendar,        label:'Seasonal'     },
    { to:'/correlation', icon:GitBranch,       label:'Correlation'  },
    { to:'/scraper',     icon:Globe,           label:'Web Scraper'  },
    { to:'/export',      icon:Download,        label:'Export Data'  },
    { to:'/news',        icon:Newspaper,       label:'News'         },
    { to:'/stocks',      icon:LineChart,      label:'Stocks & TA'  },
    { to:'/sentiment',   icon:Brain,          label:'Sentiment'    },
    { to:'/backtest',    icon:History,        label:'Backtest'     },
    { to:'/calendar',    icon:Calendar,        label:'MCX Calendar' },
    { to:'/alerts',      icon:Bell,            label:'Alerts'       },
  ]

  return (
    <div style={{ display:'flex', minHeight:'100vh', background:'var(--bg-primary)' }}>
      <aside style={{ width:210, background:'var(--bg-subtle)', borderRight:'1px solid var(--border)',
        display:'flex', flexDirection:'column', padding:'1rem 0.75rem',
        gap:2, flexShrink:0, overflowY:'auto' }}>

        <div style={{ marginBottom:'1rem', padding:'0 4px', display:'flex', alignItems:'center', gap:10 }}>
          <div style={{ width:34, height:34, borderRadius:10, display:'flex',
            alignItems:'center', justifyContent:'center', fontSize:18,
            background:'linear-gradient(135deg,#6366f1,#8b5cf6)', flexShrink:0 }}>📈</div>
          <div>
            <div style={{ fontWeight:700, fontSize:13, color:'var(--text-primary)' }}>CommodityIQ</div>
            <div style={{ fontSize:10, color:'var(--text-muted)' }}>Market Intelligence</div>
          </div>
        </div>

        {nav.map(({ to, icon:Icon, label }) => (
          <NavLink key={to} to={to} end={to==='/'} style={({ isActive }) => ({
            display:'flex', alignItems:'center', gap:9, padding:'7px 10px',
            borderRadius:7, fontSize:12, fontWeight:500, textDecoration:'none',
            color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
            background: isActive ? 'var(--bg-surface)' : 'transparent',
            transition:'all 0.15s'
          })}>
            <Icon size={14}/>{label}
          </NavLink>
        ))}

        <div style={{ marginTop:'auto', paddingTop:8 }}>
          <button onClick={toggle}
            style={{ width:'100%', display:'flex', alignItems:'center', gap:9,
              padding:'7px 10px', background:'var(--bg-surface)',
              border:'1px solid var(--border)', borderRadius:7,
              cursor:'pointer', marginBottom:8, color:'var(--text-secondary)',
              fontSize:12, fontWeight:500 }}>
            {isDark ? <Sun size={13} color="#f59e0b"/> : <Moon size={13} color="#6366f1"/>}
            {isDark ? 'Light Mode' : 'Dark Mode'}
          </button>

          <div style={{ padding:10, background:'var(--bg-surface)',
            border:'1px solid var(--border)', borderRadius:9 }}>
            <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:3 }}>
              <span className="live-dot"
                style={{ background: ws.connected ? '#22c55e' : '#ef4444', width:7, height:7 }}/>
              <span style={{ fontSize:11, fontWeight:500,
                color: ws.connected ? '#22c55e' : '#ef4444' }}>
                {ws.connected ? 'Live' : 'Offline'}
              </span>
            </div>
            <div style={{ fontSize:10, color:'var(--text-hint)' }}>
              {Object.keys(ws.prices).length} commodities · {ws.lastUpdate?.toLocaleTimeString()||'—'}
            </div>
          </div>
        </div>
      </aside>

      <main style={{ flex:1, overflow:'auto', padding:'1.5rem' }}>
        <Routes>
          <Route path="/"            element={<Dashboard        wsData={ws}/>}/>
          <Route path="/portfolio"   element={<PortfolioPage    wsData={ws}/>}/>
          <Route path="/heatmap"     element={<HeatmapPage      wsData={ws}/>}/>
          <Route path="/mcx"         element={<MCXPage          wsData={ws}/>}/>
          <Route path="/predict"     element={<PredictPage      wsData={ws}/>}/>
          <Route path="/target"      element={<PriceTargetPage  wsData={ws}/>}/>
          <Route path="/compare"     element={<ComparePage      wsData={ws}/>}/>
          <Route path="/seasonal"    element={<SeasonalPage/>}/>
          <Route path="/correlation" element={<CorrelationPage/>}/>
          <Route path="/scraper"     element={<ScraperPage/>}/>
          <Route path="/export"      element={<ExportPage   wsData={ws}/>}/>
          <Route path="/news"        element={<NewsPage/>}/>
          <Route path="/stocks"     element={<StocksIndicatorsPage/>}/>
          <Route path="/sentiment"  element={<SentimentQuantPage/>}/>
          <Route path="/backtest"   element={<BacktestQuantPage/>}/>
          <Route path="/calendar"    element={<MarketCalendarPage/>}/>
          <Route path="/alerts"      element={<AlertsPage       wsData={ws}/>}/>
        </Routes>
      </main>
    </div>
  )
}
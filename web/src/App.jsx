import { useState, useEffect } from 'react'
import { Microscope, BarChart3, Layers, FolderOpen } from 'lucide-react'
import { checkHealth } from './api'
import AnalyzePage from './components/AnalyzePage'
import BackboneComparison from './components/BackboneComparison'
import TrainingDashboard from './components/TrainingDashboard'
import BatchAnalysis from './components/BatchAnalysis'

const TABS = [
  { id: 'analyze', label: 'Analyze', icon: Microscope },
  { id: 'backbones', label: 'Backbones', icon: Layers },
  { id: 'training', label: 'Training', icon: BarChart3 },
  { id: 'batch', label: 'Batch', icon: FolderOpen },
]

export default function App() {
  const [tab, setTab] = useState('analyze')
  const [health, setHealth] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }))
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <div className="h-0.5 bg-gradient-to-r from-transparent via-rose-300 to-transparent" />

      <header className="bg-umb-panel border-b border-umb-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Microscope className="w-6 h-6 text-rose-300" />
          <h1 className="text-xl font-display font-bold tracking-tight">
            LEX<span className="text-rose-300">AI</span>
          </h1>
          <span className="text-xs text-umb-dim ml-2">Explainable Leukemia Detection</span>
        </div>
        <div className="flex items-center gap-4">
          {health && (
            <span className={`text-xs px-2 py-1 rounded-full ${
              health.status === 'healthy'
                ? 'bg-emerald-900/30 text-emerald-300'
                : 'bg-rose-900/30 text-rose-300'
            }`}>
              {health.status === 'healthy'
                ? `Model loaded (${health.device})`
                : 'API offline'}
            </span>
          )}
        </div>
      </header>

      <nav className="bg-umb-panel/50 border-b border-umb-border px-6">
        <div className="flex gap-1">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
                tab === id
                  ? 'border-rose-300 text-rose-300'
                  : 'border-transparent text-umb-muted hover:text-[#f0e8ea]'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        {tab === 'analyze' && <AnalyzePage result={result} setResult={setResult} />}
        {tab === 'backbones' && <BackboneComparison result={result} />}
        {tab === 'training' && <TrainingDashboard />}
        {tab === 'batch' && <BatchAnalysis />}
      </main>
    </div>
  )
}

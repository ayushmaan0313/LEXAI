import { useState, useCallback } from 'react'
import { Upload, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import { analyzeImage } from '../api'

const CLASS_COLORS = {
  Normal: 'bg-emerald-300', ALL: 'bg-rose-300', AML: 'bg-orange-200', CML: 'bg-amber-200',
}
const CLASS_RING = {
  Normal: 'ring-emerald-300', ALL: 'ring-rose-300', AML: 'ring-orange-200', CML: 'ring-amber-200',
}

function ProbabilityBar({ name, value }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 text-sm text-umb-muted">{name}</span>
      <div className="flex-1 h-3 bg-umb-raised rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${CLASS_COLORS[name] || 'bg-rose-300'}`}
          style={{ width: `${pct}%`, opacity: 0.8 }}
        />
      </div>
      <span className="w-14 text-right text-sm font-mono">{pct}%</span>
    </div>
  )
}

function UncertaintyPanel({ result }) {
  if (!result.prediction_variance || Object.keys(result.prediction_variance).length === 0) return null
  return (
    <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
      <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Uncertainty (MC Dropout)</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-umb-dim">Confidence</span>
          <p className="font-mono text-lg">{(result.confidence * 100).toFixed(1)}%</p>
        </div>
        <div>
          <span className="text-umb-dim">Status</span>
          <p className={`font-semibold ${result.is_uncertain ? 'text-amber-200' : 'text-emerald-300'}`}>
            {result.is_uncertain ? 'Uncertain' : 'Confident'}
          </p>
        </div>
      </div>
      {Object.keys(result.prediction_variance).length > 0 && (
        <div className="mt-3 space-y-1">
          <span className="text-xs text-umb-dim">Variance per class</span>
          {Object.entries(result.prediction_variance).map(([cls, v]) => (
            <div key={cls} className="flex justify-between text-xs font-mono">
              <span className="text-umb-muted">{cls}</span>
              <span>{v.toFixed(4)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function CellAnalysisTable({ cells }) {
  const [expanded, setExpanded] = useState(false)
  if (!cells || cells.length === 0) return null
  const shown = expanded ? cells : cells.slice(0, 5)
  return (
    <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-display font-semibold text-umb-soft">
          Per-Cell Analysis ({cells.length} cells)
        </h3>
        {cells.length > 5 && (
          <button onClick={() => setExpanded(!expanded)} className="text-xs text-rose-300 flex items-center gap-1">
            {expanded ? <><ChevronUp className="w-3 h-3" /> Less</> : <><ChevronDown className="w-3 h-3" /> Show all</>}
          </button>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-umb-dim text-xs border-b border-umb-border">
              <th className="text-left py-1">#</th>
              <th className="text-left py-1">Class</th>
              <th className="text-right py-1">Confidence</th>
              <th className="text-right py-1">Position</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((cell, i) => (
              <tr key={i} className="border-b border-umb-border/50">
                <td className="py-1 text-umb-dim">{cell.cell_id}</td>
                <td className="py-1">
                  <span className={`inline-block w-2 h-2 rounded-full mr-2 ${CLASS_COLORS[cell.predicted_class] || 'bg-umb-muted'}`} />
                  {cell.predicted_class}
                </td>
                <td className="py-1 text-right font-mono">{(cell.confidence * 100).toFixed(1)}%</td>
                <td className="py-1 text-right text-umb-dim font-mono text-xs">
                  ({cell.centroid?.x}, {cell.centroid?.y})
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function AnalyzePage({ result, setResult }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setPreview(URL.createObjectURL(file))
    setLoading(true)
    setError(null)
    try {
      const res = await analyzeImage(file)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [setResult])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('image/')) handleFile(file)
  }, [handleFile])

  return (
    <div className="space-y-6">
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
          dragOver ? 'border-rose-300 bg-rose-950/20' : 'border-umb-border hover:border-umb-border-hl'
        }`}
        onClick={() => document.getElementById('file-input').click()}
      >
        <Upload className="w-8 h-8 mx-auto mb-3 text-umb-dim" />
        <p className="text-umb-muted">Drop a blood smear image here or click to upload</p>
        <p className="text-xs text-umb-dim mt-1">JPG, PNG, BMP, TIFF</p>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {loading && (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-2 border-rose-300 border-t-transparent rounded-full animate-spin" />
          <p className="mt-3 text-umb-muted">Analyzing image...</p>
        </div>
      )}

      {error && (
        <div className="bg-rose-950/30 border border-rose-800/50 rounded-xl p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-300 shrink-0" />
          <span className="text-rose-200">{error}</span>
        </div>
      )}

      {result && !loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
              <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Input Image</h3>
              {preview && <img src={preview} alt="Input" className="rounded-lg w-full" />}
            </div>

            {result.gradcam_heatmap && (
              <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
                <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Grad-CAM Explanation</h3>
                <img src={result.gradcam_heatmap} alt="GradCAM" className="rounded-lg w-full" />
              </div>
            )}

            {result.gnn_graph_visualization && (
              <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
                <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Cell Graph (GNN)</h3>
                <img src={result.gnn_graph_visualization} alt="GNN Graph" className="rounded-lg w-full" />
              </div>
            )}
          </div>

          <div className="space-y-4">
            <div className={`bg-umb-panel rounded-xl p-5 border-2 ${CLASS_RING[result.predicted_class] || 'border-umb-border'}`}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-xs text-umb-dim uppercase tracking-wider font-display">Prediction</p>
                  <p className="text-2xl font-display font-bold">{result.predicted_class}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-umb-dim font-display">Confidence</p>
                  <p className="text-2xl font-bold font-mono">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="space-y-2">
                {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                  <ProbabilityBar key={cls} name={cls} value={prob} />
                ))}
              </div>
            </div>

            <div className="bg-umb-panel rounded-xl p-4 border border-umb-border">
              <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Spatial Analysis</h3>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-xs text-umb-dim">Cells</p>
                  <p className="text-xl font-bold font-mono">{result.cell_count}</p>
                </div>
                <div>
                  <p className="text-xs text-umb-dim">Blast %</p>
                  <p className="text-xl font-bold font-mono">{result.blast_percentage?.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-xs text-umb-dim">Spatial Score</p>
                  <p className="text-xl font-bold font-mono">{result.spatial_pattern_score?.toFixed(3)}</p>
                </div>
              </div>
            </div>

            <UncertaintyPanel result={result} />
            <CellAnalysisTable cells={result.cell_analysis} />
          </div>
        </div>
      )}
    </div>
  )
}

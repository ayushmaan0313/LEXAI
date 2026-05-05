import { useState, useCallback } from 'react'
import { Upload, Download, AlertTriangle, CheckCircle } from 'lucide-react'
import { analyzeImage } from '../api'

const CLASS_COLORS = {
  Normal: 'text-emerald-300', ALL: 'text-rose-300', AML: 'text-orange-200', CML: 'text-amber-200',
}

export default function BatchAnalysis() {
  const [files, setFiles] = useState([])
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)

  const handleFiles = useCallback((fileList) => {
    const images = Array.from(fileList).filter(f => f.type.startsWith('image/'))
    setFiles(images)
    setResults([])
  }, [])

  const runBatch = useCallback(async () => {
    if (files.length === 0) return
    setLoading(true)
    setProgress(0)
    const all = []
    for (let i = 0; i < files.length; i++) {
      try {
        const result = await analyzeImage(files[i])
        all.push({ file: files[i].name, ...result, error: null })
      } catch (e) {
        all.push({ file: files[i].name, error: e.message })
      }
      setProgress(((i + 1) / files.length * 100))
      setResults([...all])
    }
    setLoading(false)
  }, [files])

  const exportCSV = useCallback(() => {
    if (results.length === 0) return
    const headers = ['File', 'Predicted Class', 'Confidence', 'Cell Count', 'Blast %', 'Normal', 'ALL', 'AML', 'CML']
    const rows = results.filter(r => !r.error).map(r => [
      r.file, r.predicted_class, r.confidence?.toFixed(4),
      r.cell_count, r.blast_percentage?.toFixed(1),
      r.probabilities?.Normal?.toFixed(4), r.probabilities?.ALL?.toFixed(4),
      r.probabilities?.AML?.toFixed(4), r.probabilities?.CML?.toFixed(4),
    ])
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'lexai_batch_results.csv'; a.click()
  }, [results])

  const exportJSON = useCallback(() => {
    if (results.length === 0) return
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'lexai_batch_results.json'; a.click()
  }, [results])

  const successResults = results.filter(r => !r.error)
  const classCounts = {}
  successResults.forEach(r => {
    classCounts[r.predicted_class] = (classCounts[r.predicted_class] || 0) + 1
  })

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-display font-bold">Batch Analysis</h2>

      <div
        className="border-2 border-dashed border-umb-border hover:border-umb-border-hl rounded-xl p-8 text-center cursor-pointer transition-colors"
        onClick={() => document.getElementById('batch-input').click()}
      >
        <Upload className="w-8 h-8 mx-auto mb-3 text-umb-dim" />
        <p className="text-umb-muted">Select multiple blood smear images</p>
        <p className="text-xs text-umb-dim mt-1">JPG, PNG, BMP, TIFF</p>
        <input
          id="batch-input" type="file" accept="image/*" multiple className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-umb-muted">{files.length} images selected</p>
          <button onClick={runBatch} disabled={loading}
            className="px-4 py-2 bg-rose-400/80 hover:bg-rose-300/80 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors text-umb-bg">
            {loading ? `Analyzing... ${progress.toFixed(0)}%` : 'Analyze All'}
          </button>
        </div>
      )}

      {loading && (
        <div className="w-full bg-umb-raised rounded-full h-2">
          <div className="bg-rose-400/80 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
        </div>
      )}

      {successResults.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-umb-panel rounded-xl p-4 border border-umb-border text-center">
            <p className="text-xs text-umb-dim">Total</p>
            <p className="text-2xl font-bold font-mono">{successResults.length}</p>
          </div>
          {['Normal', 'ALL', 'AML', 'CML'].map(cls => (
            <div key={cls} className="bg-umb-panel rounded-xl p-4 border border-umb-border text-center">
              <p className="text-xs text-umb-dim">{cls}</p>
              <p className={`text-2xl font-bold font-mono ${CLASS_COLORS[cls]}`}>{classCounts[cls] || 0}</p>
            </div>
          ))}
        </div>
      )}

      {results.length > 0 && (
        <div className="bg-umb-panel rounded-xl border border-umb-border overflow-hidden">
          <div className="flex items-center justify-between p-4 border-b border-umb-border">
            <h3 className="text-sm font-display font-semibold text-umb-soft">Results</h3>
            <div className="flex gap-2">
              <button onClick={exportCSV} className="flex items-center gap-1 px-3 py-1.5 bg-umb-border hover:bg-umb-border-hl rounded-lg text-xs transition-colors">
                <Download className="w-3 h-3" /> CSV
              </button>
              <button onClick={exportJSON} className="flex items-center gap-1 px-3 py-1.5 bg-umb-border hover:bg-umb-border-hl rounded-lg text-xs transition-colors">
                <Download className="w-3 h-3" /> JSON
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-umb-dim text-xs border-b border-umb-border">
                  <th className="text-left px-4 py-2">File</th>
                  <th className="text-left px-4 py-2">Class</th>
                  <th className="text-right px-4 py-2">Confidence</th>
                  <th className="text-right px-4 py-2">Cells</th>
                  <th className="text-right px-4 py-2">Blast %</th>
                  <th className="text-center px-4 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-umb-border/50 hover:bg-umb-raised/30">
                    <td className="px-4 py-2 font-mono text-xs max-w-48 truncate">{r.file}</td>
                    {r.error ? (
                      <>
                        <td colSpan={4} className="px-4 py-2 text-rose-300 text-xs">{r.error}</td>
                        <td className="px-4 py-2 text-center">
                          <AlertTriangle className="w-4 h-4 text-rose-300 mx-auto" />
                        </td>
                      </>
                    ) : (
                      <>
                        <td className={`px-4 py-2 font-semibold ${CLASS_COLORS[r.predicted_class]}`}>
                          {r.predicted_class}
                        </td>
                        <td className="px-4 py-2 text-right font-mono">{(r.confidence * 100).toFixed(1)}%</td>
                        <td className="px-4 py-2 text-right">{r.cell_count}</td>
                        <td className="px-4 py-2 text-right">{r.blast_percentage?.toFixed(1)}%</td>
                        <td className="px-4 py-2 text-center">
                          <CheckCircle className={`w-4 h-4 mx-auto ${r.is_uncertain ? 'text-amber-200' : 'text-emerald-300'}`} />
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

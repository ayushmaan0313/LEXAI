const API_BASE = '/api'

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Analysis failed')
  }
  return res.json()
}

export async function analyzeBatch(files) {
  const results = []
  for (const file of files) {
    try {
      const result = await analyzeImage(file)
      results.push({ file: file.name, ...result, error: null })
    } catch (e) {
      results.push({ file: file.name, error: e.message })
    }
  }
  return results
}

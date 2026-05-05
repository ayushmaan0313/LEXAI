import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid
} from 'recharts'
import { Play, Square, RefreshCw } from 'lucide-react'

const TOOLTIP_STYLE = { backgroundColor: '#14121a', border: '1px solid #2c2434', color: '#f0e8ea' }

function generateDemoData(epochs) {
  const data = []
  for (let i = 1; i <= epochs; i++) {
    const progress = i / epochs
    data.push({
      epoch: i,
      train_loss: 2.0 * Math.exp(-3 * progress) + 0.1 + Math.random() * 0.05,
      val_loss: 2.2 * Math.exp(-2.5 * progress) + 0.15 + Math.random() * 0.08,
      train_acc: 1 - Math.exp(-3 * progress) * 0.7 + Math.random() * 0.02,
      val_acc: 1 - Math.exp(-2.5 * progress) * 0.75 + Math.random() * 0.03,
      val_ece: 0.15 * Math.exp(-2 * progress) + 0.02 + Math.random() * 0.01,
    })
  }
  return data
}

const STAGE_INFO = [
  { name: 'Stage 1', desc: 'Frozen backbones — train heads + fusion', color: 'bg-rose-300' },
  { name: 'Stage 2', desc: 'Full fine-tuning — low LR end-to-end', color: 'bg-emerald-300' },
  { name: 'Stage 3', desc: 'Temperature calibration — LBFGS on val', color: 'bg-amber-200' },
]

export default function TrainingDashboard() {
  const [config, setConfig] = useState({
    epochs: 30,
    batch_size: 16,
    lr: '3e-4',
    finetune_lr: '5e-5',
    freeze_epochs: 10,
    patience: 15,
    label_smoothing: 0.1,
    use_vit: true,
    use_gnn: true,
  })

  const demoData = generateDemoData(config.epochs)
  const currentStage = 2
  const currentEpoch = config.epochs

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-display font-bold">Training Dashboard</h2>
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-4 py-2 bg-rose-400/80 hover:bg-rose-300/80 rounded-lg text-sm font-medium transition-colors text-umb-bg">
            <Play className="w-4 h-4" /> Start Training
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-umb-border hover:bg-umb-border-hl rounded-lg text-sm font-medium transition-colors">
            <RefreshCw className="w-4 h-4" /> Reset
          </button>
        </div>
      </div>

      <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
        <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">Training Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SliderField label="Epochs" value={config.epochs} min={10} max={100} step={5}
            onChange={(v) => setConfig({ ...config, epochs: v })} />
          <SliderField label="Batch Size" value={config.batch_size} min={4} max={64} step={4}
            onChange={(v) => setConfig({ ...config, batch_size: v })} />
          <SliderField label="Freeze Epochs" value={config.freeze_epochs} min={0} max={30} step={1}
            onChange={(v) => setConfig({ ...config, freeze_epochs: v })} />
          <SliderField label="Patience" value={config.patience} min={5} max={30} step={1}
            onChange={(v) => setConfig({ ...config, patience: v })} />
          <SliderField label="Label Smoothing" value={config.label_smoothing} min={0} max={0.3} step={0.01}
            onChange={(v) => setConfig({ ...config, label_smoothing: v })} format={(v) => v.toFixed(2)} />
          <div>
            <label className="text-xs text-umb-dim">Stage 1 LR</label>
            <select value={config.lr} onChange={(e) => setConfig({ ...config, lr: e.target.value })}
              className="w-full mt-1 bg-umb-raised border border-umb-border rounded-lg px-3 py-2 text-sm">
              <option value="1e-3">1e-3</option>
              <option value="3e-4">3e-4</option>
              <option value="1e-4">1e-4</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-umb-dim">Stage 2 LR</label>
            <select value={config.finetune_lr} onChange={(e) => setConfig({ ...config, finetune_lr: e.target.value })}
              className="w-full mt-1 bg-umb-raised border border-umb-border rounded-lg px-3 py-2 text-sm">
              <option value="1e-4">1e-4</option>
              <option value="5e-5">5e-5</option>
              <option value="1e-5">1e-5</option>
            </select>
          </div>
          <div className="flex flex-col gap-2">
            <ToggleField label="ViT Backbone" checked={config.use_vit}
              onChange={(v) => setConfig({ ...config, use_vit: v })} />
            <ToggleField label="GNN Pathway" checked={config.use_gnn}
              onChange={(v) => setConfig({ ...config, use_gnn: v })} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {STAGE_INFO.map((stage, i) => (
          <div key={i} className={`rounded-xl p-4 border ${
            i + 1 === currentStage ? 'border-rose-400/50 bg-rose-950/20' :
            i + 1 < currentStage ? 'border-emerald-800/50 bg-emerald-950/10' :
            'border-umb-border bg-umb-panel'
          }`}>
            <div className="flex items-center gap-2 mb-1">
              <div className={`w-2 h-2 rounded-full ${stage.color}`} />
              <h4 className="text-sm font-display font-semibold">{stage.name}</h4>
              {i + 1 < currentStage && <span className="text-xs text-emerald-300">Done</span>}
              {i + 1 === currentStage && <span className="text-xs text-rose-300">Active</span>}
            </div>
            <p className="text-xs text-umb-dim">{stage.desc}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
          <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">Loss</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={demoData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2c2434" />
              <XAxis dataKey="epoch" stroke="#8a7a82" fontSize={12} />
              <YAxis stroke="#8a7a82" fontSize={12} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend />
              <Line type="monotone" dataKey="train_loss" stroke="#fda4af" name="Train" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_loss" stroke="#c4b5fd" name="Val" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
          <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">Accuracy</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={demoData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2c2434" />
              <XAxis dataKey="epoch" stroke="#8a7a82" fontSize={12} />
              <YAxis stroke="#8a7a82" fontSize={12} domain={[0, 1]} />
              <Tooltip contentStyle={TOOLTIP_STYLE}
                formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Legend />
              <Line type="monotone" dataKey="train_acc" stroke="#fda4af" name="Train" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_acc" stroke="#6ee7b7" name="Val" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-umb-panel rounded-xl p-5 border border-umb-border lg:col-span-2">
          <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">
            Calibration (ECE)
            <span className="text-xs text-umb-dim ml-2">Target: &lt; 0.05</span>
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={demoData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2c2434" />
              <XAxis dataKey="epoch" stroke="#8a7a82" fontSize={12} />
              <YAxis stroke="#8a7a82" fontSize={12} domain={[0, 0.2]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Line type="monotone" dataKey="val_ece" stroke="#fde68a" name="ECE" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey={() => 0.05} stroke="#fda4af" strokeDasharray="5 5" name="Target" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

function SliderField({ label, value, min, max, step, onChange, format }) {
  return (
    <div>
      <div className="flex justify-between">
        <label className="text-xs text-umb-dim">{label}</label>
        <span className="text-xs font-mono text-umb-soft">{format ? format(value) : value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full mt-1 accent-rose-400" />
    </div>
  )
}

function ToggleField({ label, checked, onChange }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <div className={`w-8 h-4 rounded-full transition-colors ${checked ? 'bg-rose-400' : 'bg-umb-border'}`}
        onClick={() => onChange(!checked)}>
        <div className={`w-4 h-4 bg-white rounded-full transition-transform ${checked ? 'translate-x-4' : ''}`} />
      </div>
      <span className="text-xs text-umb-muted">{label}</span>
    </label>
  )
}

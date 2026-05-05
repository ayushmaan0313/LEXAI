import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = {
  efficientnet: '#c4b5fd', resnet50: '#fda4af', densenet121: '#6ee7b7', vit: '#fde68a',
}

const TOOLTIP_STYLE = { backgroundColor: '#14121a', border: '1px solid #2c2434', color: '#f0e8ea' }

export default function BackboneComparison({ result }) {
  if (!result) {
    return (
      <div className="text-center py-20 text-umb-dim">
        <p className="text-lg font-display">No analysis result yet.</p>
        <p className="text-sm mt-2">Analyze an image first to see backbone comparison.</p>
      </div>
    )
  }

  const weights = result.cnn_backbone_weights || {}
  const pieData = Object.entries(weights).map(([name, value]) => ({
    name, value: parseFloat(value.toFixed(4)),
  }))

  const probData = Object.entries(result.probabilities || {}).map(([cls, prob]) => ({
    class: cls, probability: parseFloat((prob * 100).toFixed(1)),
  }))

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-display font-bold">Per-Backbone Comparison</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
          <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">Learned Fusion Weights</h3>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, value }) => `${name} ${(value * 100).toFixed(1)}%`}
                  labelLine={true}
                >
                  {pieData.map((entry) => (
                    <Cell key={entry.name} fill={COLORS[entry.name] || '#c4b5fd'} fillOpacity={0.85} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `${(v * 100).toFixed(2)}%`} contentStyle={TOOLTIP_STYLE} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-umb-dim mt-2 text-center">
            Weights are learned during training via softmax normalization
          </p>
        </div>

        <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
          <h3 className="text-sm font-display font-semibold text-umb-soft mb-4">Class Probabilities</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={probData}>
              <XAxis dataKey="class" stroke="#8a7a82" fontSize={12} />
              <YAxis stroke="#8a7a82" fontSize={12} domain={[0, 100]} unit="%" />
              <Tooltip formatter={(v) => `${v}%`} contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                {probData.map((entry) => {
                  const color = { Normal: '#6ee7b7', ALL: '#fda4af', AML: '#fed7aa', CML: '#fde68a' }
                  return <Cell key={entry.class} fill={color[entry.class] || '#c4b5fd'} fillOpacity={0.8} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {pieData.map(({ name, value }) => (
          <div key={name} className="bg-umb-panel rounded-xl p-4 border border-umb-border">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[name], opacity: 0.85 }} />
              <h4 className="text-sm font-display font-semibold capitalize">{name}</h4>
            </div>
            <p className="text-2xl font-bold font-mono">{(value * 100).toFixed(1)}%</p>
            <p className="text-xs text-umb-dim mt-1">fusion weight</p>
          </div>
        ))}
      </div>

      <div className="bg-umb-panel rounded-xl p-5 border border-umb-border">
        <h3 className="text-sm font-display font-semibold text-umb-soft mb-3">Architecture Details</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          {[
            { label: 'EfficientNet-B0', desc: 'Compound scaling, lightweight' },
            { label: 'ResNet-50', desc: 'Deep residual connections' },
            { label: 'DenseNet-121', desc: 'Dense feature reuse' },
            { label: 'ViT-B/16', desc: 'Vision transformer, global attention' },
          ].map(({ label, desc }) => (
            <div key={label}>
              <p className="font-medium">{label}</p>
              <p className="text-xs text-umb-dim">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

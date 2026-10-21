import React, { useMemo } from "react";

const NODE_COLORS = {
  Customer: "#5B8DBE",
  Account: "#B98530",
  Device: "#B54B42",
  Address: "#8B6FB5",
  Email: "#4B7A63",
  Broker: "#7C8CA3",
};

export default function FraudGraph({ nodes, edges }) {
  const layout = useMemo(() => {
    if (!nodes || nodes.length === 0) return null;
    const width = 560, height = 340;
    const cx = width / 2, cy = height / 2;
    const center = nodes[0];
    const rest = nodes.slice(1);
    const radius = Math.min(width, height) / 2 - 55;

    const positioned = { [center.id]: { x: cx, y: cy, ...center } };
    rest.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, rest.length);
      positioned[n.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle), ...n };
    });
    return { width, height, positioned };
  }, [nodes]);

  if (!layout) {
    return <p className="text-sm text-ink-dim font-serif italic">No connected entities found for this case.</p>;
  }

  const { width, height, positioned } = layout;
  const nodeList = Object.values(positioned);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {edges.map((e, i) => {
          const a = positioned[e.source], b = positioned[e.target];
          if (!a || !b) return null;
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#1E2D42" strokeWidth="1.5" />;
        })}
        {nodeList.map((n) => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r={n.type === "Customer" ? 7 : 5}
                    fill={NODE_COLORS[n.type] || "#4B5A70"} stroke="#0B1420" strokeWidth="2" />
            <text x={n.x} y={n.y - 11} textAnchor="middle" fontSize="9" fill="#7C8CA3" fontFamily="IBM Plex Mono, monospace">
              {(n.name || n.id).length > 14 ? (n.name || n.id).slice(0, 12) + "…" : (n.name || n.id)}
            </text>
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 mt-2 text-[11px] font-mono text-ink-dim">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}

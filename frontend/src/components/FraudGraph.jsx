import React, { useCallback, useMemo, useRef, useState } from "react";

const NODE_COLORS = {
  Customer: "#5B8DBE",
  Account: "#B98530",
  Device: "#B54B42",
  Address: "#8B6FB5",
  Email: "#4B7A63",
  Broker: "#7C8CA3",
};

// The relation that connects a customer to the entity it shares. Used to
// label edges in the readout.
const RELATION_LABELS = {
  OWNS_ACCOUNT: "shares a bank account",
  USES_DEVICE: "shares a device",
  HAS_ADDRESS: "shares an address",
  HAS_EMAIL: "shares an email",
  FILED_VIA: "shares a broker",
};

/**
 * A small force-directed layout, hand-rolled.
 *
 * The previous version placed every node on a perfect circle, which threw
 * away the one thing this view exists to show: fraud rings are CLUSTERS,
 * and a circle makes a cluster look like everything else. A few dozen
 * lines of spring/repulsion simulation put connected entities next to
 * each other, so a ring reads as a ring.
 *
 * The simulation is bounded (fixed iteration count) and runs once per
 * graph, so it costs no more than the old layout did.
 */
function layoutGraph(nodes, edges, width, height, iterations = 220) {
  const index = new Map();
  const points = nodes.map((node, i) => {
    index.set(node.id, i);
    // deterministic start positions -- no random jitter, so the same
    // graph always renders the same way
    const angle = (2 * Math.PI * i) / Math.max(1, nodes.length);
    const radius = Math.min(width, height) / 2 - 60;
    return {
      id: node.id,
      node,
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle),
      vx: 0,
      vy: 0,
    };
  });

  const links = edges
    .map((edge) => ({
      source: index.get(edge.source),
      target: index.get(edge.target),
      relation: edge.relation,
    }))
    .filter((link) => link.source !== undefined && link.target !== undefined);

  const repulsion = 5200;
  const springLength = 92;
  const springStrength = 0.045;
  const damping = 0.86;

  for (let step = 0; step < iterations; step += 1) {
    // repulsion between every pair
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const a = points[i];
        const b = points[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distance = Math.hypot(dx, dy);
        if (distance < 0.01) {
          dx = (i % 2 === 0 ? 1 : -1) * 0.5;
          dy = (j % 2 === 0 ? 1 : -1) * 0.5;
          distance = Math.hypot(dx, dy);
        }
        const force = repulsion / (distance * distance);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    // springs along edges
    for (const link of links) {
      const a = points[link.source];
      const b = points[link.target];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(0.01, Math.hypot(dx, dy));
      const force = (distance - springLength) * springStrength;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    // centre pull keeps disconnected nodes in frame
    for (const point of points) {
      point.vx += (width / 2 - point.x) * 0.012;
      point.vy += (height / 2 - point.y) * 0.012;
      point.vx *= damping;
      point.vy *= damping;
      point.x += point.vx;
      point.y += point.vy;
    }
  }

  // Keep every node inside the frame, then scale the whole graph to fill
  // the canvas. Without this the simulation settles into a tight knot in
  // the middle of an empty viewBox, which wastes most of the space and
  // makes the labels collide.
  const pad = 34;
  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY, 1.6);
  const offsetX = (width - spanX * scale) / 2 - minX * scale;
  const offsetY = (height - spanY * scale) / 2 - minY * scale;

  for (const point of points) {
    point.x = Math.max(pad, Math.min(width - pad, point.x * scale + offsetX));
    point.y = Math.max(pad, Math.min(height - pad, point.y * scale + offsetY));
  }

  const positioned = new Map();
  for (const point of points) positioned.set(point.id, point);

  return { positioned, links };
}

/**
 * The SVG canvas.
 *
 * React.memo + imperative hover handling, and that combination is the
 * whole point of this component's structure.
 *
 * The earlier version tracked the hovered node in React state, which
 * meant every mouse move over the graph re-rendered the whole SVG and
 * re-rendered the readout panel. The readout panel then changed height,
 * the page height changed with it, the browser adjusted the scroll
 * position, the node moved out from under the cursor, the hover ended,
 * the panel shrank, the page scrolled back -- and the whole thing
 * oscillated. That is the "shiver".
 *
 * So: the canvas is memoized and never re-renders while hovering, the
 * highlight is applied by setting a data attribute on the DOM elements
 * directly, and only the small readout panel re-renders (in a box with
 * a fixed height, so nothing around it can move).
 */
const GraphCanvas = React.memo(function GraphCanvas({ nodes, links, positioned, highlightIds, onHover }) {
  const width = 620;
  const height = 400;
  const svgRef = useRef(null);
  const highlightSet = useMemo(() => new Set(highlightIds || []), [highlightIds]);

  const applyHover = useCallback((nodeId, notifyReadout = true) => {
    const svg = svgRef.current;
    if (!svg) return;

    const related = new Set();
    const relations = [];
    if (nodeId) {
      related.add(nodeId);
      svg.querySelectorAll(".graph-edge").forEach((el) => {
        const source = el.getAttribute("data-source");
        const target = el.getAttribute("data-target");
        if (source === nodeId || target === nodeId) {
          related.add(source);
          related.add(target);
          el.setAttribute("data-active", "true");
          const other = source === nodeId ? target : source;
          relations.push({
            relation: el.getAttribute("data-relation"),
            other: other,
          });
        } else {
          el.removeAttribute("data-active");
        }
      });
    } else {
      svg.querySelectorAll(".graph-edge").forEach((el) => el.removeAttribute("data-active"));
    }

    svg.querySelectorAll(".graph-node").forEach((el) => {
      const id = el.getAttribute("data-node-id");
      if (nodeId && !related.has(id)) el.setAttribute("data-dim", "true");
      else el.removeAttribute("data-dim");
    });

    if (notifyReadout) onHover(nodeId, relations);
  }, [onHover]);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="graph-canvas"
      ref={svgRef}
      role="group"
      aria-label="Connected entity graph"
      onMouseLeave={() => applyHover(null, false)}
    >
      {links.map((link, i) => {
        const source = positioned.get(link.sourceId);
        const target = positioned.get(link.targetId);
        if (!source || !target) return null;
        const highlighted = highlightSet.has(link.sourceId) || highlightSet.has(link.targetId);
        return (
          <line
            key={i}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            className={`graph-edge ${highlighted ? "is-highlighted" : ""}`}
            data-source={link.sourceId}
            data-target={link.targetId}
            data-relation={link.relation}
          />
        );
      })}

      {nodes.map((node) => {
        const point = positioned.get(node.id);
        if (!point) return null;
        const radius = node.type === "Customer" ? 7 : 5;
        const label = node.name || node.id;
        return (
          <g
            key={node.id}
            className="graph-node"
            data-node-id={node.id}
            onMouseEnter={() => applyHover(node.id)}
            onFocus={() => applyHover(node.id)}
            onBlur={() => applyHover(null)}
            tabIndex={0}
            role="button"
            aria-label={`${node.type} ${label}`}
          >
            {highlightSet.has(node.id) && (
              <circle cx={point.x} cy={point.y} r={radius + 5} className="graph-node-halo" />
            )}
            <circle
              cx={point.x}
              cy={point.y}
              r={radius}
              fill={NODE_COLORS[node.type] || "#4B5A70"}
              stroke="#0B1420"
              strokeWidth="2"
            />
            <text
              x={point.x}
              y={point.y - 11}
              textAnchor="middle"
              fontSize="9"
              fill="#7C8CA3"
              fontFamily="IBM Plex Mono, monospace"
            >
              {label.length > 14 ? `${label.slice(0, 12)}…` : label}
            </text>
          </g>
        );
      })}
    </svg>
  );
});

function GraphReadout({ node, relations, nodeCount, linkCount }) {
  if (!node) {
    return (
      <div className="graph-readout">
        <span className="graph-readout-hint">
          Hover a node to see what it shares with whom. {nodeCount} entit{nodeCount === 1 ? "y" : "ies"},{" "}
          {linkCount} link{linkCount === 1 ? "" : "s"}.
        </span>
      </div>
    );
  }

  // capped, so the readout can never grow and shift the page under the
  // cursor -- that shift is what turned a hover into an oscillation
  return (
    <div className="graph-readout">
      <div className="graph-readout-head">
        <strong>{node.name || node.id}</strong>
        <span className="graph-readout-type">{node.type} · {relations.length} connection{relations.length === 1 ? "" : "s"}</span>
      </div>
      {relations.length > 0 ? (
        <ul>
          {relations.map((relation, i) => (
            <li key={i}>
              {RELATION_LABELS[relation.relation] || relation.relation} ·{" "}
              <em>{relation.other}</em>
            </li>
          ))}
        </ul>
      ) : <span className="graph-readout-hint">No direct connections for this entity.</span>}
    </div>
  );
}

export default function FraudGraph({ nodes, edges, highlightIds = [] }) {
  const [hover, setHover] = useState({ node: null, relations: [] });

  const width = 620;
  const height = 400;

  const { layoutNodes, layoutLinks, positioned } = useMemo(() => {
    if (!nodes || nodes.length === 0) {
      return { layoutNodes: [], layoutLinks: [], positioned: new Map() };
    }
    const { positioned, links } = layoutGraph(nodes, edges || [], width, height);
    const layoutLinks = links
      .map((link) => ({
        sourceId: nodes[link.source]?.id,
        targetId: nodes[link.target]?.id,
        relation: link.relation,
      }))
      .filter((link) => link.sourceId && link.targetId);
    return { layoutNodes: nodes, layoutLinks, positioned };
  }, [nodes, edges]);

  const handleHover = useCallback((nodeId, relations) => {
    if (!nodeId) {
      setHover({ node: null, relations: [] });
      return;
    }
    const node = layoutNodes.find((n) => n.id === nodeId) || null;
    setHover({
      node,
      relations: relations.map((r) => ({
        ...r,
        other: (layoutNodes.find((n) => n.id === r.other)?.name) || r.other,
      })),
    });
  }, [layoutNodes]);

  if (!layoutNodes.length) {
    return <p className="text-sm text-ink-dim font-serif italic">No connected entities found for this case.</p>;
  }

  return (
    <div className="graph" onMouseLeave={() => setHover({ node: null, relations: [] })}>
      <GraphCanvas
        nodes={layoutNodes}
        links={layoutLinks}
        positioned={positioned}
        highlightIds={highlightIds}
        onHover={handleHover}
      />

      <div className="graph-footer">
        <div className="graph-legend">
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <span key={type} className="graph-legend-item">
              <span className="graph-legend-dot" style={{ background: color }} />
              {type}
            </span>
          ))}
        </div>

        <GraphReadout
          node={hover.node}
          relations={hover.relations}
          nodeCount={layoutNodes.length}
          linkCount={layoutLinks.length}
        />
      </div>
    </div>
  );
}

"use client";

import React, { useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// TypeScript Interfaces
// ---------------------------------------------------------------------------

export interface CausalEdge {
  from: string;
  to: string;
  relation: string;
}

export interface CausalNode {
  node_id?: number;
  text: string;
  cause?: string;
  score?: number;
}

interface CausalTraceProps {
  edges: CausalEdge[];
  nodes?: CausalNode[];
}

// ---------------------------------------------------------------------------
// CausalTrace Component — Canvas-based causal graph visualization
// ---------------------------------------------------------------------------

export default function CausalTrace({ edges, nodes = [] }: CausalTraceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || edges.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Extract unique node labels
    const nodeLabels = new Set<string>();
    edges.forEach((e) => {
      if (e.from) nodeLabels.add(e.from);
      if (e.to) nodeLabels.add(e.to);
    });
    const labels = Array.from(nodeLabels);

    if (labels.length === 0) return;

    // Layout: Multi-orbit concentric rings to prevent node overlap
    const cx = width / 2;
    const cy = height / 2;
    const positions: Record<string, { x: number; y: number }> = {};

    if (labels.length <= 6) {
      const radius = Math.min(width, height) * 0.35;
      labels.forEach((label, i) => {
        const angle = (2 * Math.PI * i) / labels.length - Math.PI / 2;
        positions[label] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
      });
    } else if (labels.length <= 15) {
      // 2 rings
      const innerCount = Math.floor(labels.length / 3);
      const outerCount = labels.length - innerCount;
      const innerRadius = Math.min(width, height) * 0.18;
      const outerRadius = Math.min(width, height) * 0.40;
      
      labels.forEach((label, i) => {
        if (i < innerCount) {
          const angle = (2 * Math.PI * i) / innerCount - Math.PI / 2;
          positions[label] = { x: cx + innerRadius * Math.cos(angle), y: cy + innerRadius * Math.sin(angle) };
        } else {
          const ox = i - innerCount;
          const angle = (2 * Math.PI * ox) / outerCount - Math.PI / 4;
          positions[label] = { x: cx + outerRadius * Math.cos(angle), y: cy + outerRadius * Math.sin(angle) };
        }
      });
    } else {
      // 3 rings for massive graphs
      const innerCount = 5;
      const midCount = 10;
      const outerCount = labels.length - 15;
      const r1 = Math.min(width, height) * 0.12;
      const r2 = Math.min(width, height) * 0.28;
      const r3 = Math.min(width, height) * 0.45;

      labels.forEach((label, i) => {
        if (i < innerCount) {
          const angle = (2 * Math.PI * i) / innerCount;
          positions[label] = { x: cx + r1 * Math.cos(angle), y: cy + r1 * Math.sin(angle) };
        } else if (i < innerCount + midCount) {
          const mx = i - innerCount;
          const angle = (2 * Math.PI * mx) / midCount + 0.5;
          positions[label] = { x: cx + r2 * Math.cos(angle), y: cy + r2 * Math.sin(angle) };
        } else {
          const ox = i - innerCount - midCount;
          const angle = (2 * Math.PI * ox) / outerCount + 0.25;
          positions[label] = { x: cx + r3 * Math.cos(angle), y: cy + r3 * Math.sin(angle) };
        }
      });
    }

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Draw edges
    edges.forEach((edge) => {
      const from = positions[edge.from];
      const to = positions[edge.to];
      if (!from || !to) return;

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"; // Subtle silver edge
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const arrowLen = 7;
      const arrowX = to.x - 16 * Math.cos(angle);
      const arrowY = to.y - 16 * Math.sin(angle);

      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(
        arrowX - arrowLen * Math.cos(angle - 0.4),
        arrowY - arrowLen * Math.sin(angle - 0.4)
      );
      ctx.lineTo(
        arrowX - arrowLen * Math.cos(angle + 0.4),
        arrowY - arrowLen * Math.sin(angle + 0.4)
      );
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 255, 255, 0.5)"; // White-silver arrowhead
      ctx.fill();
    });

    // Draw nodes
    labels.forEach((label) => {
      const pos = positions[label];
      if (!pos) return;

      // Find matching node for score-based coloring
      const matchingNode = nodes.find(
        (n) => n.text === label || n.cause === label
      );
      const score = matchingNode?.score ?? 0.5;

      // Node circle
      const gradient = ctx.createRadialGradient(
        pos.x, pos.y, 0,
        pos.x, pos.y, 14
      );

      if (score >= 0.7) {
        gradient.addColorStop(0, "rgba(239, 68, 68, 0.95)"); // Pure sharp crimson
        gradient.addColorStop(1, "rgba(239, 68, 68, 0.1)");
      } else {
        gradient.addColorStop(0, "rgba(255, 255, 255, 0.9)"); // Sharp white
        gradient.addColorStop(1, "rgba(255, 255, 255, 0.05)");
      }

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
      ctx.strokeStyle =
        score >= 0.7 ? "rgba(239, 68, 68, 0.3)" : "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label text (truncated)
      const displayLabel =
        label.length > 25 ? label.substring(0, 22) + "..." : label;
      ctx.fillStyle = "#a1a1aa"; // text-zinc-400
      ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(displayLabel, pos.x, pos.y + 16);
    });
  }, [edges, nodes]);

  if (edges.length === 0) {
    return (
      <div className="glass-card p-6 text-center">
        <div className="text-gray-500 text-sm">
          <svg
            className="mx-auto mb-2 h-8 w-8 text-gray-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
            />
          </svg>
          No causal traces recorded yet.
          <br />
          <span className="text-gray-600 text-xs">
            Contamination events will appear here as a directed graph.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="border-b border-white/5 px-5 py-4">
        <h3 className="text-[13px] font-medium tracking-wide uppercase text-zinc-100 flex items-center gap-2">
          <svg className="h-4 w-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Contamination Causal Graph
        </h3>
        <p className="text-xs text-zinc-500 mt-1 font-light tracking-wide">
          Non-semantic directed chains defining exact failure origin vectors
        </p>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: "280px" }}
      />
    </div>
  );
}

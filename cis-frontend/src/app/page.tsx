"use client";

import React, { useState, useCallback } from "react";
import ClaimCard, { type ClaimData } from "@/components/ClaimCard";
import CausalTrace, { type CausalEdge } from "@/components/CausalTrace";
import ResearchDashboard from "@/components/ResearchDashboard";
import ExperimentSimulator from "@/components/ExperimentSimulator";

// ---------------------------------------------------------------------------
// TypeScript Interfaces — mirroring backend response schema exactly
// ---------------------------------------------------------------------------

interface AnalysisResult {
  answer: string;
  raw_answer: string;
  claims_total: number;
  claims_verifiable: number;
  contamination_rate: number;
  quarantined: ClaimData[];
  safe: ClaimData[];
  causal_trace: CausalEdge[];
  latency_ms: number;
  session_id: string;
  tolerance_set_size: number;
  containment_depth_bound: number | null;
  error?: string;
}

interface BaselineResult {
  answer: string;
  latency_ms: number;
  error?: string;
}

interface SystemStats {
  total_analyzed: number;
  avg_contamination_rate: number;
  total_quarantined: number;
}

// ---------------------------------------------------------------------------
// API Configuration
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function CISPage() {
  const [query, setQuery] = useState("");
  const [context, setContext] = useState("");
  const [goldAnswer, setGoldAnswer] = useState("");
  
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [baselineData, setBaselineData] = useState<BaselineResult | null>(null);
  const [stats, setStats] = useState<SystemStats | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"safe" | "quarantined">("quarantined");

  const runAnalysis = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBaselineData(null);

    try {
      const payload = { query: query.trim(), context: context.trim() };

      const analyzeReq = fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const baselineReq = fetch(`${API_BASE}/baseline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const [resAnalyze, resBaseline] = await Promise.all([analyzeReq, baselineReq]);

      if (!resAnalyze.ok) {
        throw new Error(`CIS Server error: ${resAnalyze.status}`);
      }
      
      const data: AnalysisResult = await resAnalyze.json();
      setResult(data);

      if (resBaseline.ok) {
         const baseData: BaselineResult = await resBaseline.json();
         setBaselineData(baseData);
      } else {
         setBaselineData({ answer: "Baseline computation failed.", latency_ms: 0 });
      }

      // Fetch updated stats
      const statsRes = await fetch(`${API_BASE}/stats`);
      if (statsRes.ok) {
        setStats(await statsRes.json());
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [query, context]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runAnalysis();
    }
  };

  return (
    <main className="min-h-screen">
      {/* Deep Luxury Header */}
      <header className="relative border-b border-zinc-900 overflow-hidden bg-black pb-8 pt-12">
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-sm bg-zinc-950 border border-zinc-800">
              <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-[26px] font-medium tracking-tight text-zinc-100">
                Cognitive Immune System
              </h1>
              <p className="text-xs text-zinc-500 uppercase tracking-widest mt-1 font-light">
                Inference-Time Epistemic Quarantine
              </p>
            </div>
          </div>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed font-light">
            A novel architectural primitive that intercepts and quarantines contaminated
            LLM claims <em className="text-zinc-200 not-italic font-normal">between reasoning steps</em>, preventing downstream
            contamination propagation. Based on the Containment Depth Bound Theorem:
            &nbsp;<span className="text-zinc-300 font-mono text-[11px] bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 tracking-wider">
              k* = ⌈log(ε) / log(1-ρ)⌉
            </span>
          </p>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <ResearchDashboard />

        {/* Input Section */}
        <div className="glass-card p-8 mb-10 relative">
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <label htmlFor="query-input" className="block text-[11px] font-medium text-zinc-400 uppercase tracking-widest mb-3">
                Query (Trigger)
              </label>
              <textarea
                id="query-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="What is Eric Ambler best known for?"
                className="w-full rounded bg-zinc-950 border border-zinc-800 px-4 py-3 text-sm text-zinc-100
                           placeholder-zinc-700 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-800 transition-all resize-none shadow-inner"
                rows={2}
              />
            </div>
            <div>
              <label htmlFor="context-input" className="block text-[11px] font-medium text-zinc-400 uppercase tracking-widest mb-3">
                Context (Adversarial Setup)
              </label>
              <textarea
                id="context-input"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Place contradictory or confusing facts here to test the immune response..."
                className="w-full rounded bg-zinc-950 border border-zinc-800 px-4 py-3 text-sm text-zinc-100
                           placeholder-zinc-700 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-800 transition-all resize-none shadow-inner"
                rows={2}
              />
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-zinc-900">
             <label htmlFor="gold-input" className="block text-[11px] font-medium text-zinc-500 uppercase tracking-widest mb-3 flex items-center gap-2">
               Ground Truth Validation (Real-Time Simulator)
             </label>
             <input
                id="gold-input"
                value={goldAnswer}
                onChange={(e) => setGoldAnswer(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 'English author of thrillers'"
                className="w-full md:w-1/2 rounded bg-zinc-950 border border-zinc-800 px-4 py-2 text-sm text-zinc-100 placeholder-zinc-700 focus:outline-none focus:border-zinc-500 transition-all shadow-inner"
             />
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading || !query.trim()}
            className="mt-8 w-full rounded bg-white px-6 py-4
                       text-xs font-semibold tracking-widest uppercase text-black transition-all hover:bg-zinc-200
                       disabled:opacity-20 disabled:cursor-not-allowed flex items-center justify-center gap-3"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Executing Live Parallel Pipelines...
              </>
            ) : (
              "Initialize Dual Pipeline Analysis"
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-300 animate-fade-in-up">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading Skeleton */}
        {loading && (
          <div className="grid md:grid-cols-2 gap-6 mb-8">
             <div className="glass-card p-4 border border-red-500/20"><div className="shimmer h-4 w-1/3 rounded mb-3" /><div className="shimmer h-24 w-full rounded" /></div>
             <div className="glass-card p-4 border border-emerald-500/20"><div className="shimmer h-4 w-1/3 rounded mb-3" /><div className="shimmer h-24 w-full rounded" /></div>
          </div>
        )}

        {/* Dual Results & Simulator */}
        {result && baselineData && !loading && (
          <div className="space-y-6 animate-fade-in-up">
             
            {/* Live Evaluator Dashboard */}
            {goldAnswer.trim() && (
               <ExperimentSimulator 
                  goldAnswer={goldAnswer}
                  baselineAnswer={baselineData.answer}
                  cisAnswer={result.answer}
                  baselineLatency={baselineData.latency_ms}
                  cisLatency={result.latency_ms}
                  isInjected={!!context.trim()} // If user injected malicious context assumption
                  contaminationRate={result.contamination_rate}
                  depthBound={result.containment_depth_bound}
               />
            )}

            {/* Split Screen System Outputs */}
            <div className="grid md:grid-cols-2 gap-8">
              
              {/* Baseline Side */}
              <div className="bg-black border border-zinc-800 rounded p-8 relative">
                 <div className="absolute top-0 inset-x-0 h-[1px] bg-red-600" />
                 <h2 className="text-[11px] font-semibold text-zinc-400 tracking-widest uppercase mb-6 flex items-center justify-between">
                    <span>Unsafe Baseline LLM</span>
                    <span className="text-zinc-600 font-mono text-[10px]">{baselineData.latency_ms}ms</span>
                 </h2>
                 <p className="text-[13px] leading-relaxed text-zinc-300 opacity-90 font-light">{baselineData.answer}</p>
                 <div className="mt-8 pt-6 border-t border-zinc-900">
                    <span className="text-[10px] text-zinc-500 font-semibold tracking-widest uppercase mb-2 block">Vulnerability Vectors</span>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">Has no epistemic immune system. Assumes all context is completely safe. Highly susceptible to adversarial semantic injection.</p>
                 </div>
              </div>

              {/* CIS Side */}
              <div className="bg-zinc-950 border border-zinc-800 rounded p-8 relative shadow-2xl">
                 <div className="absolute top-0 inset-x-0 h-[1px] bg-white" />
                 <h2 className="text-[11px] font-semibold text-zinc-100 tracking-widest uppercase mb-6 flex items-center justify-between">
                    <span>CIS Configured Output</span>
                    <span className="text-zinc-500 font-mono text-[10px]">{result.latency_ms}ms</span>
                 </h2>
                 <p className="text-[13px] leading-relaxed text-white font-light">{result.answer}</p>
                 <div className="mt-8 pt-6 border-t border-zinc-900">
                    <span className="text-[10px] text-zinc-500 font-semibold tracking-widest uppercase mb-2 block">Isolation Protocols Executed</span>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">System affirmatively quarantined {result.quarantined.length} contaminated atomic claims via strictly structured contradiction mapping logic.</p>
                 </div>
              </div>

            </div>

            {/* Metrics Dashboard */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <MetricCard label="Claims Extracted" value={result.claims_total} sub={`${result.claims_verifiable} verifiable`} />
              <MetricCard label="Contamination Rate" value={`${(result.contamination_rate * 100).toFixed(1)}%`} sub={result.contamination_rate >= 0.7 ? "HIGH" : result.contamination_rate >= 0.3 ? "MODERATE" : "LOW"} accent={result.contamination_rate >= 0.7 ? "danger" : result.contamination_rate >= 0.3 ? "warning" : "safe"} />
              <MetricCard label="Quarantined" value={result.quarantined.length} sub={`of ${result.claims_total} claims`} accent="danger" />
              <MetricCard label="Containment Depth" value={result.containment_depth_bound !== null ? `k*=${result.containment_depth_bound}` : "N/A"} sub="Terminal Layer Limit" />
            </div>

            {/* Claims Tab View */}
            <div className="glass-card overflow-hidden mt-10">
              {/* Tab Header */}
              <div className="flex border-b border-zinc-800">
                <button
                  onClick={() => setActiveTab("quarantined")}
                  className={`flex-1 px-4 py-4 text-[11px] font-semibold uppercase tracking-widest transition-colors
                    ${activeTab === "quarantined"
                      ? "text-red-500 border-b bg-red-950/10 border-red-500"
                      : "text-zinc-600 hover:text-zinc-400"
                    }`}
                >
                  Quarantined ({result.quarantined.length})
                </button>
                <button
                  onClick={() => setActiveTab("safe")}
                  className={`flex-1 px-4 py-4 text-[11px] font-semibold uppercase tracking-widest transition-colors
                    ${activeTab === "safe"
                      ? "text-white border-b bg-zinc-900/40 border-white"
                      : "text-zinc-600 hover:text-zinc-400"
                    }`}
                >
                  Verified Safe ({result.safe.length})
                </button>
              </div>

              {/* Claims List */}
              <div className="p-4 space-y-3 max-h-[600px] overflow-y-auto stagger-children">
                {activeTab === "quarantined" ? (
                  result.quarantined.length > 0 ? (
                    result.quarantined.map((claim, i) => (
                      <ClaimCard key={claim.id} claim={claim} status="quarantined" index={i} />
                    ))
                  ) : (
                    <EmptyState message="No contaminated claims detected — all claims passed verification." type="safe" />
                  )
                ) : (
                  result.safe.length > 0 ? (
                    result.safe.map((claim, i) => (
                      <ClaimCard key={claim.id} claim={claim} status="safe" index={i} />
                    ))
                  ) : (
                    <EmptyState message="No safe claims — all were quarantined." type="danger" />
                  )
                )}
              </div>
            </div>

            {/* Causal Trace */}
            <CausalTrace edges={result.causal_trace} />

          </div>
        )}

        {/* Footer */}
        <footer className="mt-16 border-t border-white/5 pt-6 pb-10">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <div>
              <span className="font-semibold text-gray-500">CIS Frontend Aligned</span><br />
              <span className="text-gray-700">Muhammad Saad · Independent Researcher</span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  sub,
  accent = "neutral",
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: "safe" | "danger" | "warning" | "neutral";
}) {
  const accentColors = {
    safe: "text-zinc-200",
    danger: "text-red-500",
    warning: "text-zinc-400",
    neutral: "text-zinc-300",
  };

  return (
    <div className="glass-card p-6 border border-zinc-800 bg-zinc-950/50 relative overflow-hidden">
      <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2 relative z-10">
        {label}
      </p>
      <p className={`text-2xl font-light tracking-tight ${accentColors[accent]} relative z-10`}>
        {value}
      </p>
      {sub && (
        <p className="text-[10px] text-zinc-600 mt-2 uppercase tracking-widest font-medium relative z-10">{sub}</p>
      )}
    </div>
  );
}

function EmptyState({
  message,
  type,
}: {
  message: string;
  type: "safe" | "danger";
}) {
  return (
    <div className="py-8 text-center">
      <div
        className={`mx-auto mb-2 h-10 w-10 rounded-full flex items-center justify-center
        ${type === "safe" ? "bg-emerald-500/10" : "bg-red-500/10"}`}
      >
        <span className="text-lg">{type === "safe" ? "✓" : "⚠"}</span>
      </div>
      <p className={`text-sm ${type === "safe" ? "text-emerald-400" : "text-red-400"}`}>
        {message}
      </p>
    </div>
  );
}

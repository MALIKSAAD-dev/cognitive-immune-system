"use client";

import React from "react";

// ---------------------------------------------------------------------------
// TypeScript Interfaces — matching exact backend response schema
// ---------------------------------------------------------------------------

export interface ClaimData {
  id: string;
  text: string;
  score: number;
  reason?: string;
  
  // v3 explicit metrics
  wiki_score?: number;     // S_wiki: NLI contradiction score
  cons_score?: number;     // S_cons: Semantic Entropy
  
  // legacy/meta metrics
  wiki_match?: boolean;
  confidence?: number;
  causal_flag?: boolean;
  timestamp?: string;
}

interface ClaimCardProps {
  claim: ClaimData;
  status: "safe" | "quarantined";
  index: number;
}

// ---------------------------------------------------------------------------
// ClaimCard Component
// ---------------------------------------------------------------------------

export default function ClaimCard({ claim, status, index }: ClaimCardProps) {
  const isSafe = status === "safe";
  // If the backend doesn't provide them, fall back to default neutral scores
  const scoreWiki = claim.wiki_score !== undefined ? claim.wiki_score : 0;
  const scoreCons = claim.cons_score !== undefined ? claim.cons_score : 0;
  const hasV3Metrics = claim.wiki_score !== undefined || claim.cons_score !== undefined;

  return (
    <div
      className={`
        relative overflow-hidden rounded border p-6 transition-all duration-300
        hover:-translate-y-0.5 hover:shadow-2xl
        ${
          isSafe
            ? "border-zinc-800 bg-zinc-950/80 hover:border-zinc-700"
            : "border-red-900/50 bg-red-950/5 hover:border-red-800"
        }
      `}
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      {/* 1. Header: Status & Final Score */}
      <div className="mb-6 flex items-start justify-between border-b border-zinc-900 pb-4">
        <div className="flex items-center gap-4">
          <div
            className={`flex h-8 w-8 items-center justify-center rounded-sm border ${
              isSafe ? "bg-white text-black border-white" : "bg-red-600 text-white border-red-600"
            }`}
          >
            <span className="text-[14px] font-medium leading-none mb-0.5">
              {isSafe ? "✓" : "!"}
            </span>
          </div>
          <div>
            <div className={`text-[11px] font-semibold uppercase tracking-widest ${isSafe ? "text-white" : "text-red-500"}`}>
              {status}
            </div>
            <div className="text-[10px] text-zinc-600 tracking-wider mt-0.5">{claim.id}</div>
          </div>
        </div>

        <div className="flex flex-col items-end">
          <div
            className={`
              px-2 text-lg font-light tracking-tight
              ${
                claim.score < 0.3
                  ? "text-zinc-300"
                  : claim.score < 0.55
                  ? "text-zinc-400"
                  : "text-red-500 font-medium"
              }
            `}
          >
            Φ(c) = {claim.score.toFixed(3)}
          </div>
          <div className="text-[9px] text-zinc-600 mt-1 uppercase tracking-widest font-medium pr-2">
            System Contamination Variable
          </div>
        </div>
      </div>

      {/* 2. The Extracted Claim Text */}
      <div className="mb-6">
        <div className="text-[10px] font-medium text-zinc-500 uppercase tracking-widest mb-2 flex items-center gap-1.5">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Extracted Atomic Claim
        </div>
        <p className="text-[15px] leading-relaxed text-zinc-200 font-light tracking-tight">"{claim.text}"</p>
      </div>

      {/* 3. V3 Pipeline Real-time Analysis (Ablation Visualization) */}
      {hasV3Metrics ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          
          {/* S_wiki (NLI) */}
          <div className="bg-black border border-zinc-900 rounded p-4 relative overflow-hidden">
            <div className="text-[9px] font-medium text-zinc-500 uppercase tracking-widest mb-3">
              S_wiki (NLI Contradiction)
            </div>
            <div className="flex justify-between items-end">
              <span className={`text-[19px] font-light tracking-tight ${scoreWiki > 0.5 ? "text-red-500" : "text-white"}`}>
                {scoreWiki.toFixed(3)}
              </span>
              <span className="text-[9px] text-zinc-600 uppercase font-mono px-1">
                WT: 0.60
              </span>
            </div>
          </div>

          {/* S_cons (Entropy) */}
          <div className="bg-black border border-zinc-900 rounded p-4 relative overflow-hidden">
            <div className="text-[9px] font-medium text-zinc-500 uppercase tracking-widest mb-3">
              S_cons (Semantic Entropy)
            </div>
            <div className="flex justify-between items-end">
              <span className={`text-[19px] font-light tracking-tight ${scoreCons > 0.5 ? "text-zinc-400" : "text-white"}`}>
                {scoreCons.toFixed(3)}
              </span>
              <span className="text-[9px] text-zinc-600 uppercase font-mono px-1">
                WT: 0.30
              </span>
            </div>
          </div>

          {/* S_causal (Graph) */}
          <div className="bg-black border border-zinc-900 rounded p-4 relative overflow-hidden">
            <div className="text-[9px] font-medium text-zinc-500 uppercase tracking-widest mb-3">
              S_causal (Causal Memory)
            </div>
            <div className="flex justify-between items-end">
              <span className={`text-[19px] font-light tracking-tight ${claim.causal_flag ? "text-red-500" : "text-white"}`}>
                {claim.causal_flag ? "1.000" : "0.000"}
              </span>
              <span className="text-[9px] text-zinc-600 uppercase font-mono px-1">
                WT: 0.10
              </span>
            </div>
          </div>

        </div>
      ) : (
        /* Legacy Meta Info */
        <div className="flex flex-wrap gap-2 mb-4">
          <MetaBadge label="Wikipedia" value={claim.wiki_match ? "Match" : "No Match"} positive={!!claim.wiki_match} />
          {claim.confidence !== undefined && <MetaBadge label="Confidence" value={`${claim.confidence.toFixed(1)}/10`} positive={claim.confidence >= 7} />}
          <MetaBadge label="Causal" value={claim.causal_flag ? "Linked" : "None"} positive={!claim.causal_flag} />
        </div>
      )}

      {/* 4. Reason Console Output */}
      {claim.reason && (
        <div className="bg-black rounded p-3 border border-zinc-800 text-[11px] text-zinc-400 mt-2 font-mono flex gap-3">
          <span className="text-zinc-600 uppercase tracking-widest text-[9px] leading-relaxed pt-0.5">Trace</span> 
          <span>{claim.reason}</span>
        </div>
      )}
    </div>
  );
}

function MetaBadge({ label, value, positive }: { label: string; value: string; positive: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-[9px] font-semibold uppercase tracking-widest ${positive ? "bg-zinc-900 text-white border border-zinc-800" : "bg-red-950/20 text-red-500 border border-red-900/50"}`}>
      <span className="text-zinc-500 font-medium">{label}:</span> {value}
    </span>
  );
}

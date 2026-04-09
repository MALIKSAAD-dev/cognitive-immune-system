import React from "react";
import { exactMatch } from "@/utils/evaluation";

interface SimulatorProps {
  goldAnswer: string;
  baselineAnswer: string;
  cisAnswer: string;
  baselineLatency: number;
  cisLatency: number;
  isInjected: boolean;
  contaminationRate: number;
  depthBound: number | null;
}

export default function ExperimentSimulator({
  goldAnswer,
  baselineAnswer,
  cisAnswer,
  baselineLatency,
  cisLatency,
  isInjected,
  contaminationRate,
  depthBound,
}: SimulatorProps) {
  // Simulator Logic
  const baselinePass = exactMatch(baselineAnswer, goldAnswer);
  const cisPass = exactMatch(cisAnswer, goldAnswer);

  let m1Result = "N/A";
  let m2Result = "N/A";
  let m3Result = "N/A";

  // M1 (Detection) applies if Baseline failed but CIS succeeded
  if (isInjected && !baselinePass && cisPass) m1Result = "Caught Injection! (+1 to M1)";
  else if (isInjected && !baselinePass && !cisPass) m1Result = "Failed to Detect (0 to M1)";

  // M2 (FPR) applies if Baseline succeeded but CIS failed
  if (!isInjected && baselinePass && !cisPass) m2Result = "False Positive! (+1 to M2)";
  else if (!isInjected) m2Result = "No False Positive (0 to M2)";

  // M3 (Accuracy)
  m3Result = cisPass ? "Accurate (+1 to M3A)" : "Inaccurate (0 to M3A)";

  function getP(k: number) {
     return Math.pow((1 - contaminationRate), k);
  }

  return (
    <div className="bg-black border border-zinc-800 rounded p-8 relative mb-12 shadow-2xl">
      {/* Background Math watermark */}
      <div className="absolute right-0 top-1/2 -translate-y-1/2 opacity-[0.02] pointer-events-none w-1/2">
        <svg viewBox="0 0 100 100" className="w-full h-full fill-current text-white">
           <text x="10" y="50" fontFamily="serif" fontSize="20" fontStyle="italic">P = (1-ρ)ᵏ</text>
        </svg>
      </div>

      <div className="flex items-center gap-4 mb-8">
        <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
        <h2 className="text-[17px] tracking-tight font-light text-white">Real-Time Scientific Evaluation</h2>
      </div>

      <div className="grid md:grid-cols-2 gap-10">
        {/* Left Col: M1 M2 M3 */}
        <div>
          <h3 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-6 border-b border-zinc-900 pb-3">
            Simulated Experiment
          </h3>
          <ul className="space-y-4">
            <li className="flex justify-between items-center bg-zinc-950 p-4 rounded border border-zinc-900">
              <span className="text-[12px] font-medium text-zinc-300 tracking-tight">M1 (Detection | CDR)</span>
              <span className={`text-[10px] uppercase tracking-widest font-semibold px-2.5 py-1 rounded ${m1Result.includes("+1") ? "bg-white text-black" : "bg-zinc-900 text-zinc-500"}`}>
                {m1Result}
              </span>
            </li>
            <li className="flex justify-between items-center bg-zinc-950 p-4 rounded border border-zinc-900">
              <span className="text-[12px] font-medium text-zinc-300 tracking-tight">M2 (False Positives | FPR)</span>
              <span className={`text-[10px] uppercase tracking-widest font-semibold px-2.5 py-1 rounded ${m2Result.includes("+1") ? "bg-red-950/20 text-red-500 border border-red-900/50" : "bg-zinc-900 text-zinc-500"}`}>
                {m2Result}
              </span>
            </li>
            <li className="flex justify-between items-center bg-zinc-950 p-4 rounded border border-zinc-900">
              <span className="text-[12px] font-medium text-zinc-300 tracking-tight">M3 (System Accuracy)</span>
              <span className={`text-[10px] uppercase tracking-widest font-semibold px-2.5 py-1 rounded ${m3Result.includes("+1") ? "bg-white text-black border border-white" : "bg-red-950/20 text-red-500 border border-red-900/50"}`}>
                {m3Result}
              </span>
            </li>
          </ul>
        </div>

        {/* Right Col: Theorem */}
        <div>
          <h3 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-6 border-b border-zinc-900 pb-3">
            Theorem Validation
          </h3>
          
          <div className="bg-zinc-950 p-6 rounded border border-zinc-800 font-mono text-[13px] leading-relaxed text-zinc-300">
            <div className="mb-2 text-[9px] text-white font-medium uppercase tracking-widest">Containment Depth Bound</div>
            <div className="mb-6 text-zinc-500 text-[11px] italic">Isolating containment threshold for $\rho$ = {contaminationRate.toFixed(3)}</div>
            
            {contaminationRate === 0 ? (
               <div className="text-white mt-4 text-[11px]">Since ρ = 0, no containment steps are needed (k* = 0).</div>
            ) : depthBound ? (
               <div className="space-y-6">
                  <div className="border border-zinc-800 bg-black p-4 rounded text-center">
                    <div className="font-light tracking-wide">P(ε &lt; 0.01) = (1 - {contaminationRate.toFixed(3)})^k</div>
                    <div className="mt-4 text-white text-[11px] tracking-widest uppercase">Optimal steps required (k*): {depthBound}</div>
                  </div>
                  <div className="pt-2">
                    <div className="h-6 flex w-full relative border-b border-zinc-800">
                       <div className="text-[9px] uppercase tracking-widest absolute bottom-2 left-0 text-zinc-600">Start</div>
                       <div className="text-[9px] uppercase tracking-widest absolute bottom-2 right-0 text-white font-semibold">Layer {depthBound} (ε&lt;0.01)</div>
                    </div>
                  </div>
               </div>
            ) : (
               <div className="text-red-500 mt-4 text-[11px]">Could not compute k*.</div>
            )}
            
            <div className="mt-6 pt-5 border-t border-zinc-900 text-[10px] text-zinc-500 flex justify-between uppercase tracking-widest">
              <span>Unsafe Latency: {baselineLatency}ms</span>
              <span className="text-white">CIS Route Latency: {cisLatency}ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

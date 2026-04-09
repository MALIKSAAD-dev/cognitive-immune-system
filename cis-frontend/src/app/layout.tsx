import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "CIS — Cognitive Immune System",
  description:
    "Inference-Time Epistemic Quarantine for LLM Hallucination Containment. " +
    "A research-grade AI system that intercepts and quarantines contaminated claims " +
    "between reasoning steps. Paper by Muhammad Saad.",
  keywords: [
    "LLM hallucination",
    "epistemic quarantine",
    "AI safety",
    "contamination containment",
    "causal reasoning",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-black text-white antialiased font-sans selection:bg-zinc-800 selection:text-white" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}

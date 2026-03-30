export const COLORS = {
  CADMM: '#00FF88',
  MAAN: '#FF2244',
  STATIC_GREEDY: '#888888',
  CONSTELLATION: '#00E5FF',
  BACKGROUND: '#0A0A0F',
  TEXT_MUTED: 'rgba(255, 255, 255, 0.5)',
  TEXT_DIM: 'rgba(255, 255, 255, 0.3)',
};

export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export const BEATS = [
  {
    id: 'orientation',
    badge: 'LIVE SIMULATION RUNNING',
    title: 'Three algorithms.\nOne network. Who wins?',
    subtitle: 'Three different mathematical strategies are competing right now to decide how to share 5G network bandwidth. Scroll to watch each one perform.',
    color: '#FFFFFF',
    accent: '#00FF88',
  },
  {
    id: 'cadmm',
    badge: 'ALGORITHM 01 / 03',
    title: 'C_ADMM',
    subtitle: 'Consensus Optimizer',
    description: 'Splits the network into slices and optimizes all of them simultaneously using mathematical consensus — like multiple people solving different parts of the same puzzle at once.',
    color: COLORS.CADMM,
    accent: COLORS.CADMM,
  },
  {
    id: 'maan',
    badge: 'ALGORITHM 02 / 03',
    title: 'MAAN',
    subtitle: 'Neural Predictor',
    description: "Watches historical traffic patterns and predicts what bandwidth will be needed before users ask for it — like a waiter who refills your glass before it's empty.",
    color: COLORS.MAAN,
    accent: COLORS.MAAN,
  },
  {
    id: 'static_greedy',
    badge: 'ALGORITHM 03 / 03 — BASELINE',
    title: 'STATIC_GREEDY',
    subtitle: 'The Benchmark',
    description: 'Uses a fixed set of rules that never change, no matter what the network conditions are. It is fast and predictable, but blind to what is actually happening. Every other algorithm must perform better than this.',
    color: COLORS.STATIC_GREEDY,
    accent: COLORS.STATIC_GREEDY,
  },
  {
    id: 'constellation',
    badge: 'FULL SYSTEM ACTIVE',
    title: 'The Network Decides.',
    subtitle: 'All three strategies are running simultaneously. The orchestration engine in the center receives their outputs and applies the best decision to the live network every 500 milliseconds.',
    color: COLORS.CONSTELLATION,
    accent: COLORS.CONSTELLATION,
  },
  {
    id: 'cta',
    title: 'Connect Your Network.',
    subtitle: 'Enter your API endpoint and watch the simulation replace with live 5G telemetry data.',
    color: COLORS.CONSTELLATION,
    accent: COLORS.CONSTELLATION,
  }
];

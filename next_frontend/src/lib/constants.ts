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
    subtitle: 'Price-Coordinated RL Agent',
    description: 'A reinforcement-learning agent (PPO) that decides allocations from the current network state. Each radio, compute and transport resource carries a "price" that rises as it gets oversubscribed, and every slice factors those prices into what it asks for — coordination by market signal rather than by forecasting.',
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
    subtitle: 'All three strategies animate side by side. These values are simulated in the browser — sine-wave telemetry redrawn every 500 milliseconds — not a live network and not the benchmark output. The real head-to-head comparison lives in the result figures.',
    color: COLORS.CONSTELLATION,
    accent: COLORS.CONSTELLATION,
  },
  {
    id: 'cta',
    title: 'See the Real Numbers.',
    subtitle: 'Everything above is illustrative animation. The actual study — 5 algorithms, 6 random seeds, 5 traffic load levels, paired t-tests with Holm-Bonferroni correction — runs in the Python backend. Open the figures it produced.',
    color: COLORS.CONSTELLATION,
    accent: COLORS.CONSTELLATION,
  }
];

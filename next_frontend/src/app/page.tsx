'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { useScroll, motion, AnimatePresence, useSpring } from 'framer-motion';
import dynamic from 'next/dynamic';
import { useNetworkData, SliceData } from '@/hooks/useNetworkData';
import { BACKEND_URL, COLORS, BEATS } from '@/lib/constants';
import { PlotGalleryOverlay } from '@/components/PlotGalleryOverlay';

const NetworkSliceCanvas = dynamic(() => import('@/components/NetworkSliceCanvas').then(m => m.NetworkSliceCanvas), { ssr: false });

// --- UI COMPONENTS ---
const Sparkline = ({ history, color }: { history: number[]; color: string }) => {
  const points = history.map((v, i) => `${(i / (history.length - 1)) * 120},${48 - (v / 100) * 48}`).join(' ');
  return (
    <svg width="120" height="48" viewBox="0 0 120 48" className="overflow-visible">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
};

interface BeatUIProps {
  beat: typeof BEATS[0];
  index: number;
  active: boolean;
  data: SliceData[];
  slices: number;
  setSlices: (val: number) => void;
  load: number;
  setLoad: (val: number) => void;
  onOpenPlots: () => void;
}

const BeatUI = ({ beat, index, active, data, slices, setSlices, load, setLoad, onOpenPlots }: BeatUIProps) => {
  const cAdmmData = data.find((d: SliceData) => d.algorithm === 'C_ADMM') || { utilValue: 0, history: [] };
  const maanData = data.find((d: SliceData) => d.algorithm === 'MAAN') || { utilValue: 0, history: [] };
  const greedyData = data.find((d: SliceData) => d.algorithm === 'Static_Greedy') || { utilValue: 0, history: [] };
  
  const avgUtil = (cAdmmData.utilValue + maanData.utilValue + greedyData.utilValue) / 3;

  const isLeftAligned = index === 1 || index === 3;
  const isRightAligned = index === 2;
  const isCenter = index === 0 || index === 4 || index === 5;

  return (
    <section className={`h-[100vh] w-full flex flex-col relative z-10 pointer-events-none ${
      isLeftAligned ? 'justify-center items-start pl-6 md:pl-[12vw]' :
      isRightAligned ? 'justify-center items-end pr-6 md:pr-[12vw]' :
      'items-center justify-end pb-[8vh]'
    }`}>
      <AnimatePresence mode="wait">
        {active && (
          <motion.div
            key={beat.id}
            initial={{ opacity: 0, y: isCenter ? 20 : 0, x: isLeftAligned ? -30 : isRightAligned ? 30 : 0 }}
            animate={{ opacity: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, y: isCenter ? -20 : 0, x: isLeftAligned ? -30 : isRightAligned ? 30 : 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className={`w-full pointer-events-auto ${isCenter ? 'max-w-[640px] text-center px-4' : 'max-w-[440px] text-left'}`}
          >
            {isCenter ? (
              // Center Beats (0, 4, 5)
              <>
                {beat.badge && (
                  <span className="inline-block text-[0.65rem] tracking-[0.2em] uppercase px-3.5 py-1 rounded-full border mb-6"
                    style={{ color: beat.accent, borderColor: `${beat.accent}33`, background: `${beat.accent}14` }}>
                    {beat.badge}
                  </span>
                )}
                <h1 className="text-white font-bold leading-[1.15] mb-3 whitespace-pre-line text-[clamp(2rem,3.5vw,3rem)]">
                  {beat.title}
                </h1>
                {beat.subtitle && (
                  <p className="text-white/50 italic mb-4 text-[1.1rem]">
                    {beat.subtitle}
                  </p>
                )}
                {beat.description && (
                  <p className="text-white/40 leading-[1.75] mb-8 mx-auto max-w-[480px]">
                    {beat.description}
                  </p>
                )}

                {index === 0 && (
                  <div className="flex flex-wrap justify-center gap-4">
                    {[
                      { l: 'C_ADMM', val: cAdmmData.utilValue, col: COLORS.CADMM },
                      { l: 'MAAN', val: maanData.utilValue, col: COLORS.MAAN },
                      { l: 'STATIC', val: greedyData.utilValue, col: COLORS.STATIC_GREEDY }
                    ].map(b => (
                      <div key={b.l} className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[0.8rem] text-white/70 flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ background: b.col }} /> {b.l} {b.val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                )}

                {index === 4 && (
                  <div className="glass-card p-6 md:p-8">
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      {[
                        { label: 'C_ADMM', val: cAdmmData.utilValue, color: COLORS.CADMM, history: cAdmmData.history },
                        { label: 'MAAN', val: maanData.utilValue, color: COLORS.MAAN, history: maanData.history },
                        { label: 'STATIC', val: greedyData.utilValue, color: COLORS.STATIC_GREEDY, history: greedyData.history },
                      ].map(col => (
                        <div key={col.label} className="text-center">
                          <div className="text-[0.6rem] md:text-[0.7rem] tracking-[0.15em] uppercase mb-1" style={{ color: col.color }}>{col.label}</div>
                          <div className="text-[1.5rem] md:text-[1.8rem] font-bold mb-1" style={{ color: col.color }}>{col.val.toFixed(1)}</div>
                          <div className="flex justify-center scale-75 origin-center">
                            <Sparkline history={col.history} color={col.color} />
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="h-[1px] bg-white/5 my-4" />
                    <div className="text-center">
                      <div className="text-[0.65rem] tracking-[0.18em] text-white/30 uppercase mb-1">AVERAGE SYSTEM UTILITY</div>
                      <div className="text-[2rem] font-bold text-[#00E5FF]">
                        {avgUtil.toFixed(1)}
                      </div>
                    </div>
                  </div>
                )}

                {index === 5 && (
                  <div className="flex flex-col items-center">
                    <button
                      onClick={onOpenPlots}
                      className="w-full max-w-[400px] bg-[#00E5FF] hover:bg-[#0A0A0F] text-[#0A0A0F] hover:text-[#00E5FF] border border-[#00E5FF] font-bold tracking-[0.2em] uppercase py-3.5 rounded-xl transition-all shadow-[0_0_15px_rgba(0,229,255,0.3)]"
                    >
                      VIEW BENCHMARK RESULTS →
                    </button>
                    <p className="text-[0.75rem] text-white/25 mt-3 max-w-[400px] text-center leading-relaxed">
                      Served from the backend API. Ingesting telemetry from an external 5G
                      network is not implemented.
                    </p>
                  </div>
                )}
              </>
            ) : (
              // Side Beats (1, 2, 3) - The new Unified Title Card
              <div className="glass-card p-6 md:p-8 flex flex-col gap-6 backdrop-blur-3xl shadow-2xl shadow-black/50 border border-white/10" style={{ boxShadow: `0 20px 40px -20px ${beat.accent}22` }}>
                
                {/* Title Header Section */}
                <div>
                  {beat.badge && (
                    <div className="mb-4">
                      <span className="inline-block text-[0.6rem] tracking-[0.2em] uppercase font-bold" style={{ color: beat.accent }}>
                        {beat.badge}
                      </span>
                    </div>
                  )}
                  <h2 className="text-[2.2rem] font-bold leading-none mb-2 text-white">{beat.title}</h2>
                  {beat.subtitle && (
                    <p className="text-white/50 italic mb-4 text-[0.95rem]">{beat.subtitle}</p>
                  )}
                  {beat.description && (
                    <p className="text-white/60 leading-[1.6] text-[0.85rem]">{beat.description}</p>
                  )}
                </div>

                <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                {/* Score & Controls Section */}
                {(index === 1 || index === 2) && (
                  <div>
                    <div className="flex justify-between items-end mb-4">
                      <div>
                        <div className="text-[0.55rem] tracking-[0.2em] text-white/30 uppercase mb-1">OPTIMIZATION SCORE</div>
                        <div className="text-[2.5rem] font-bold leading-none" style={{ color: beat.accent }}>
                          {(index === 1 ? cAdmmData.utilValue : maanData.utilValue).toFixed(1)}
                        </div>
                      </div>
                      <div className="scale-90 origin-bottom-right">
                        <Sparkline history={index === 1 ? cAdmmData.history : maanData.history} color={beat.accent} />
                        <div className="text-[0.55rem] tracking-[0.2em] text-white/30 uppercase mt-2 text-right">LAST 20 SECONDS</div>
                      </div>
                    </div>
                    
                    <div className="bg-white/5 rounded-xl p-4 mt-2">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-[0.75rem] text-white/70 font-medium">{index === 1 ? 'Live Network Slices' : 'Simulated Network Load'}</span>
                        <span className="text-[0.8rem] font-bold bg-white/10 px-2 py-0.5 rounded" style={{ color: beat.accent }}>{index === 1 ? slices : load.toFixed(1)}</span>
                      </div>
                      {/* Ranges mirror the backend's ResearchRunRequest bounds
                          (num_slices 3-6, load_center 0.6-2.0). Allowing values outside
                          them let "Run Full Research" fail validation with a 422. */}
                      <input
                        type="range"
                        min={index === 1 ? 3 : 0.6}
                        max={index === 1 ? 6 : 2.0}
                        step={index === 1 ? 1 : 0.1}
                        value={index === 1 ? slices : load}
                        onChange={(e) => index === 1 ? setSlices(parseInt(e.target.value)) : setLoad(parseFloat(e.target.value))}
                        style={{ color: beat.accent }}
                      />
                      <p className="text-[0.65rem] text-white/40 mt-3 leading-relaxed">
                        {index === 1 
                          ? "Higher slices = C_ADMM works harder to find consensus. Try sliding it to 10."
                          : "Higher load triggers MAAN's neural overdrive (red cage around the model)."}
                      </p>
                    </div>
                  </div>
                )}

                {index === 3 && (
                  <div>
                    <div className="text-[0.6rem] tracking-[0.2em] text-white/30 uppercase mb-3 text-center">CURRENT PERFORMANCE</div>
                    <div className="space-y-4 mb-2">
                      {[
                        { label: 'Static_Greedy (This model)', val: greedyData.utilValue, color: COLORS.STATIC_GREEDY, fill: '#555566' },
                        { label: 'C_ADMM', val: cAdmmData.utilValue, color: COLORS.CADMM, fill: COLORS.CADMM },
                        { label: 'MAAN', val: maanData.utilValue, color: COLORS.MAAN, fill: COLORS.MAAN },
                      ].map(bar => (
                        <div key={bar.label}>
                          <div className="flex justify-between text-[0.75rem] mb-1.5 font-medium">
                            <span style={{ color: bar.color }}>{bar.label}</span>
                            <span style={{ color: bar.color }}>{bar.val.toFixed(1)}</span>
                          </div>
                          <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                            <motion.div 
                              className="h-full rounded-full" 
                              style={{ background: bar.fill }}
                              animate={{ width: `${Math.min(100, Math.max(0, bar.val))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-[0.65rem] text-center text-white/30 italic mt-4">No adjustable parameters available.</p>
                  </div>
                )}

              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
};

type ResearchJob = {
  job_id: string;
  status: 'running' | 'completed' | 'failed' | string;
  progress: number;
  message: string;
  params?: {
    num_slices?: number;
    load_center?: number;
    seeds?: number;
    horizon?: number;
    n_mc_urlcc?: number;
  };
  error?: string | null;
};

// --- MAIN APP COMPONENT ---
export default function Home() {
  const { data, cAdmmSlices, setCAdmmSlices, maanLoad, setMaanLoad } = useNetworkData();
  const [plotGalleryOpen, setPlotGalleryOpen] = useState(false);
  const [plotsRefreshToken, setPlotsRefreshToken] = useState(0);
  const [researchJob, setResearchJob] = useState<ResearchJob | null>(null);
  const [researchStarting, setResearchStarting] = useState(false);
  const [researchError, setResearchError] = useState<string | null>(null);
  
  // We use the entire window scroll for this since the parent layout can just scroll normally
  const { scrollYProgress } = useScroll();
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 55, damping: 38 });
  
  const [currentBeat, setCurrentBeat] = useState(0);
  const [progressVal, setProgressVal] = useState(0);

  const startFullResearch = async () => {
    setResearchError(null);
    setResearchStarting(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/research/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Sized so the run finishes in a few minutes on a small shared-CPU instance.
        // The backend clamps these anyway; see ResearchRunRequest in main.py.
        body: JSON.stringify({
          num_slices: cAdmmSlices,
          load_center: maanLoad,
          seeds: 2,
          horizon: 120,
          n_mc_urlcc: 24,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        if (payload?.running_job) {
          setResearchJob(payload.running_job);
        }
        throw new Error(payload?.detail || 'Failed to start full research run');
      }
      setResearchJob(payload.job);
    } catch (err) {
      setResearchError(err instanceof Error ? err.message : 'Failed to start full research run');
    } finally {
      setResearchStarting(false);
    }
  };

  useEffect(() => {
    return scrollYProgress.on('change', (v) => {
      // Because we have 6 beats (0-5), each gets ~16.6% of scroll
      const beat = Math.min(5, Math.floor(v * 6 + 0.1));
      if (beat !== currentBeat) setCurrentBeat(beat);
    });
  }, [scrollYProgress, currentBeat]);

  useEffect(() => {
    return smoothProgress.on('change', setProgressVal);
  }, [smoothProgress]);

  useEffect(() => {
    let mounted = true;
    const loadLatestJob = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/research/status`);
        if (!res.ok) return;
        const payload = await res.json();
        if (mounted && payload?.job) {
          setResearchJob(payload.job);
        }
      } catch {
        // silent: research controls are optional
      }
    };
    loadLatestJob();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const id = researchJob?.job_id;
    if (!id) return;
    if (researchJob.status === 'completed' || researchJob.status === 'failed') return;

    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/research/status/${id}`);
        if (!res.ok) return;
        const payload = await res.json();
        const job = payload?.job;
        if (!job) return;
        setResearchJob(job);
        if (job.status === 'completed') {
          setPlotsRefreshToken(v => v + 1);
          setPlotGalleryOpen(true);
        }
      } catch {
        // keep polling on transient errors
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [researchJob?.job_id, researchJob?.status]);

  const scrollToBeat = (index: number) => {
    window.scrollTo({
      top: index * window.innerHeight,
      behavior: 'smooth'
    });
  };

  const currentAccent = useMemo(() => {
    if (currentBeat === 0) return '#FFFFFF';
    if (currentBeat === 1) return COLORS.CADMM;
    if (currentBeat === 2) return COLORS.MAAN;
    if (currentBeat === 3) return COLORS.STATIC_GREEDY;
    return COLORS.CONSTELLATION;
  }, [currentBeat]);

  const researchProgressPct = Math.round((researchJob?.progress || 0) * 100);

  return (
    <main className="bg-[#0A0A0F] text-white font-sans selection:bg-[#00E5FF]/30 min-h-screen">
      <div className="radial-glow" />
      
      {/* 3D CANVAS */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <NetworkSliceCanvas scrollProgress={progressVal} data={data} />
      </div>

      {/* NAV BAR */}
      <header className="fixed top-0 left-0 right-0 h-[60px] bg-[#0A0A0F]/80 backdrop-blur-2xl border-b border-white/5 flex items-center justify-between px-6 md:px-10 z-[100]">
        <div className="text-base font-bold tracking-[0.08em] text-white">AETHER_OS</div>
        <nav className="hidden lg:flex items-center gap-10">
          {['C_ADMM', 'MAAN', 'STATIC_GREEDY'].map((name, i) => (
            <button
              key={name}
              onClick={() => scrollToBeat(i + 1)}
              className="text-[0.72rem] tracking-[0.15em] font-bold transition-all border-b-[1.5px] py-1"
              style={{ 
                color: currentBeat === i + 1 ? '#FFFFFF' : 'rgba(255,255,255,0.35)',
                borderColor: currentBeat === i + 1 ? [COLORS.CADMM, COLORS.MAAN, COLORS.STATIC_GREEDY][i] : 'transparent'
              }}
            >
              {name}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <button
            onClick={startFullResearch}
            disabled={researchStarting || researchJob?.status === 'running'}
            className="text-[0.62rem] tracking-[0.16em] uppercase border border-[#00FF88]/35 text-[#00FF88] hover:text-white hover:border-[#00FF88]/70 px-2.5 py-1 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Runs a full benchmark sweep and regenerates plot artifacts"
          >
            {researchJob?.status === 'running' ? 'Research Running' : researchStarting ? 'Starting...' : 'Run Full Research'}
          </button>
          <button
            onClick={() => setPlotGalleryOpen(true)}
            className="text-[0.62rem] tracking-[0.16em] uppercase border border-[#00E5FF]/35 text-[#00E5FF] hover:text-white hover:border-[#00E5FF]/70 px-2.5 py-1 rounded-md transition-colors"
          >
            Result Plots
          </button>
          {researchJob && (
            <div className="hidden md:flex items-center gap-2 px-2 py-1 rounded-md border border-white/10 bg-white/[0.04]">
              <span
                className="text-[0.6rem] tracking-[0.14em] uppercase"
                style={{
                  color:
                    researchJob.status === 'completed'
                      ? '#00FF88'
                      : researchJob.status === 'failed'
                        ? '#FF6B6B'
                        : '#FFD166',
                }}
              >
                {researchJob.status}
              </span>
              <div className="w-16 h-1 rounded-full bg-white/15 overflow-hidden">
                <motion.div
                  className="h-full bg-[#00FF88]"
                  animate={{ width: `${researchProgressPct}%` }}
                  transition={{ duration: 0.4 }}
                />
              </div>
              <span className="text-[0.58rem] text-white/55">{researchProgressPct}%</span>
            </div>
          )}
          <motion.div 
            className="w-1.5 h-1.5 rounded-full bg-[#00FF88] shadow-[0_0_6px_#00FF88]"
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span className="text-[0.65rem] tracking-[0.15em] text-white/50 uppercase">
            {researchJob?.status === 'running' ? 'FULL RESEARCH MODE' : 'SIMULATION ACTIVE'}
          </span>
        </div>
      </header>

      {(researchError || researchJob) && (
        <div className="fixed top-[64px] left-1/2 -translate-x-1/2 z-[105] w-[min(92vw,760px)]">
          <div className="glass-card px-4 py-2.5 border-white/10 text-[0.74rem]">
            {researchError && <div className="text-[#FF8A8A]">{researchError}</div>}
            {researchJob && (
              <div className="text-white/75">
                <span className="font-semibold text-white/90">Full Research:</span>{' '}
                {researchJob.message || 'Running...'}
              </div>
            )}
          </div>
        </div>
      )}

      {/* SCROLL PROGRESS */}
      <aside className="fixed right-4 md:right-7 top-1/2 -translate-y-1/2 flex flex-col gap-[10px] z-[90]">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <button
            key={i}
            onClick={() => scrollToBeat(i)}
            className="rounded-full transition-all duration-300"
            style={{
              width: currentBeat === i ? 7 : 5,
              height: currentBeat === i ? 7 : 5,
              background: currentBeat === i ? currentAccent : 'rgba(255,255,255,0.15)',
              boxShadow: currentBeat === i ? `0 0 8px ${currentAccent}` : 'none'
            }}
          />
        ))}
      </aside>

      {/* SCROLL CONTAINER */}
      <div style={{ height: '600vh' }}>
        {BEATS.map((beat, i) => (
          <BeatUI 
            key={beat.id} 
            beat={beat} 
            index={i} 
            active={currentBeat === i}
            data={data}
            slices={cAdmmSlices}
            setSlices={setCAdmmSlices}
            load={maanLoad}
            setLoad={setMaanLoad}
            onOpenPlots={() => setPlotGalleryOpen(true)}
          />
        ))}
      </div>

      {/* FOOTER HINT */}
      <motion.div 
        className="fixed bottom-8 left-1/2 -translate-x-1/2 font-mono text-[0.72rem] tracking-[0.25em] text-white/25 pointer-events-none"
        animate={{ opacity: currentBeat === 0 ? 1 : 0 }}
      >
        scroll to explore ↓
      </motion.div>
      <PlotGalleryOverlay open={plotGalleryOpen} refreshToken={plotsRefreshToken} onClose={() => setPlotGalleryOpen(false)} />
    </main>
  );
}

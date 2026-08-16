'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { BACKEND_URL, COLORS } from '@/lib/constants';

type PlotItem = {
  name: string;
  title: string;
  url: string;
};

type DemoConfig = {
  seeds?: number;
  horizon?: number;
  load_scales?: number[];
};

type PlotManifest = {
  core: PlotItem[];
  publication: PlotItem[];
  legacy: PlotItem[];
  demo: PlotItem[];
  demo_config?: DemoConfig | null;
  counts: {
    core: number;
    publication: number;
    legacy: number;
    demo: number;
  };
};

type TabKey = 'core' | 'publication' | 'legacy' | 'demo';

const EMPTY_MANIFEST: PlotManifest = {
  core: [],
  publication: [],
  legacy: [],
  demo: [],
  demo_config: null,
  counts: { core: 0, publication: 0, legacy: 0, demo: 0 },
};

const TAB_META: Array<{ key: TabKey; label: string; accent: string }> = [
  { key: 'core', label: 'Core 14', accent: COLORS.CONSTELLATION },
  { key: 'publication', label: 'Publication Pack', accent: COLORS.CADMM },
  { key: 'legacy', label: 'Legacy', accent: COLORS.STATIC_GREEDY },
  { key: 'demo', label: 'Demo Run', accent: '#FFB020' },
];

interface PlotGalleryOverlayProps {
  open: boolean;
  onClose: () => void;
  refreshToken?: number;
  /** Which tab to select when the overlay opens. Set to 'demo' after a demo run finishes. */
  initialTab?: TabKey;
}

export function PlotGalleryOverlay({ open, onClose, refreshToken, initialTab }: PlotGalleryOverlayProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manifest, setManifest] = useState<PlotManifest>(EMPTY_MANIFEST);
  const [activeTab, setActiveTab] = useState<TabKey>('core');
  const [selected, setSelected] = useState<PlotItem | null>(null);

  const fetchPlots = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/plots`);
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = (await res.json()) as PlotManifest;
      setManifest(data);
      if (initialTab === 'demo' && (data.demo?.length ?? 0) > 0) {
        setActiveTab('demo');
      } else if (data.core.length === 0 && data.publication.length > 0) {
        setActiveTab('publication');
      } else if (data.core.length === 0 && data.publication.length === 0 && data.legacy.length > 0) {
        setActiveTab('legacy');
      } else {
        setActiveTab('core');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load plots');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    fetchPlots();
  }, [open, refreshToken]);

  // Escape closes the overlay. Without this the only way out is the Close button, and
  // while the overlay is up it covers the nav entirely.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // A zoomed figure is a layer above the gallery: back out of that first.
      // `selected` is read from the closure, not inside a state updater -- calling the
      // parent's onClose() from within setSelected() updates Home while this component
      // is rendering, which React warns about and which is not safe under concurrent
      // rendering. The extra `selected` dependency just re-binds the listener.
      if (selected) setSelected(null);
      else onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose, selected]);

  const currentPlots = useMemo(() => manifest[activeTab] || [], [manifest, activeTab]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[220] bg-[#05060A]/85 backdrop-blur-xl"
        >
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_45%_30%_at_50%_10%,rgba(0,229,255,0.12),transparent_70%)] pointer-events-none" />
          <div className="relative h-full w-full p-4 md:p-8">
            <div className="glass-card h-full flex flex-col overflow-hidden border-white/15">
              <header className="px-5 md:px-7 py-4 md:py-5 border-b border-white/10 flex items-center justify-between gap-4">
                <div>
                  <div className="text-[0.65rem] tracking-[0.2em] uppercase text-white/40">Benchmark Visuals</div>
                  <h2 className="text-[1.2rem] md:text-[1.4rem] font-bold text-white">
                    Publication Plot Gallery
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={fetchPlots}
                    className="px-3 py-1.5 rounded-lg border border-[#00E5FF]/35 hover:border-[#00E5FF]/60 text-[#00E5FF] hover:text-white transition-colors text-sm"
                  >
                    Refresh
                  </button>
                  <button
                    onClick={onClose}
                    className="px-3 py-1.5 rounded-lg border border-white/20 hover:border-[#00E5FF]/40 text-white/75 hover:text-white transition-colors text-sm"
                  >
                    Close
                  </button>
                </div>
              </header>

              <div className="px-5 md:px-7 py-4 border-b border-white/10 flex flex-wrap gap-2">
                {TAB_META.filter(t => t.key !== 'demo' || manifest.counts.demo > 0).map(tab => {
                  const count = manifest.counts[tab.key];
                  const active = activeTab === tab.key;
                  return (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className="px-3 py-1.5 rounded-full text-[0.72rem] tracking-[0.12em] uppercase border transition-all"
                      style={{
                        color: active ? tab.accent : 'rgba(255,255,255,0.6)',
                        borderColor: active ? `${tab.accent}88` : 'rgba(255,255,255,0.14)',
                        background: active ? `${tab.accent}14` : 'transparent',
                      }}
                    >
                      {tab.label} ({count})
                    </button>
                  );
                })}
              </div>

              <div className="flex-1 overflow-y-auto p-5 md:p-7">
                {activeTab === 'demo' && (
                  <div className="mb-5 rounded-xl border border-[#FFB020]/40 bg-[#FFB020]/[0.08] px-4 py-3">
                    <div className="text-[0.7rem] tracking-[0.18em] uppercase text-[#FFB020] mb-1">
                      Demo run — not the study
                    </div>
                    <div className="text-[0.85rem] text-white/75 leading-relaxed">
                      Generated by the dashboard button: a reduced sweep
                      {manifest.demo_config
                        ? ` (${manifest.demo_config.seeds} seeds x ${manifest.demo_config.load_scales?.length ?? '?'} loads x horizon ${manifest.demo_config.horizon})`
                        : ' (2 seeds x 3 loads x horizon 120)'}
                      , sized to finish in a few minutes. The published results — 6 seeds, 5 load
                      levels, horizon 500 — are under <span className="text-white/90">Core 14</span> and{' '}
                      <span className="text-white/90">Publication Pack</span>, and are never
                      overwritten by this run.
                    </div>
                  </div>
                )}
                {activeTab !== 'demo' && manifest.counts.demo > 0 && (
                  <div className="mb-5 text-[0.8rem] text-white/45">
                    Showing the committed 6-seed x 5-load study. Your demo run is under{' '}
                    <span className="text-[#FFB020]">Demo Run</span>.
                  </div>
                )}
                {loading && <div className="text-white/60 text-sm">Loading plot manifest...</div>}
                {error && <div className="text-[#FF6B6B] text-sm">Could not load plots: {error}</div>}
                {!loading && !error && currentPlots.length === 0 && (
                  <div className="text-white/60 text-sm">
                    No plots found in this category. Run phase-2 benchmark first:
                    <div className="font-mono mt-2 text-white/75">python -m src.experiments.run_benchmark_phase2</div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {currentPlots.map(item => (
                    <motion.button
                      key={item.url}
                      whileHover={{ y: -3 }}
                      onClick={() => setSelected(item)}
                      className="text-left rounded-2xl overflow-hidden border border-white/12 bg-white/[0.03] hover:border-[#00E5FF]/40 transition-colors"
                    >
                      <div className="aspect-[16/10] bg-black/40">
                        <img
                          src={`${BACKEND_URL}${item.url}`}
                          alt={item.title}
                          loading="lazy"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="px-3.5 py-3">
                        <div className="text-[0.78rem] uppercase tracking-[0.12em] text-white/45 mb-1">Figure</div>
                        <div className="text-[0.92rem] text-white/92 leading-snug">{item.title}</div>
                      </div>
                    </motion.button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <AnimatePresence>
            {selected && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[230] bg-black/90 p-4 md:p-8 flex items-center justify-center"
                onClick={() => setSelected(null)}
              >
                <div
                  className="w-full max-w-[1400px] bg-[#0C0E13] border border-white/15 rounded-2xl overflow-hidden"
                  onClick={e => e.stopPropagation()}
                >
                  <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
                    <div className="text-sm text-white/85">{selected.title}</div>
                    <div className="flex gap-2">
                      <a
                        href={`${BACKEND_URL}${selected.url}`}
                        target="_blank"
                        rel="noreferrer"
                        className="px-3 py-1.5 text-xs border border-white/25 rounded-md text-white/80 hover:text-white hover:border-[#00E5FF]/45 transition-colors"
                      >
                        Open Full
                      </a>
                      <button
                        onClick={() => setSelected(null)}
                        className="px-3 py-1.5 text-xs border border-white/25 rounded-md text-white/80 hover:text-white transition-colors"
                      >
                        Close
                      </button>
                    </div>
                  </div>
                  <div className="max-h-[82vh] overflow-auto">
                    <img
                      src={`${BACKEND_URL}${selected.url}`}
                      alt={selected.title}
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

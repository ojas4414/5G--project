/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { motion, AnimatePresence, useScroll, useSpring, useTransform, MotionValue } from 'motion/react';
import { Line, PerspectiveCamera } from '@react-three/drei';
import { createNoise3D } from 'simplex-noise';
import { COLORS, BEATS } from './constants';

// --- DATA SIMULATION HOOK ---
const useSimulationData = () => {
  const [data, setData] = useState({
    cadmm: 47.3,
    maan: 61.8,
    static: 29.1,
    cadmmHistory: Array(20).fill(47.3),
    maanHistory: Array(20).fill(61.8),
    staticHistory: Array(20).fill(29.1),
  });

  const [slices, setSlices] = useState(7);
  const [load, setLoad] = useState(1.2);

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const nextCadmm = Math.max(10, Math.min(95, prev.cadmm + (Math.random() - 0.5) * 10));
        const nextMaan = Math.max(10, Math.min(95, prev.maan + (Math.random() - 0.5) * 12));
        const nextStatic = Math.max(10, Math.min(95, prev.static + (Math.random() - 0.5) * 5));

        return {
          cadmm: nextCadmm,
          maan: nextMaan,
          static: nextStatic,
          cadmmHistory: [...prev.cadmmHistory.slice(1), nextCadmm],
          maanHistory: [...prev.maanHistory.slice(1), nextMaan],
          staticHistory: [...prev.staticHistory.slice(1), nextStatic],
        };
      });
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return { data, slices, setSlices, load, setLoad };
};

// --- 3D COMPONENTS ---

const CADMMModel = ({ progress, util }: { progress: MotionValue<number>; util: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state, delta) => {
    if (!meshRef.current) return;
    const p = progress.get();
    
    // Beat 0: Orbiting
    // Beat 1: Focused [0,0,0], scale 1.9
    // Beat 2-5: Ghost
    
    let targetScale = 0.7;
    let targetPos = new THREE.Vector3(0, 0, 0);
    let targetOpacity = 1.0;

    if (p < 0.16) {
      const angle = (state.clock.elapsedTime * 0.25);
      targetPos.set(Math.cos(angle) * 2.2, Math.sin(angle) * 2.2, 0);
    } else if (p >= 0.18 && p <= 0.34) {
      targetScale = 1.9;
      targetPos.set(0, 0, 0);
    } else {
      targetScale = 0.3;
      targetPos.set(-4, 1.5, -3);
      targetOpacity = 0.08;
    }

    // Beat 4 Constellation
    if (p >= 0.72 && p <= 0.88) {
      const angle = (state.clock.elapsedTime * 0.2) + 0;
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 0.85;
      targetOpacity = 1.0;
    }
    
    // Beat 5 CTA
    if (p >= 0.9) {
      const angle = (state.clock.elapsedTime * 0.35) + 0;
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 1.0;
      targetOpacity = 1.0;
    }

    meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    meshRef.current.position.lerp(targetPos, 0.1);
    (meshRef.current.material as THREE.MeshPhysicalMaterial).opacity = THREE.MathUtils.lerp((meshRef.current.material as THREE.MeshPhysicalMaterial).opacity, targetOpacity, 0.1);
    
    meshRef.current.rotation.x += delta * 0.12;
    meshRef.current.rotation.y += delta * (0.18 + util / 220);
    meshRef.current.rotation.z += delta * 0.06;
  });

  return (
    <mesh ref={meshRef}>
      <octahedronGeometry args={[1, 0]} />
      <meshPhysicalMaterial
        color="#0a1a10"
        emissive={COLORS.CADMM}
        emissiveIntensity={util / 38}
        roughness={0.04}
        metalness={0.05}
        transmission={0.82}
        thickness={2.0}
        ior={1.9}
        envMapIntensity={1.2}
        transparent
      />
    </mesh>
  );
};

const MAANModel = ({ progress, util }: { progress: MotionValue<number>; util: number }) => {
  const solidRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (!groupRef.current || !solidRef.current || !wireRef.current) return;
    const p = progress.get();
    
    let targetScale = 0.7;
    let targetPos = new THREE.Vector3(0, 0, 0);
    let targetOpacity = 1.0;

    if (p < 0.16) {
      const angle = (state.clock.elapsedTime * 0.25) + (Math.PI * 2 / 3);
      targetPos.set(Math.cos(angle) * 2.2, Math.sin(angle) * 2.2, 0);
    } else if (p >= 0.36 && p <= 0.52) {
      targetScale = 1.85;
      targetPos.set(0, 0, 0);
    } else {
      targetScale = 0.3;
      targetPos.set(4, -1, -3);
      targetOpacity = 0.08;
    }

    // Beat 4 Constellation
    if (p >= 0.72 && p <= 0.88) {
      const angle = (state.clock.elapsedTime * 0.2) + (Math.PI * 2 / 3);
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 0.85;
      targetOpacity = 1.0;
    }
    
    // Beat 5 CTA
    if (p >= 0.9) {
      const angle = (state.clock.elapsedTime * 0.35) + (Math.PI * 2 / 3);
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 1.0;
      targetOpacity = 1.0;
    }

    groupRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    groupRef.current.position.lerp(targetPos, 0.1);
    
    const mat = solidRef.current.material as THREE.MeshStandardMaterial;
    mat.opacity = THREE.MathUtils.lerp(mat.opacity, targetOpacity, 0.1);
    mat.emissiveIntensity = util / 38;
    
    const wireMat = wireRef.current.material as THREE.MeshBasicMaterial;
    wireMat.opacity = (0.25 + (util / 160)) * targetOpacity;

    solidRef.current.rotation.x += delta * (0.08 + util / 280);
    solidRef.current.rotation.y += delta * 0.22;
    solidRef.current.rotation.z += delta * 0.05;
    
    wireRef.current.rotation.copy(solidRef.current.rotation);
  });

  return (
    <group ref={groupRef}>
      <mesh ref={solidRef}>
        <icosahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          color="#1a0008"
          emissive={COLORS.MAAN}
          roughness={0.35}
          metalness={0.55}
          transparent
        />
      </mesh>
      <mesh ref={wireRef}>
        <icosahedronGeometry args={[1.02, 0]} />
        <meshBasicMaterial
          color="#FF4466"
          wireframe
          transparent
        />
      </mesh>
    </group>
  );
};

const StaticGreedyModel = ({ progress, util }: { progress: MotionValue<number>; util: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const noise3D = useMemo(() => createNoise3D(), []);
  
  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(1, 28, 28);
    const pos = geo.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      const noise = noise3D(v.x * 1.8, v.y * 1.8, v.z * 1.8);
      v.multiplyScalar(1 + noise * 0.32);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    return geo;
  }, [noise3D]);

  useFrame((state, delta) => {
    if (!meshRef.current) return;
    const p = progress.get();
    
    let targetScale = 0.7;
    let targetPos = new THREE.Vector3(0, 0, 0);
    let targetOpacity = 1.0;

    if (p < 0.16) {
      const angle = (state.clock.elapsedTime * 0.25) + (Math.PI * 4 / 3);
      targetPos.set(Math.cos(angle) * 2.2, Math.sin(angle) * 2.2, 0);
    } else if (p >= 0.54 && p <= 0.70) {
      targetScale = 1.75;
      targetPos.set(0, 0, 0);
    } else {
      targetScale = 0.3;
      targetPos.set(4, -1, -3);
      targetOpacity = 0.08;
    }

    // Beat 4 Constellation
    if (p >= 0.72 && p <= 0.88) {
      const angle = (state.clock.elapsedTime * 0.2) + (Math.PI * 4 / 3);
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 0.85;
      targetOpacity = 1.0;
    }
    
    // Beat 5 CTA
    if (p >= 0.9) {
      const angle = (state.clock.elapsedTime * 0.35) + (Math.PI * 4 / 3);
      targetPos.set(Math.cos(angle) * 1.8, Math.sin(angle) * 1.8, 0);
      targetScale = 1.0;
      targetOpacity = 1.0;
    }

    meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    meshRef.current.position.lerp(targetPos, 0.1);
    
    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    mat.opacity = THREE.MathUtils.lerp(mat.opacity, targetOpacity, 0.1);
    mat.emissiveIntensity = util / 55;

    meshRef.current.rotation.x += delta * 0.06;
    meshRef.current.rotation.y += delta * 0.09;
  });

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color="#1a1a20"
        emissive="#555566"
        roughness={0.88}
        metalness={0.92}
        transparent
      />
    </mesh>
  );
};

const ConstellationCore = ({ progress, util }: { progress: MotionValue<number>; util: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state, delta) => {
    if (!meshRef.current) return;
    const p = progress.get();
    
    let targetScale = 0;
    let targetPos = new THREE.Vector3(0, -5, 0);
    
    if (p >= 0.72) {
      targetScale = 1.2;
      targetPos.set(0, 0, 0);
    }
    
    meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.05);
    meshRef.current.position.lerp(targetPos, 0.05);
    
    meshRef.current.rotation.x += delta * 0.18;
    meshRef.current.rotation.y += delta * 0.28;
    
    (meshRef.current.material as THREE.MeshPhysicalMaterial).emissiveIntensity = util / 32;
  });

  return (
    <mesh ref={meshRef}>
      <torusKnotGeometry args={[1, 0.3, 128, 16]} />
      <meshPhysicalMaterial
        color="#001515"
        emissive={COLORS.CONSTELLATION}
        roughness={0.0}
        metalness={0.0}
        transmission={0.65}
        thickness={1.8}
        ior={1.7}
        transparent
      />
    </mesh>
  );
};

const Scene = ({ progress, data }: { progress: MotionValue<number>; data: any }) => {
  const accentLightRef = useRef<THREE.PointLight>(null);
  
  useFrame((state) => {
    if (!accentLightRef.current) return;
    const p = progress.get();
    let targetColor = new THREE.Color('#FFFFFF');
    if (p < 0.16) targetColor.set('#FFFFFF');
    else if (p < 0.34) targetColor.set(COLORS.CADMM);
    else if (p < 0.52) targetColor.set(COLORS.MAAN);
    else if (p < 0.70) targetColor.set(COLORS.STATIC_GREEDY);
    else targetColor.set(COLORS.CONSTELLATION);
    
    accentLightRef.current.color.lerp(targetColor, 0.05);
    
    // Camera breathing
    state.camera.position.z = 8 + Math.sin(state.clock.elapsedTime * 0.4) * 0.15;
    if (p >= 0.18 && p <= 0.70) {
      state.camera.position.z = 5.5;
      state.camera.position.x = Math.sin(state.clock.elapsedTime * 0.08) * 0.9;
      state.camera.position.y = Math.cos(state.clock.elapsedTime * 0.06) * 0.35;
    } else {
      state.camera.position.x = THREE.MathUtils.lerp(state.camera.position.x, 0, 0.05);
      state.camera.position.y = THREE.MathUtils.lerp(state.camera.position.y, 0, 0.05);
    }
    state.camera.lookAt(0, 0, 0);
  });

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 8]} fov={52} />
      <ambientLight color="#ffffff" intensity={0.25} />
      <directionalLight color="#ffffff" intensity={1.8} position={[6, 10, 6]} castShadow />
      <directionalLight color="#8888ff" intensity={0.4} position={[-4, -2, -4]} />
      <pointLight ref={accentLightRef} intensity={3.0} position={[0, 2, 3]} distance={8} decay={2} />
      
      <CADMMModel progress={progress} util={data.cadmm} />
      <MAANModel progress={progress} util={data.maan} />
      <StaticGreedyModel progress={progress} util={data.static} />
      <ConstellationCore progress={progress} util={(data.cadmm + data.maan + data.static) / 3} />
    </>
  );
};

// --- UI COMPONENTS ---

const Sparkline = ({ history, color }: { history: number[]; color: string }) => {
  const points = history.map((v, i) => `${(i / 19) * 120},${48 - (v / 100) * 48}`).join(' ');
  return (
    <svg width="120" height="48" viewBox="0 0 120 48">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
};

const BeatUI = ({ beat, index, active, data, slices, setSlices, load, setLoad }: any) => {
  return (
    <section className="beat-section">
      <AnimatePresence mode="wait">
        {active && (
          <motion.div
            key={beat.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="w-full max-w-[640px] text-center"
          >
            {beat.badge && (
              <span className="inline-block text-[0.65rem] tracking-[0.2em] uppercase px-3.5 py-1 rounded-full border mb-6"
                style={{ color: beat.accent, borderColor: `${beat.accent}33`, background: `${beat.accent}14` }}>
                {beat.badge}
              </span>
            )}
            
            <h1 className="text-white font-bold leading-[1.15] mb-3 whitespace-pre-line"
                style={{ fontSize: 'clamp(2rem, 3.5vw, 3rem)' }}>
              {beat.title}
            </h1>
            
            {beat.subtitle && (
              <p className="text-neutral-500 italic mb-4" style={{ fontSize: '1.1rem' }}>
                {beat.subtitle}
              </p>
            )}
            
            {beat.description && (
              <p className="text-neutral-400 leading-[1.75] mb-8 mx-auto max-w-[480px]">
                {beat.description}
              </p>
            )}

            {/* Data Cards based on beat */}
            {index === 0 && (
              <div className="flex justify-center gap-4">
                <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[0.8rem] text-white/70 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: COLORS.CADMM }} /> C_ADMM {data.cadmm.toFixed(1)}
                </div>
                <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[0.8rem] text-white/70 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: COLORS.MAAN }} /> MAAN {data.maan.toFixed(1)}
                </div>
                <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-[0.8rem] text-white/70 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: COLORS.STATIC_GREEDY }} /> STATIC_GREEDY {data.static.toFixed(1)}
                </div>
              </div>
            )}

            {(index === 1 || index === 2) && (
              <div className="glass-card p-6 md:p-8 text-left">
                <div className="grid grid-cols-2 gap-8 mb-4">
                  <div>
                    <div className="text-[2.8rem] font-bold leading-none mb-1" style={{ color: beat.accent }}>
                      {(index === 1 ? data.cadmm : data.maan).toFixed(1)}
                    </div>
                    <div className="text-[0.6rem] tracking-[0.18em] text-white/30 uppercase">OPTIMIZATION SCORE</div>
                  </div>
                  <div>
                    <Sparkline history={index === 1 ? data.cadmmHistory : data.maanHistory} color={beat.accent} />
                    <div className="text-[0.6rem] tracking-[0.18em] text-white/30 uppercase mt-2">LAST 20 READINGS</div>
                  </div>
                </div>
                <div className="h-[1px] bg-white/5 my-4" />
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[0.8rem] text-white/60">{index === 1 ? 'Active Network Slices' : 'Max Network Load'}</span>
                    <span className="text-[0.8rem] font-bold" style={{ color: beat.accent }}>{index === 1 ? slices : load.toFixed(1)}</span>
                  </div>
                  <input 
                    type="range" 
                    min={index === 1 ? 1 : 0.1} 
                    max={index === 1 ? 10 : 2.0} 
                    step={index === 1 ? 1 : 0.1}
                    value={index === 1 ? slices : load}
                    onChange={(e) => index === 1 ? setSlices(parseInt(e.target.value)) : setLoad(parseFloat(e.target.value))}
                    style={{ color: beat.accent }}
                  />
                  <div className="text-[0.72rem] text-white/25 mt-1">
                    {index === 1 ? 'More slices = higher optimization complexity' : 'Higher max load = MAAN oscillates faster'}
                  </div>
                </div>
                <div className="p-3 rounded-r-lg border-l-2 bg-white/5" style={{ borderLeftColor: beat.accent }}>
                  <p className="text-[0.8rem] text-white/45 leading-relaxed">
                    {index === 1 
                      ? "The score above shows how efficiently C_ADMM is distributing bandwidth right now. Higher = better. Move the slider to simulate more or fewer users competing for the same network."
                      : "Network load is how many requests per second the system must handle. Push it up and watch MAAN work harder — the red cage around the model shows its processing intensity."}
                  </p>
                </div>
              </div>
            )}

            {index === 3 && (
              <div className="glass-card p-6 md:p-8 text-left">
                <div className="text-[0.65rem] tracking-[0.18em] text-white/30 uppercase mb-4">HOW DOES IT COMPARE?</div>
                <div className="space-y-4 mb-6">
                  {[
                    { label: 'Static_Greedy (this one)', val: data.static, color: COLORS.STATIC_GREEDY, fill: '#555566' },
                    { label: 'C_ADMM', val: data.cadmm, color: COLORS.CADMM, fill: COLORS.CADMM },
                    { label: 'MAAN', val: data.maan, color: COLORS.MAAN, fill: COLORS.MAAN },
                  ].map(bar => (
                    <div key={bar.label}>
                      <div className="flex justify-between text-[0.8rem] mb-1">
                        <span style={{ color: bar.color }}>{bar.label}</span>
                        <span style={{ color: bar.color }}>{bar.val.toFixed(1)}</span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full rounded-full" 
                          style={{ background: bar.fill }}
                          animate={{ width: `${bar.val}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-3 rounded-r-lg border-l-2 bg-white/5 border-neutral-600">
                  <p className="text-[0.8rem] text-white/45 leading-relaxed">
                    The bar chart above shows all three algorithms performing right now. Static_Greedy almost always scores lowest because it cannot adapt. The higher the other bars climb above it, the more value the smarter algorithms are adding.
                  </p>
                </div>
                <p className="text-[0.75rem] text-white/20 italic text-center mt-3">
                  No adjustable parameters — this algorithm uses fixed rules that never change by design.
                </p>
              </div>
            )}

            {index === 4 && (
              <div className="glass-card p-6 md:p-8">
                <div className="grid grid-cols-3 gap-4 mb-6">
                  {[
                    { label: 'C_ADMM', val: data.cadmm, color: COLORS.CADMM, history: data.cadmmHistory },
                    { label: 'MAAN', val: data.maan, color: COLORS.MAAN, history: data.maanHistory },
                    { label: 'STATIC_GREEDY', val: data.static, color: COLORS.STATIC_GREEDY, history: data.staticHistory },
                  ].map(col => (
                    <div key={col.label} className="text-center">
                      <div className="text-[0.7rem] tracking-[0.15em] uppercase mb-1" style={{ color: col.color }}>{col.label}</div>
                      <div className="text-[1.8rem] font-bold mb-1" style={{ color: col.color }}>{col.val.toFixed(1)}</div>
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
                    {((data.cadmm + data.maan + data.static) / 3).toFixed(1)}
                  </div>
                </div>
              </div>
            )}

            {index === 5 && (
              <div className="flex flex-col items-center">
                <input 
                  type="text" 
                  placeholder="wss://your-api-endpoint.com/telemetry"
                  className="w-full max-w-[400px] bg-white/5 border border-white/10 rounded-xl px-5 py-3.5 text-white font-mono text-[0.85rem] mb-4 focus:outline-none focus:border-[#00E5FF]/50 transition-colors"
                />
                <button className="w-full max-w-[400px] bg-[#00E5FF] hover:bg-[#00CCEE] text-[#0A0A0F] font-bold tracking-[0.2em] uppercase py-3.5 rounded-xl transition-all transform hover:-translate-y-0.5 active:scale-[0.98]">
                  CONNECT API →
                </button>
                <p className="text-[0.75rem] text-white/20 mt-3">Or continue with simulation mode</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
};

// --- MAIN APP ---

export default function App() {
  const { data, slices, setSlices, load, setLoad } = useSimulationData();
  const scrollerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: scrollerRef });
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 55, damping: 38 });
  const [currentBeat, setCurrentBeat] = useState(0);

  useEffect(() => {
    return scrollYProgress.on('change', (v) => {
      const beat = Math.min(5, Math.floor(v * 6 + 0.1));
      if (beat !== currentBeat) setCurrentBeat(beat);
    });
  }, [scrollYProgress, currentBeat]);

  const scrollToBeat = (index: number) => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTo({
        top: index * window.innerHeight,
        behavior: 'smooth'
      });
    }
  };

  const currentAccent = useMemo(() => {
    if (currentBeat === 0) return '#FFFFFF';
    if (currentBeat === 1) return COLORS.CADMM;
    if (currentBeat === 2) return COLORS.MAAN;
    if (currentBeat === 3) return COLORS.STATIC_GREEDY;
    return COLORS.CONSTELLATION;
  }, [currentBeat]);

  return (
    <div className="bg-surface text-white font-body selection:bg-[#00E5FF]/30">
      <div className="radial-glow" />
      
      {/* 3D CANVAS */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <Canvas dpr={[1, 2]}>
          <Scene progress={smoothProgress} data={data} />
        </Canvas>
      </div>

      {/* NAV BAR */}
      <header className="fixed top-0 left-0 right-0 h-[60px] bg-[#0A0A0F]/80 backdrop-blur-2xl border-b border-white/5 flex items-center justify-between px-10 z-[100]">
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
          <motion.div 
            className="w-1.5 h-1.5 rounded-full bg-[#00FF88] shadow-[0_0_6px_#00FF88]"
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span className="text-[0.65rem] tracking-[0.15em] text-white/50 uppercase">SIMULATION ACTIVE</span>
        </div>
      </header>

      {/* SCROLL PROGRESS */}
      <aside className="fixed right-7 top-1/2 -translate-y-1/2 flex flex-col gap-[10px] z-[90]">
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
      <div className="snap-container" ref={scrollerRef}>
        <div style={{ height: '600vh' }}>
          {BEATS.map((beat, i) => (
            <BeatUI 
              key={beat.id} 
              beat={beat} 
              index={i} 
              active={currentBeat === i}
              data={data}
              slices={slices}
              setSlices={setSlices}
              load={load}
              setLoad={setLoad}
            />
          ))}
        </div>
      </div>

      {/* FOOTER HINT */}
      <motion.div 
        className="fixed bottom-8 left-1/2 -translate-x-1/2 font-mono text-[0.72rem] tracking-[0.25em] text-white/25 pointer-events-none"
        animate={{ opacity: currentBeat === 0 ? 1 : 0 }}
      >
        scroll to explore ↓
      </motion.div>
    </div>
  );
}

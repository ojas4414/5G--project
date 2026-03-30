'use client';

import React, { useRef, useMemo, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, ContactShadows, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { createNoise3D } from 'simplex-noise';
import { SliceData } from '@/hooks/useNetworkData';
import { CustomNebula } from './CustomNebula';
import { Astronaut } from './Astronaut';

interface CanvasProps {
  scrollProgress: number; // 0.0 to 1.0
  data: SliceData[];
}

const noise3D = createNoise3D();

// Helper: frame-rate independent damped motion
const damp = THREE.MathUtils.damp;

// --- COMPONENTS ---

// 1. C_ADMM (Octahedron - The Gem)
const CadmmModel = ({ util, scrollProgress }: { util: number, scrollProgress: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshPhysicalMaterial>(null);

  useFrame((state, delta) => {
    if (!meshRef.current || !materialRef.current) return;

    let targetX = 0, targetY = 0, targetZ = 0, targetScale = 1.0, targetOpacity = 1.0;

    if (scrollProgress <= 0.16) {
      const angle = state.clock.elapsedTime * 0.2;
      targetX = Math.cos(angle) * 2.0;
      targetZ = Math.sin(angle) * 2.0 - 2; 
    } else if (scrollProgress > 0.16 && scrollProgress < 0.33) {
      // Focused - slight breathing based on utilization
      targetX = 1.8; // Shifted right for asymmetric UI
      targetY = Math.sin(state.clock.elapsedTime * 1.5) * 0.05;
      targetZ = 0;
      targetScale = 1.6 + (util / 100) * 0.15; // physical bulge when load is high
      targetOpacity = 1.0;
    } else if (scrollProgress >= 0.33) {
      if (scrollProgress > 0.66) {
        targetX = -3; targetY = 2 + Math.sin(state.clock.elapsedTime * 1) * 0.1; targetZ = -1;
        targetScale = 1.0 + (util / 100) * 0.1;
        targetOpacity = 1.0;
      } else {
        targetX = -3.5; targetY = 1; targetZ = -2;
        targetScale = 0.4;
        targetOpacity = 0.08;
      }
    }

    meshRef.current.position.x = damp(meshRef.current.position.x, targetX, 4, delta);
    meshRef.current.position.y = damp(meshRef.current.position.y, targetY, 4, delta);
    meshRef.current.position.z = damp(meshRef.current.position.z, targetZ, 4, delta);
    
    const s = damp(meshRef.current.scale.x, targetScale, 4, delta);
    meshRef.current.scale.set(s, s, s);
    
    materialRef.current.opacity = damp(materialRef.current.opacity, targetOpacity, 4, delta);
    materialRef.current.transparent = materialRef.current.opacity < 0.99;

    // Glowing core effect that intensifies
    materialRef.current.emissiveIntensity = damp(materialRef.current.emissiveIntensity, (util / 40) + ((Math.sin(state.clock.elapsedTime * 4) + 1) * 0.2), 6, delta);

    // Momentum-based rotation
    const baseRotSpeed = scrollProgress > 0.16 && scrollProgress < 0.33 ? 0.3 : 0.1;
    meshRef.current.rotation.x += delta * (baseRotSpeed * 0.5);
    meshRef.current.rotation.y += delta * (baseRotSpeed + util / 150);
  });

  return (
    <mesh ref={meshRef}>
      <octahedronGeometry args={[1, 0]} />
      <meshPhysicalMaterial 
        ref={materialRef}
        color="#08101a" 
        emissive="#00FF88" 
        roughness={0.08} 
        metalness={0.2} 
        transmission={0.95} 
        thickness={2.0} 
        ior={1.6}
        clearcoat={1.0}
        clearcoatRoughness={0.1}
        transparent
      />
    </mesh>
  );
};

// 2. MAAN (Icosahedron - The Neural Engine)
const MaanModel = ({ util, scrollProgress }: { util: number, scrollProgress: number }) => {
  const groupRef = useRef<THREE.Group>(null);
  const solidMatRef = useRef<THREE.MeshPhysicalMaterial>(null);
  const wireMatRef = useRef<THREE.MeshBasicMaterial>(null);

  useFrame((state, delta) => {
    if (!groupRef.current || !solidMatRef.current || !wireMatRef.current) return;

    let targetX = 0, targetY = 0, targetZ = 0, targetScale = 1.0, targetOpacity = 1.0;

    if (scrollProgress <= 0.16) {
      const angle = state.clock.elapsedTime * 0.2 + (Math.PI * 2 / 3);
      targetX = Math.cos(angle) * 2.0;
      targetZ = Math.sin(angle) * 2.0 - 2;
    } else if (scrollProgress > 0.16 && scrollProgress < 0.33) {
      targetX = 3.5; targetY = -1; targetZ = -2;
      targetScale = 0.4;
      targetOpacity = 0.08;
    } else if (scrollProgress >= 0.33 && scrollProgress < 0.50) {
      // Focused - nervous, alive movement
      targetX = -1.8 + (Math.sin(state.clock.elapsedTime * 3) * 0.02 * (util/50)); // Shifted left
      targetY = Math.cos(state.clock.elapsedTime * 2.5) * 0.02 * (util/50);
      targetZ = 0;
      targetScale = 1.6 + (util / 100) * 0.1;
      targetOpacity = 1.0;
    } else {
      if (scrollProgress > 0.66) {
        targetX = 3; targetY = 2 + Math.cos(state.clock.elapsedTime * 1) * 0.1; targetZ = -1;
        targetScale = 1.0 + (util / 100) * 0.1;
        targetOpacity = 1.0;
      } else {
        targetX = -3.5; targetY = 1; targetZ = -2;
        targetScale = 0.4;
        targetOpacity = 0.08;
      }
    }

    groupRef.current.position.x = damp(groupRef.current.position.x, targetX, 4, delta);
    groupRef.current.position.y = damp(groupRef.current.position.y, targetY, 4, delta);
    groupRef.current.position.z = damp(groupRef.current.position.z, targetZ, 4, delta);
    
    const s = damp(groupRef.current.scale.x, targetScale, 4, delta);
    groupRef.current.scale.set(s, s, s);

    solidMatRef.current.opacity = damp(solidMatRef.current.opacity, targetOpacity, 4, delta);
    solidMatRef.current.transparent = solidMatRef.current.opacity < 0.99;
    solidMatRef.current.emissiveIntensity = damp(solidMatRef.current.emissiveIntensity, (util / 30) * (0.8 + Math.random() * 0.4), 8, delta); // Neural flicker effect

    const wireTargetOpacity = targetOpacity < 0.9 ? 0 : 0.3 + (util / 100);
    wireMatRef.current.opacity = damp(wireMatRef.current.opacity, wireTargetOpacity, 5, delta);

    const rotMultiplier = scrollProgress >= 0.33 && scrollProgress < 0.50 ? 1.0 : 0.3;
    groupRef.current.rotation.x += delta * (0.2 + util / 250) * rotMultiplier;
    groupRef.current.rotation.y += delta * 0.35 * rotMultiplier;
    groupRef.current.rotation.z += delta * 0.1 * rotMultiplier;
  });

  return (
    <group ref={groupRef}>
      <mesh>
        <icosahedronGeometry args={[1, 0]} />
        <meshPhysicalMaterial 
          ref={solidMatRef} 
          color="#15000a" 
          emissive="#FF2244" 
          roughness={0.2} 
          metalness={0.8}
          clearcoat={0.5}
          transparent
        />
      </mesh>
      <mesh scale={1.03}>
        <icosahedronGeometry args={[1, 0]} />
        <meshBasicMaterial 
          ref={wireMatRef} 
          color="#FF4466" 
          wireframe 
          transparent
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
};

// 3. STATIC_GREEDY (Crumpled Sphere - The Baseline Heavyweight)
const StaticGreedyModel = ({ util, scrollProgress }: { util: number, scrollProgress: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(1, 64, 64);
    const posAttribute = geo.getAttribute('position');
    const vertex = new THREE.Vector3();
    for (let i = 0; i < posAttribute.count; i++) {
      vertex.fromBufferAttribute(posAttribute, i);
      const n = noise3D(vertex.x * 2.0, vertex.y * 2.0, vertex.z * 2.0);
      vertex.multiplyScalar(1 + n * 0.3);
      posAttribute.setXYZ(i, vertex.x, vertex.y, vertex.z);
    }
    geo.computeVertexNormals();
    return geo;
  }, []);

  useFrame((state, delta) => {
    if (!meshRef.current || !materialRef.current) return;

    let targetX = 0, targetY = 0, targetZ = 0, targetScale = 1.0, targetOpacity = 1.0;

    if (scrollProgress <= 0.16) {
      const angle = state.clock.elapsedTime * 0.2 + (Math.PI * 4 / 3);
      targetX = Math.cos(angle) * 2.0;
      targetZ = Math.sin(angle) * 2.0 - 2;
    } else if (scrollProgress > 0.16 && scrollProgress < 0.50) {
      targetX = 3.5; targetY = -1; targetZ = -2;
      targetScale = 0.4;
      targetOpacity = 0.08;
    } else if (scrollProgress >= 0.50 && scrollProgress < 0.66) {
      // Focused - heavy, very slow drift
      targetX = 1.8; // Shifted right
      targetY = Math.sin(state.clock.elapsedTime * 0.5) * 0.05;
      targetZ = 0;
      targetScale = 1.6;
      targetOpacity = 1.0;
    } else {
      targetX = 0; targetY = -2.5 + Math.sin(state.clock.elapsedTime * 0.8) * 0.05; targetZ = -1;
      targetScale = 1.0;
      targetOpacity = 1.0;
    }

    meshRef.current.position.x = damp(meshRef.current.position.x, targetX, 3, delta);
    meshRef.current.position.y = damp(meshRef.current.position.y, targetY, 3, delta);
    meshRef.current.position.z = damp(meshRef.current.position.z, targetZ, 3, delta);
    
    const s = damp(meshRef.current.scale.x, targetScale, 3, delta);
    meshRef.current.scale.set(s, s, s);

    materialRef.current.opacity = damp(materialRef.current.opacity, targetOpacity, 4, delta);
    materialRef.current.transparent = materialRef.current.opacity < 0.99;
    
    materialRef.current.emissiveIntensity = damp(materialRef.current.emissiveIntensity, (util / 80), 3, delta);

    // Very slow, monolithic rotation
    meshRef.current.rotation.x += delta * 0.05;
    meshRef.current.rotation.y += delta * 0.08;
  });

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial 
        ref={materialRef} 
        color="#222222" 
        emissive="#777777" 
        roughness={0.6} 
        metalness={0.9}
        transparent
      />
    </mesh>
  );
};

// 4. CORE (TorusKnot - The Network Hub)
const CoreModel = ({ avgUtil, scrollProgress }: { avgUtil: number, scrollProgress: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshPhysicalMaterial>(null);

  useFrame((state, delta) => {
    if (!meshRef.current || !materialRef.current) return;

    // Rises from below on Beat 4, creating an emotional arrival
    const targetY = scrollProgress > 0.66 ? Math.sin(state.clock.elapsedTime * 1.5) * 0.1 : -5;
    const targetOpacity = scrollProgress > 0.66 ? 1 : 0;
    const targetScale = scrollProgress > 0.66 ? 1.4 : 0.5;

    meshRef.current.position.y = damp(meshRef.current.position.y, targetY, 3, delta);
    
    const s = damp(meshRef.current.scale.x, targetScale, 3, delta);
    meshRef.current.scale.set(s, s, s);

    materialRef.current.opacity = damp(materialRef.current.opacity, targetOpacity, 4, delta);
    materialRef.current.transparent = materialRef.current.opacity < 0.99;
    
    // Core throbs strongly when active
    const throb = Math.sin(state.clock.elapsedTime * 3) * 0.3;
    materialRef.current.emissiveIntensity = damp(materialRef.current.emissiveIntensity, (avgUtil / 25) + Math.max(0, throb), 5, delta);

    meshRef.current.rotation.x += delta * 0.3;
    meshRef.current.rotation.y += delta * 0.4;
  });

  return (
    <mesh ref={meshRef} position={[0, -5, 0]}>
      <torusKnotGeometry args={[1.2, 0.4, 256, 32]} />
      <meshPhysicalMaterial 
        ref={materialRef} 
        color="#001a1a" 
        emissive="#00E5FF" 
        roughness={0.1} 
        metalness={0.5} 
        transmission={0.8}
        clearcoat={1.0}
        thickness={2.5}
        ior={1.5}
        transparent
        opacity={0}
      />
    </mesh>
  );
};

// Data stream lines
const DataStreams = ({ scrollProgress, data }: { scrollProgress: number, data: SliceData[] }) => {
  const linesRef = useRef<THREE.Group>(null);

  const l1 = useMemo(() => new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-3, 2, -1), new THREE.Vector3(0, 0, 0)]), new THREE.LineBasicMaterial({ color: "#00FF88", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, linewidth: 2 })), []);
  const l2 = useMemo(() => new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(3, 2, -1), new THREE.Vector3(0, 0, 0)]), new THREE.LineBasicMaterial({ color: "#FF2244", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, linewidth: 2 })), []);
  const l3 = useMemo(() => new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -2.5, -1), new THREE.Vector3(0, 0, 0)]), new THREE.LineBasicMaterial({ color: "#888888", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, linewidth: 2 })), []);

  useFrame((state, delta) => {
    if (!linesRef.current) return;
    const time = state.clock.elapsedTime;
    
    const cUtil = data.find(d => d.algorithm === 'C_ADMM')?.utilValue || 0;
    const mUtil = data.find(d => d.algorithm === 'MAAN')?.utilValue || 0;
    const sUtil = data.find(d => d.algorithm === 'Static_Greedy')?.utilValue || 0;

    const targetOpacityMult = scrollProgress > 0.66 ? 0.6 : 0;
    
    const lines = linesRef.current.children as THREE.Line[];
    if (lines.length >= 3) {
      const mat1 = lines[0].material as THREE.LineBasicMaterial;
      const mat2 = lines[1].material as THREE.LineBasicMaterial;
      const mat3 = lines[2].material as THREE.LineBasicMaterial;

      // Soft pulse rather than sudden flashes
      mat1.opacity = damp(mat1.opacity, targetOpacityMult * ((cUtil / 100) + Math.sin(time * 4) * 0.2), 6, delta);
      mat2.opacity = damp(mat2.opacity, targetOpacityMult * ((mUtil / 100) + Math.sin(time * 4.5 + 2) * 0.2), 6, delta);
      mat3.opacity = damp(mat3.opacity, targetOpacityMult * ((sUtil / 100) + Math.sin(time * 3.5 + 4) * 0.2), 6, delta);
    }
  });

  return (
    <group ref={linesRef}>
      <primitive object={l1} />
      <primitive object={l2} />
      <primitive object={l3} />
    </group>
  );
};


// Space Environment (Dynamic Stars & IBL Lighting)
const SpaceEnvironment = () => {
  return (
    <>
      {/* Colorful Deep Space Nebulas! */}
      <CustomNebula position={[-30, 10, -50]} scale={[80, 50, 1]} color1="#FF2266" color2="#4400FF" opacityMod={0.6} rotation={[0, 0, Math.PI/6]} />
      <CustomNebula position={[35, -15, -60]} scale={[100, 60, 1]} color1="#00E5FF" color2="#0022AA" opacityMod={0.5} rotation={[0, 0, -Math.PI/4]} />
      <CustomNebula position={[0, 30, -70]} scale={[120, 40, 1]} color1="#00FF88" color2="#005522" opacityMod={0.3} rotation={[0, 0, Math.PI]} />
      <CustomNebula position={[0, -20, -40]} scale={[60, 30, 1]} color1="#FF8800" color2="#FF0044" opacityMod={0.25} />

      <Stars radius={20} depth={60} count={6000} factor={6} saturation={1} fade speed={1.5} />
      
      {/* Faint Hexagonal / Geometric Space Web */}
      <mesh scale={30}>
        <icosahedronGeometry args={[1, 4]} />
        <meshBasicMaterial color="#00E5FF" wireframe transparent opacity={0.03} />
      </mesh>

      {/* Bake purely from stars - no artificial lights in the scene */}
      <Environment background={false} resolution={512} frames={1}>
        {/* We build a custom scene specifically for the reflections to capture */}
        <mesh scale={100}>
          <sphereGeometry args={[1, 32, 32]} />
          <meshBasicMaterial color="#02050A" side={THREE.BackSide} />
        </mesh>
        
        {/* Render bright stars for reflections */}
        <Stars radius={10} depth={30} count={3000} factor={8} saturation={1} />
        
        {/* Super-bright 'hero' star clusters to act as dramatic specular light sources */}
        <mesh position={[15, 20, 10]}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#00E5FF" /></mesh>
        <mesh position={[-20, -10, 15]}><sphereGeometry args={[2, 16, 16]} /><meshBasicMaterial color="#FF2244" /></mesh>
        <mesh position={[5, -25, -20]}><sphereGeometry args={[1.5, 16, 16]} /><meshBasicMaterial color="#FFFFFF" /></mesh>
        <mesh position={[-15, 25, -15]}><sphereGeometry args={[1.8, 16, 16]} /><meshBasicMaterial color="#00FF88" /></mesh>
      </Environment>
    </>
  );
};

// Scene Controller (Camera Parallax Only)
const SceneController = ({ scrollProgress }: { scrollProgress: number }) => {
  const { camera, pointer } = useThree();

  useFrame((state, delta) => {
    // Camera inertia and drift - the product photography feel
    const baseZ = scrollProgress > 0.66 ? 9.5 : 7.5; // Slightly further back for the grand finale
    let nextCamX = 0, nextCamY = 0, nextCamZ = baseZ;

    // Subtle parallax tracking the mouse
    const pointerDriftX = pointer.x * 0.4;
    const pointerDriftY = pointer.y * 0.4;

    if (scrollProgress <= 0.16) {
      // Introduction
      nextCamX = pointerDriftX;
      nextCamY = pointerDriftY;
      nextCamZ += Math.sin(state.clock.elapsedTime * 0.2) * 0.3;
    } else if (scrollProgress > 0.16 && scrollProgress <= 0.66) {
      // Focused objects - subtle orbit + parallax
      const orbitOrbit = state.clock.elapsedTime * 0.1;
      nextCamX = Math.sin(orbitOrbit) * 1.0 + pointerDriftX;
      nextCamY = Math.cos(orbitOrbit * 0.8) * 0.5 + pointerDriftY;
      nextCamZ += Math.sin(orbitOrbit * 1.5) * 0.2;
    } else {
      // Constellation overview
      nextCamX = Math.sin(state.clock.elapsedTime * 0.15) * 2.5 + pointerDriftX * 1.5;
      nextCamY = pointerDriftY * 1.5;
      nextCamZ = baseZ + Math.cos(state.clock.elapsedTime * 0.15) * 2 - 1;
    }

    camera.position.x = damp(camera.position.x, nextCamX, 3, delta);
    camera.position.y = damp(camera.position.y, nextCamY, 3, delta);
    camera.position.z = damp(camera.position.z, nextCamZ, 3, delta);
    
    // Smooth lookAt tracking origin but slightly offset by pointer
    camera.lookAt(pointer.x * -0.5, pointer.y * -0.5, 0);
  });

  return null;
};


export const NetworkSliceCanvas = ({ scrollProgress, data }: CanvasProps) => {
  const cUtil = data.find(d => d.algorithm === 'C_ADMM')?.utilValue || 0;
  const mUtil = data.find(d => d.algorithm === 'MAAN')?.utilValue || 0;
  const sUtil = data.find(d => d.algorithm === 'Static_Greedy')?.utilValue || 0;
  const avgUtil = (cUtil + mUtil + sUtil) / 3;

  return (
    <Suspense fallback={<Loader />}>
      <Canvas 
        camera={{ position: [0, 0, 7.5], fov: 45 }} 
        style={{ background: 'transparent', position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} 
        gl={{ alpha: true, antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <SceneController scrollProgress={scrollProgress} />
        
        {/* Dynamic Space Background & IBL Star Lighting */}
        <SpaceEnvironment />
        
        {/* The interactive Boids-physics Spaceman! */}
        <Astronaut scrollProgress={scrollProgress} />
        
        <group position={[0, 0.5, 0]}>
          <CadmmModel util={cUtil} scrollProgress={scrollProgress} />
          <MaanModel util={mUtil} scrollProgress={scrollProgress} />
          <StaticGreedyModel util={sUtil} scrollProgress={scrollProgress} />
          <CoreModel avgUtil={avgUtil} scrollProgress={scrollProgress} />
          <DataStreams data={data} scrollProgress={scrollProgress} />
        </group>

        {/* Soft, premium contact shadow that grounds the product in reality */}
        <ContactShadows 
          position={[0, -3.5, 0]} 
          opacity={0.35} 
          scale={25} 
          blur={2.5} 
          far={10} 
          resolution={512} 
          color="#000000" 
        />
      </Canvas>
    </Suspense>
  );
};

const Loader = () => {
  return (
    <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center bg-[#0A0A0F] text-white z-50">
      <div className="w-10 h-10 border-2 border-white/10 border-t-[#00FF88] rounded-full animate-spin mb-6 shadow-sm"></div>
      <div className="font-mono text-[0.7rem] tracking-[0.2em] text-white/50 uppercase">Initializing Core</div>
    </div>
  );
};

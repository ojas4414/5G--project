'use client';

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Sparkles } from '@react-three/drei';
import * as THREE from 'three';
import { createNoise3D } from 'simplex-noise';

interface Props {
  label: string;
  utilValue: number;
  color: string;
  geometryType: string;
  radius: number;
  startAngle?: number;
  isFocused?: boolean;
  isDimmed?: boolean;
  history?: number[];
}

export const AlgorithmNode: React.FC<Props> = ({ label, utilValue, color, geometryType, radius, startAngle = 0, isFocused, isDimmed, history }) => {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const materialRef = useRef<any>(null);
  
  const angleRef = useRef(startAngle);
  const noise3D = useMemo(() => createNoise3D(), []);

  // Use the lowercase or exactly mapped names
  const geoType = geometryType.toLowerCase();
  
  const geo = useMemo(() => {
    switch(geoType) {
      case 'octahedron': return new THREE.OctahedronGeometry(0.8, 0);
      case 'icosahedron': return new THREE.IcosahedronGeometry(0.8, 0);
      case 'crumpledsphere':
      case 'sphere': {
        const g = new THREE.SphereGeometry(0.8, 64, 64);
        const pos = g.attributes.position;
        const vec = new THREE.Vector3();
        for (let i = 0; i < pos.count; i++) {
          vec.fromBufferAttribute(pos, i);
          const n = noise3D(vec.x * 2, vec.y * 2, vec.z * 2);
          vec.normalize().multiplyScalar(0.8 + n * 0.15);
          pos.setXYZ(i, vec.x, vec.y, vec.z);
        }
        g.computeVertexNormals();
        return g;
      }
      case 'torusknot':
      default:
        return new THREE.TorusKnotGeometry(0.5, 0.15, 128, 16);
    }
  }, [geoType, noise3D]);

  useFrame((state, delta) => {
    const cappedDelta = Math.min(delta, 0.05);
    
    // Independent orbit radius + speed
    if (groupRef.current && radius > 0) {
      
      if (isFocused) {
        groupRef.current.position.x = THREE.MathUtils.damp(groupRef.current.position.x, 0, 4, cappedDelta);
        groupRef.current.position.y = THREE.MathUtils.damp(groupRef.current.position.y, 0, 4, cappedDelta);
        groupRef.current.position.z = THREE.MathUtils.damp(groupRef.current.position.z, 2, 4, cappedDelta);
        groupRef.current.scale.setScalar(THREE.MathUtils.damp(groupRef.current.scale.x, 2, 4, cappedDelta));
      } else if (isDimmed) {
        groupRef.current.position.x = THREE.MathUtils.damp(groupRef.current.position.x, Math.cos(angleRef.current) * radius * 1.5, 4, cappedDelta);
        groupRef.current.position.y = THREE.MathUtils.damp(groupRef.current.position.y, 0, 4, cappedDelta);
        groupRef.current.position.z = THREE.MathUtils.damp(groupRef.current.position.z, Math.sin(angleRef.current) * radius - 2, 4, cappedDelta);
        groupRef.current.scale.setScalar(THREE.MathUtils.damp(groupRef.current.scale.x, 0.4, 4, cappedDelta));
      } else {
        // Normal Orbit
        const dynamicSpeed = 0.2 + (utilValue / 100) * 0.8;
        angleRef.current += cappedDelta * dynamicSpeed;
        
        groupRef.current.position.x = THREE.MathUtils.damp(groupRef.current.position.x, Math.cos(angleRef.current) * radius, 4, cappedDelta);
        groupRef.current.position.y = THREE.MathUtils.damp(groupRef.current.position.y, 0, 4, cappedDelta);
        groupRef.current.position.z = THREE.MathUtils.damp(groupRef.current.position.z, Math.sin(angleRef.current) * radius, 4, cappedDelta);
        groupRef.current.scale.setScalar(THREE.MathUtils.damp(groupRef.current.scale.x, 1, 4, cappedDelta));
      }
      
    } else if (groupRef.current && radius === 0) {
      // averageUtil maps here for Torus knot rotation
      const torusSpeed = utilValue / 100;
      groupRef.current.rotation.y += cappedDelta * torusSpeed * 2;
    }
    
    // Rotation logic
    if (meshRef.current) {
      meshRef.current.rotation.x += cappedDelta * 0.5;
      meshRef.current.rotation.y += cappedDelta * 0.3;
    }

    if (materialRef.current) {
      materialRef.current.emissiveIntensity = utilValue / 40;
    }
  });

  return (
    <group ref={groupRef}>
      <mesh ref={meshRef} geometry={geo}>
        {geoType === 'octahedron' && (
          <meshPhysicalMaterial ref={materialRef} color={color} emissive="#00FF88" emissiveIntensity={utilValue / 40} transmission={0.9} opacity={isDimmed ? 0.2 : 1} transparent roughness={0.1} />
        )}
        {geoType === 'icosahedron' && (
          <>
            <meshStandardMaterial ref={materialRef} color={color} emissive="#FF2244" emissiveIntensity={utilValue / 40} transparent opacity={isDimmed ? 0.2 : 1} />
            <mesh geometry={geo}>
              <meshBasicMaterial color="#FF2244" wireframe transparent opacity={isDimmed ? 0.2 : 1} />
            </mesh>
          </>
        )}
        {(geoType === 'sphere' || geoType === 'crumpledsphere') && (
          <meshStandardMaterial ref={materialRef} color={color} emissive="#CCCCCC" emissiveIntensity={utilValue / 40} metalness={1.0} roughness={0.4} transparent opacity={isDimmed ? 0.2 : 1} />
        )}
        {(geoType === 'torusknot' || geoType === 'default') && (
          <meshStandardMaterial ref={materialRef} color={color} emissive="#00E5FF" emissiveIntensity={utilValue / 40} wireframe transparent opacity={isDimmed ? 0.2 : 1} />
        )}
      </mesh>
      
      {utilValue > 80 && radius > 0 && !isDimmed && (
        <Sparkles count={50} scale={1.2} size={2} color={color} speed={0.5} opacity={0.8} />
      )}

      {radius > 0 && (
        <Html position={[0, isFocused ? -1.0 : -1.2, 0]} center className="pointer-events-none select-none">
          <div className={`flex flex-col ${isFocused ? 'items-start bg-[#020810]/70 p-4 min-w-[120px] scale-150' : 'items-center bg-[#020810]/90 px-3 py-1.5'} border border-white/10 backdrop-blur-lg transition-all duration-700`}>
            <span className={`uppercase font-mono tracking-widest text-white/50 ${isFocused ? 'text-[8px] mb-1' : 'text-[9px]'}`}>{label}</span>
            <span className={`font-mono font-bold mt-0.5 ${isFocused ? 'text-2xl' : 'text-sm'}`} style={{ color }}>
              {utilValue.toFixed(1)} UTIL
            </span>
            
            {isFocused && history && (
              <div className="flex items-end gap-[2px] h-6 mt-3 w-full opacity-60">
                {history.map((val, i) => (
                  <div key={i} className="flex-1 transition-all duration-300" style={{ height: (Math.max(10, val) + '%'), backgroundColor: color }} />
                ))}
              </div>
            )}
          </div>
        </Html>
      )}
    </group>
  );
};

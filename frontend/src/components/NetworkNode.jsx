import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Html } from '@react-three/drei';
import * as THREE from 'three';

// Maps algorithm names to different geometric geometries and colors
const ALGO_CONFIG = {
  'MAAN': { geometry: 'icosahedron', color: '#ff3b30' },          // Red
  'Independent_MAPPO': { geometry: 'torusKnot', color: '#0A84FF' },// Blue
  'C_ADMM': { geometry: 'octahedron', color: '#30d158' },           // Green
  'Static_Greedy': { geometry: 'box', color: '#f5f5f7' },           // White
  'OMD_BF': { geometry: 'sphere', color: '#bf5af2' }                // Purple (fallback)
};

export default function NetworkNode({ dataItem, index, total }) {
  const meshRef = useRef();
  
  // Arrange nodes in a circle based on index
  const radius = 4;
  const angle = (index / total) * Math.PI * 2;
  const position = [
    Math.cos(angle) * radius,
    Math.sin(angle * 2) * 1.5, // Dynamic height variation
    Math.sin(angle) * radius
  ];

  const config = ALGO_CONFIG[dataItem.algorithm] || { geometry: 'box', color: '#ffffff' };
  
  // Scale based on utility or load - dynamic data reception
  const scaleValue = dataItem.mean_utility ? Math.max(0.5, dataItem.mean_utility / 10) : 1;
  const scale = [scaleValue, scaleValue, scaleValue];

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.5;
      meshRef.current.rotation.y += delta * 0.3;
    }
  });

  const getGeometry = () => {
    switch (config.geometry) {
      case 'icosahedron': return <icosahedronGeometry args={[1, 0]} />;
      case 'torusKnot': return <torusKnotGeometry args={[0.7, 0.2, 100, 16]} />;
      case 'octahedron': return <octahedronGeometry args={[1, 0]} />;
      case 'sphere': return <sphereGeometry args={[1, 32, 32]} />;
      case 'box': default: return <boxGeometry args={[1.5, 1.5, 1.5]} />;
    }
  };

  return (
    <Float speed={2} rotationIntensity={1.5} floatIntensity={2}>
      <mesh ref={meshRef} position={position} scale={scale}>
        {getGeometry()}
        <meshPhysicalMaterial 
          color={config.color} 
          metalness={0.8}
          roughness={0.2}
          clearcoat={1.0}
          clearcoatRoughness={0.1}
          emissive={config.color}
          emissiveIntensity={0.5}
          wireframe={false}
        />
        
        {/* Floating HTML Label for the 3D Node */}
        <Html distanceFactor={15} center>
          <div style={{
            background: 'rgba(0,0,0,0.6)',
            padding: '4px 8px',
            borderRadius: '4px',
            borderLeft: `3px solid ${config.color}`,
            color: 'white',
            fontWeight: 'bold',
            fontSize: '12px',
            backdropFilter: 'blur(4px)',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            userSelect: 'none'
          }}>
            {dataItem.algorithm}
            <br/>
            <span style={{ fontSize: '10px', color: '#aaa' }}>Util: {dataItem.mean_utility?.toFixed(2) || 'N/A'}</span>
          </div>
        </Html>
      </mesh>
    </Float>
  );
}

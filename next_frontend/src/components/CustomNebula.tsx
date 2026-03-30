'use client';

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface NebulaProps {
  color1?: string;
  color2?: string;
  opacityMod?: number;
  position?: [number, number, number];
  scale?: [number, number, number];
  rotation?: [number, number, number];
}

export const CustomNebula = ({ 
  color1 = '#00E5FF', 
  color2 = '#B400FF', 
  opacityMod = 1,
  position = [0, 0, -10],
  scale = [30, 20, 1],
  rotation = [0, 0, 0]
}: NebulaProps) => {
  const matRef = useRef<THREE.ShaderMaterial>(null);

  const shaderArgs = useMemo(() => {
    return {
      uniforms: {
        uColor1: { value: new THREE.Color(color1) },
        uColor2: { value: new THREE.Color(color2) },
        uTime: { value: Math.random() * 100 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor1;
        uniform vec3 uColor2;
        uniform float uTime;
        varying vec2 vUv;
        
        float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f*f*(3.0-2.0*f);
          return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
                     mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
        }
        
        void main() {
          vec2 uv = vUv;
          // Organic, sweeping fluid noise
          float n = noise(uv * 2.0 + uTime * 0.05) * noise(uv * 4.0 - uTime * 0.08);
          float n2 = noise(uv * 6.0 + uTime * 0.1);
          
          vec3 col = mix(uColor1, uColor2, uv.x + n2*0.2);
          
          float alpha = n * 2.0;
          // Soften the edges of the plane
          float edgeX = smoothstep(0.0, 0.2, uv.x) * smoothstep(1.0, 0.8, uv.x);
          float edgeY = smoothstep(0.0, 0.2, uv.y) * smoothstep(1.0, 0.8, uv.y);
          
          gl_FragColor = vec4(col, alpha * edgeX * edgeY * 0.3); 
        }
      `
    };
  }, [color1, color2]);

  useFrame((state, delta) => {
    if (matRef.current) {
      matRef.current.uniforms.uTime.value += delta * 0.5;
      matRef.current.opacity = opacityMod;
    }
  });

  return (
    <mesh position={position} scale={scale} rotation={rotation}>
      <planeGeometry args={[1, 1, 32, 32]} />
      <shaderMaterial
        ref={matRef}
        args={[shaderArgs]}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  );
};

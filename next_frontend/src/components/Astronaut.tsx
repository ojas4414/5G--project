'use client';

import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export function Astronaut({ scrollProgress }: { scrollProgress: number }) {
  const group = useRef<THREE.Group>(null);
  const { pointer, viewport } = useThree();

  // Individual limb refs for procedural animation
  const leftArmRef = useRef<THREE.Mesh>(null);
  const rightArmRef = useRef<THREE.Mesh>(null);
  const leftLegRef = useRef<THREE.Mesh>(null);
  const rightLegRef = useRef<THREE.Mesh>(null);
  const headRef = useRef<THREE.Group>(null);
  const jetpackRef = useRef<THREE.Mesh>(null);
  const chestGlowRef = useRef<THREE.MeshPhysicalMaterial>(null);
  const leftThrusterRef = useRef<THREE.Mesh>(null);
  const rightThrusterRef = useRef<THREE.Mesh>(null);

  const targetPos = useRef(new THREE.Vector3());
  const velocity = useRef(new THREE.Vector3());

  useFrame((state, delta) => {
    if (!group.current) return;
    const t = state.clock.elapsedTime;

    // 1. SCROLL TRACKING
    const yTrajectory = 6 - (scrollProgress * 12);
    const organicX = Math.sin(t * 0.4) * 2.5;
    const organicZ = Math.cos(t * 0.3) * 1.5 - 2;
    const organicY = Math.sin(t * 0.7) * 0.8;

    const baseX = organicX;
    const baseY = yTrajectory + organicY;
    const baseZ = organicZ;

    // 2. MOUSE REPULSION PHYSICS
    const mouse3D = new THREE.Vector3(
      (pointer.x * viewport.width) / 2,
      (pointer.y * viewport.height) / 2,
      group.current.position.z
    );
    const dist = group.current.position.distanceTo(mouse3D);
    const fearRadius = 4.0;

    if (dist < fearRadius) {
      const escapeVec = new THREE.Vector3().subVectors(group.current.position, mouse3D).normalize();
      const force = (fearRadius - dist) * 15.0;
      velocity.current.add(escapeVec.multiplyScalar(force * delta));
    }
    velocity.current.multiplyScalar(0.92);

    targetPos.current.set(baseX, baseY, baseZ).add(velocity.current);
    group.current.position.lerp(targetPos.current, delta * 4);

    // 3. BODY TUMBLING
    const isPanicking = velocity.current.length() > 0.3;
    if (isPanicking) {
      group.current.rotation.x += velocity.current.y * delta * 2;
      group.current.rotation.y += velocity.current.x * delta * 2;
      group.current.rotation.z += velocity.current.x * delta;
    } else {
      group.current.rotation.x = THREE.MathUtils.damp(group.current.rotation.x, t * 0.2, 2, delta);
      group.current.rotation.y = THREE.MathUtils.damp(group.current.rotation.y, t * 0.25, 2, delta);
      group.current.rotation.z = THREE.MathUtils.damp(group.current.rotation.z, t * 0.1, 2, delta);
    }

    // 4. LIMB ANIMATIONS — make it feel alive, not a mannequin!
    const panicMult = isPanicking ? 3.0 : 1.0;

    // Arms: gentle floating sway in zero-G, frantic flailing when fleeing
    if (leftArmRef.current) {
      leftArmRef.current.rotation.z = 0.3 + Math.sin(t * 1.2 * panicMult) * 0.4 * panicMult;
      leftArmRef.current.rotation.x = Math.sin(t * 0.8 * panicMult + 1.0) * 0.3 * panicMult;
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.z = -0.3 + Math.sin(t * 1.2 * panicMult + Math.PI) * 0.4 * panicMult;
      rightArmRef.current.rotation.x = Math.sin(t * 0.8 * panicMult + 2.0) * 0.3 * panicMult;
    }

    // Legs: slow lazy kicks, like floating in a pool
    if (leftLegRef.current) {
      leftLegRef.current.rotation.x = Math.sin(t * 0.6 * panicMult) * 0.25 * panicMult;
      leftLegRef.current.rotation.z = Math.sin(t * 0.4) * 0.05;
    }
    if (rightLegRef.current) {
      rightLegRef.current.rotation.x = Math.sin(t * 0.6 * panicMult + Math.PI) * 0.25 * panicMult;
      rightLegRef.current.rotation.z = Math.sin(t * 0.4 + 1.5) * 0.05;
    }

    // Head: subtle curious look-around
    if (headRef.current) {
      headRef.current.rotation.y = Math.sin(t * 0.5) * 0.2;
      headRef.current.rotation.x = Math.sin(t * 0.3 + 0.5) * 0.1;
    }

    // Chest glow: pulsing heartbeat
    if (chestGlowRef.current) {
      chestGlowRef.current.emissiveIntensity = 1.5 + Math.sin(t * 3) * 1.0;
    }

    // Thrusters: tiny vibration when panicking
    if (isPanicking) {
      if (leftThrusterRef.current) {
        leftThrusterRef.current.position.y = -0.5 + Math.random() * 0.03;
      }
      if (rightThrusterRef.current) {
        rightThrusterRef.current.position.y = -0.5 + Math.random() * 0.03;
      }
    }
  });

  return (
    <group ref={group} scale={[0.5, 0.5, 0.5]}>
      {/* Head Group with look-around */}
      <group ref={headRef} position={[0, 1.3, 0]}>
        <mesh>
          <sphereGeometry args={[0.45, 32, 32]} />
          <meshStandardMaterial color="#ffffff" roughness={0.3} />
        </mesh>
        {/* Helmet Visor */}
        <mesh position={[0, 0, 0.28]}>
          <sphereGeometry args={[0.35, 32, 16]} />
          <meshPhysicalMaterial color="#0A0A0F" metalness={0.9} roughness={0.05} clearcoat={1.0} clearcoatRoughness={0.1} />
        </mesh>
      </group>

      {/* Torso */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.5, 0.45, 1.6, 32]} />
        <meshStandardMaterial color="#ffffff" roughness={0.4} />
      </mesh>

      {/* Jetpack */}
      <mesh ref={jetpackRef} position={[0, 0.2, -0.6]}>
        <boxGeometry args={[0.7, 1.2, 0.4]} />
        <meshStandardMaterial color="#cccccc" metalness={0.6} roughness={0.2} />
      </mesh>

      {/* Thrusters */}
      <mesh ref={leftThrusterRef} position={[-0.25, -0.5, -0.6]}>
        <cylinderGeometry args={[0.1, 0.15, 0.3, 16]} />
        <meshStandardMaterial color="#333333" />
      </mesh>
      <mesh ref={rightThrusterRef} position={[0.25, -0.5, -0.6]}>
        <cylinderGeometry args={[0.1, 0.15, 0.3, 16]} />
        <meshStandardMaterial color="#333333" />
      </mesh>

      {/* Left Arm - animated */}
      <mesh ref={leftArmRef} position={[-0.75, 0.2, 0]}>
        <capsuleGeometry args={[0.18, 0.9, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* Right Arm - animated */}
      <mesh ref={rightArmRef} position={[0.75, 0.2, 0]}>
        <capsuleGeometry args={[0.18, 0.9, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* Left Leg - animated */}
      <mesh ref={leftLegRef} position={[-0.25, -1.3, 0]}>
        <capsuleGeometry args={[0.22, 1.0, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* Right Leg - animated */}
      <mesh ref={rightLegRef} position={[0.25, -1.3, 0]}>
        <capsuleGeometry args={[0.22, 1.0, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>

      {/* Glowing chest mechanism - pulsing heartbeat */}
      <mesh position={[0, 0.3, 0.45]}>
        <boxGeometry args={[0.3, 0.2, 0.1]} />
        <meshPhysicalMaterial ref={chestGlowRef} color="#00E5FF" emissive="#00E5FF" emissiveIntensity={2} />
      </mesh>
    </group>
  );
}

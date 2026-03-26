import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment, ContactShadows } from '@react-three/drei';
import NetworkNode from './NetworkNode';

export default function Scene({ data }) {
  return (
    <div className="canvas-container">
      <Canvas camera={{ position: [0, 5, 10], fov: 50 }}>
        
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffffff" />
        <pointLight position={[-10, -10, -5]} intensity={1} color="#0A84FF" />
        <pointLight position={[0, 10, -10]} intensity={1} color="#ff3b30" />
        
        <Suspense fallback={null}>
          <Environment preset="city" />
          
          {/* Dynamic models appearing based on data received */}
          <group position={[0, -1, 0]}>
            {data && data.length > 0 ? (
              data.map((item, index) => (
                <NetworkNode key={index} dataItem={item} index={index} total={data.length} />
              ))
            ) : (
              // Fallback demo nodes if no data is available
              ['MAAN', 'Independent_MAPPO', 'C_ADMM', 'Static_Greedy'].map((algo, idx) => (
                <NetworkNode key={idx} dataItem={{ algorithm: algo, mean_utility: 15 }} index={idx} total={4} />
              ))
            )}
            
            <ContactShadows position={[0, -3, 0]} opacity={0.4} scale={20} blur={2} far={4.5} color="#0A84FF" />
          </group>
        </Suspense>
        
        <OrbitControls 
          enablePan={false} 
          maxPolarAngle={Math.PI / 2 + 0.1} 
          minDistance={3} 
          maxDistance={20} 
          autoRotate 
          autoRotateSpeed={0.5} 
        />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={1} fade speed={1} />
      </Canvas>
    </div>
  );
}

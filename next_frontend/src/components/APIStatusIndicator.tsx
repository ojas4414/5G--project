'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface Props {
  status: 'connected' | 'disconnected';
  lastUpdated: number;
  onReconnect: () => void;
}

export const APIStatusIndicator: React.FC<Props> = ({ status, lastUpdated, onReconnect }) => {
  const isConnected = status === 'connected';

  return (
    <div className="fixed top-8 right-8 z-50 flex flex-col items-end pointer-events-auto font-mono text-xs uppercase tracking-widest">
      <div className="flex items-center gap-3 bg-[#020810]/80 border border-white/10 px-4 py-2 backdrop-blur-md">
        <span style={{ color: isConnected ? '#00FF88' : '#FF2244' }}>
          {isConnected ? 'LIVE TELEMETRY' : 'DISCONNECTED'}
        </span>
        <motion.div
          className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[#00FF88]' : 'bg-[#FF2244]'}`}
          animate={{
            opacity: isConnected ? [1, 0.4, 1] : 1,
            scale: isConnected ? [1, 1.2, 1] : 1
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      <div className="mt-2 text-[10px] text-white/40">
        Update: {new Date(lastUpdated).toISOString().split('T')[1].replace('Z', '')}
      </div>

      {!isConnected && (
        <button
          onClick={onReconnect}
          className="mt-3 px-4 py-2 border border-cyan-400/50 text-cyan-400 hover:bg-cyan-400 hover:text-[#020810] transition-colors duration-300"
        >
          FORCE RECONNECT
        </button>
      )}
    </div>
  );
};

'use client';
import React from 'react';

export const ControlPanel = ({ 
  activeSlices, setActiveSlices, 
  maxLoad, setMaxLoad, 
  isSimulation, setIsSimulation 
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
}: any) => {
  return (
    <div className="fixed bottom-8 left-8 z-50 glass-panel-sharp p-6 w-80 pointer-events-auto">
      <h3 className="text-cyan-400 text-xs font-mono uppercase tracking-widest mb-4 border-b border-white/10 pb-2">Simulation Controls</h3>
      
      <div className="mb-4">
        <div className="flex justify-between text-xs font-mono text-white/70 mb-1">
          <span>Active Slices</span>
          <span className="text-white">{activeSlices}</span>
        </div>
        <input 
          type="range" min="1" max="10" step="1" 
          value={activeSlices} 
          onChange={(e) => setActiveSlices(parseInt(e.target.value))}
          className="w-full accent-cyan-400"
        />
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-xs font-mono text-white/70 mb-1">
          <span>Max Network Load</span>
          <span className="text-white">{maxLoad.toFixed(1)} req/slot</span>
        </div>
        <input 
          type="range" min="0.1" max="2.0" step="0.1" 
          value={maxLoad} 
          onChange={(e) => setMaxLoad(parseFloat(e.target.value))}
          className="w-full accent-cyan-400"
        />
      </div>

      <div className="flex items-center justify-between mt-6">
        <span className="text-xs font-mono text-white/70">Mode</span>
        <button 
          onClick={() => setIsSimulation(!isSimulation)}
          className={`px-3 py-1 text-xs font-mono border transition-colors ${
            isSimulation 
              ? 'border-purple-500 text-purple-400 bg-purple-500/10' 
              : 'border-green-500 text-green-400 bg-green-500/10'
          }`}
        >
          {isSimulation ? 'SIMULATION' : 'API LIVE'}
        </button>
      </div>
    </div>
  );
};

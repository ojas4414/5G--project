/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { useState, useEffect } from 'react';
import { BACKEND_URL } from '@/lib/constants';

export interface SliceData {
  algorithm: 'C_ADMM' | 'MAAN' | 'Static_Greedy' | string;
  utilValue: number;
  activeSlices: number;
  qosSuccessRate: number;
  maxNetworkLoad: number;
  timestamp: number;
  history: number[];
}

export function useNetworkData() {
  const [data, setData] = useState<SliceData[]>([
    { algorithm: 'C_ADMM', utilValue: 0, activeSlices: 4, qosSuccessRate: 100, maxNetworkLoad: 1.0, timestamp: Date.now(), history: Array(10).fill(0) },
    { algorithm: 'MAAN', utilValue: 0, activeSlices: 4, qosSuccessRate: 100, maxNetworkLoad: 1.0, timestamp: Date.now(), history: Array(10).fill(0) },
    { algorithm: 'Static_Greedy', utilValue: 0, activeSlices: 4, qosSuccessRate: 100, maxNetworkLoad: 1.0, timestamp: Date.now(), history: Array(10).fill(0) }
  ]);
  const [status, setStatus] = useState<'connected' | 'disconnected'>('disconnected');
  const [lastUpdated, setLastUpdated] = useState<number>(Date.now());
  const [retryDelay, setRetryDelay] = useState<number>(2000);
  
  const [isSimulation, setIsSimulation] = useState(true);
  const [cAdmmSlices, setCAdmmSlices] = useState(4);
  const [maanLoad, setMaanLoad] = useState(1.0);

  const fetchTelemetry = async () => {
    if (isSimulation) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/results`);
      if (!res.ok) throw new Error('API down');
      const raw = await res.json();
      if (!Array.isArray(raw) && !raw.error) throw new Error('Invalid params');

      if (!raw.error) {
        const uniqueAlgos = Array.from(new Set(raw.map((item: any) => item.algorithm)));
        const latestData = uniqueAlgos.map((algo: any) => {
          const items = raw.filter((i: any) => i.algorithm === algo);
          return items.reduce((prev: any, curr: any) => {
            const prevLoad = prev.load_scale ?? prev.load ?? 0;
            const currLoad = curr.load_scale ?? curr.load ?? 0;
            return prevLoad > currLoad ? prev : curr;
          });
        });
        
        setData(prev => latestData.map((l: any) => {
          const prevAlgo = prev.find(p => p.algorithm === l.algorithm);
          const utilCandidate = l.mean_utility ?? l.utility_mean ?? l.utility_mean_mean ?? 0;
          const qosCandidate = l.qos_success_ratio ?? l.qos_success ?? l.qos_success_mean ?? 0;
          const newUtil = Math.max(0, utilCandidate);
          const newHistory = [...(prevAlgo?.history || []), newUtil].slice(-10);
          return {
            algorithm: l.algorithm,
            utilValue: newUtil,
            activeSlices: l.algorithm === 'C_ADMM' ? cAdmmSlices : 4,
            qosSuccessRate: qosCandidate * 100,
            maxNetworkLoad: l.algorithm === 'MAAN' ? maanLoad : 1.0,
            timestamp: Date.now(),
            history: newHistory
          };
        }));
      }
      
      setStatus('connected');
      setLastUpdated(Date.now());
      setRetryDelay(2000);
    } catch {
      setStatus('disconnected');
      setRetryDelay(prev => Math.min(prev * 2, 8000));
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isSimulation) {
      setStatus('connected');
      const getUtil = (time: number, speed: number, offset: number, amplitude: number) => {
        return Math.max(0, Math.min(100,
          50 + Math.sin(time * speed + offset) * amplitude
            + Math.sin(time * speed * 2.3 + offset * 1.7) * (amplitude * 0.3)
        ));
      };

      interval = setInterval(() => {
        const t = Date.now();
        const timeSec = t / 1000;
        setData(prev => {
          const cAdmmAmp = 10 + (cAdmmSlices * 3);
          const c_ut = getUtil(timeSec, 1.2, 0, cAdmmAmp);
          
          const mSpeed = 0.5 + maanLoad * 1.5;
          const m_ut = getUtil(timeSec, mSpeed, 2.5, 25);
          
          const s_ut = getUtil(timeSec, 0.8, 5.2, 10);
          
          return [
            { 
              algorithm: 'C_ADMM', utilValue: c_ut, activeSlices: cAdmmSlices, qosSuccessRate: 100, maxNetworkLoad: 1.0, timestamp: t,
              history: [...(prev.find(p => p.algorithm === 'C_ADMM')?.history || Array(9).fill(0)), c_ut].slice(-10)
            },
            { 
              algorithm: 'MAAN', utilValue: m_ut, activeSlices: 4, qosSuccessRate: 100, maxNetworkLoad: maanLoad, timestamp: t,
              history: [...(prev.find(p => p.algorithm === 'MAAN')?.history || Array(9).fill(0)), m_ut].slice(-10)
            },
            { 
              algorithm: 'Static_Greedy', utilValue: s_ut, activeSlices: 4, qosSuccessRate: 100, maxNetworkLoad: 1.0, timestamp: t,
              history: [...(prev.find(p => p.algorithm === 'Static_Greedy')?.history || Array(9).fill(0)), s_ut].slice(-10)
            }
          ];
        });
        setLastUpdated(t);
      }, 500);
    } else {
      const poll = async () => {
        await fetchTelemetry();
        interval = setTimeout(poll, status === 'connected' ? 500 : retryDelay);
      };
      interval = setTimeout(poll, 500);
    }

    return () => clearInterval(interval);
  }, [isSimulation, cAdmmSlices, maanLoad, status, retryDelay]);

  return { data, status, lastUpdated, reconnect: fetchTelemetry, isSimulation, setIsSimulation, cAdmmSlices, setCAdmmSlices, maanLoad, setMaanLoad };
}

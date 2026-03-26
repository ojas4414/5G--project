import React from 'react';
import { Activity, Server, Zap, BarChart2 } from 'lucide-react';

export default function DashboardOverlay({ data, loading, error }) {
  // Calculate some aggregate metrics from the dynamic data
  const totalAgents = data ? data.length : 0;
  const avgUtility = data && data.length > 0 
    ? (data.reduce((acc, curr) => acc + (curr.mean_utility || 0), 0) / data.length).toFixed(2)
    : '0.00';
    
  const highestLoad = data && data.length > 0
    ? Math.max(...data.map(d => d.load || 0)).toFixed(1)
    : '1.0';

  const avgSuccessRate = data && data.length > 0
    ? (data.reduce((acc, curr) => acc + (curr.qos_success_ratio || 0), 0) / data.length * 100).toFixed(1)
    : '0.0';

  return (
    <div className="dashboard-overlay">
      <header className="dashboard-header glass-panel">
        <div>
          <h1 className="dashboard-title">
            <span className="text-gradient">5G Network</span> Slicing Core
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.6)', marginTop: '8px' }}>
            Real-time Distributed Orchestration Telemetry
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: loading ? '#f5a623' : '#30d158' }}>
            <span style={{ 
              display: 'inline-block', 
              width: '10px', 
              height: '10px', 
              borderRadius: '50%', 
              backgroundColor: loading ? '#f5a623' : (error ? '#ff3b30' : '#30d158'),
              boxShadow: `0 0 10px ${loading ? '#f5a623' : (error ? '#ff3b30' : '#30d158')}`
            }}></span>
            <span style={{ fontWeight: 600, fontSize: '0.9rem', letterSpacing: '1px' }}>
              {loading ? 'SYNCING DATA...' : (error ? 'API DISCONNECTED' : 'API ONLINE')}
            </span>
          </div>
        </div>
      </header>

      {/* Aggregate Metrics */}
      <div className="metrics-grid">
        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">Active Slices</span>
            <Activity size={20} color="#0A84FF" />
          </div>
          <span className="metric-value">{totalAgents}</span>
        </div>

        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">Avg System Utility</span>
            <Zap size={20} color="#30d158" />
          </div>
          <span className="metric-value">{avgUtility}</span>
        </div>

        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">Max Network Load</span>
            <Server size={20} color="#ff3b30" />
          </div>
          <span className="metric-value">{highestLoad} <span style={{ fontSize: '1rem', color: '#aaa'}}>req/slot</span></span>
        </div>

        <div className="glass-panel metric-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="metric-label">QoS Success Rate</span>
            <BarChart2 size={20} color="#bf5af2" />
          </div>
          <span className="metric-value">{avgSuccessRate}%</span>
        </div>
      </div>
    </div>
  );
}

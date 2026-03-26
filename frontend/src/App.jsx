import React, { useState, useEffect } from 'react';
import Scene from './components/Scene';
import DashboardOverlay from './components/DashboardOverlay';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch dynamic data from python backend API
  useEffect(() => {
    const fetchBenchmarkData = async () => {
      try {
        setLoading(true);
        // Assuming fastapi runs on 8000
        const response = await fetch('http://localhost:8000/api/results');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const jsonData = await response.json();
        
        if (jsonData.error) {
          throw new Error(jsonData.error);
        }
        
        // Take unique algorithms for the 3D representation
        if (Array.isArray(jsonData)) {
          // Filter to just latest highest load or average it out.
          // For visual purposes, let's group by algorithm.
          const uniqueAlgos = Array.from(new Set(jsonData.map(item => item.algorithm)));
          
          const visualData = uniqueAlgos.map(algo => {
            const algoItems = jsonData.filter(item => item.algorithm === algo);
            // Get the item with highest load to represent the algo's peak performance
            return algoItems.reduce((prev, current) => 
              (prev.load > current.load) ? prev : current
            );
          });
          
          setData(visualData);
        }
        
        setError(null);
      } catch (err) {
        console.error('Failed to fetch dynamic data:', err);
        setError(err.message);
        // Don't set data to null, keep previous data if available
      } finally {
        setLoading(false);
      }
    };

    fetchBenchmarkData();
    // Re-fetch every 10 seconds to make it feel like alive telemetry
    const intervalId = setInterval(fetchBenchmarkData, 10000);
    
    return () => clearInterval(intervalId);
  }, []);

  return (
    <>
      <Scene data={data} />
      <DashboardOverlay data={data} loading={loading} error={error} />
    </>
  );
}

export default App;

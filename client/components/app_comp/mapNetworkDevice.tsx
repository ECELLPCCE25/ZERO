// CameraList.tsx
'use client'
import React, { useEffect, useState } from 'react';
import axios from 'axios';

interface CameraListProps {}

const CameraList: React.FC<CameraListProps> = () => {
  const [ipCameras, setIpCameras] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/scan_network`, {
      headers: {
        'Accept': 'application/json'
      }
    })
    .then((response) => {
      setIpCameras(response.data.ip_cameras);
      setLoading(false);
    })
    .catch((err) => {
      setError('Failed to fetch IP cameras');
      setLoading(false);
    });
  }, []);

  if (loading) return <p>Scanning network...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2>Discovered IP Cameras</h2>
      <ul className="list-disc pl-5">
        {ipCameras.map((ip, index) => (
          <li key={index}>{ip}</li>
        ))}
      </ul>
    </div>
  );
};

export default CameraList;

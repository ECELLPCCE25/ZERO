"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const CameraList: React.FC = () => {
  const [ipCameras, setIpCameras] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [idCounter, setIdCounter] = useState(0);
  const [connectedCameras, setConnectedCameras] = useState<{ ip: string, id: number, visible: boolean }[]>([]);

  useEffect(() => {
    axios
      .get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/scan_network`, {
        headers: {
          Accept: "application/json",
        },
      })
      .then((response) => {
        setIpCameras(response.data.ip_cameras);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to fetch IP cameras");
        setLoading(false);
      });
  }, []);

  const handleAddCamera = (ip: string) => {
    const existingCamera = connectedCameras.find((camera) => camera.ip === ip);
    if (existingCamera) {
      setConnectedCameras((prevCameras) =>
        prevCameras.map((camera) =>
          camera.ip === ip ? { ...camera, visible: true } : camera
        )
      );
    } else {
      const id = idCounter + Math.floor(Math.random() * 1000);
      setConnectedCameras((prevCameras) => [
        ...prevCameras,
        { ip, id, visible: true },
      ]);
      setIdCounter((prev) => prev + 1);
    }
  };

  if (loading) return <p>Scanning network...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Select IP Cameras to View</h2>

      <div className="mb-4">
        {ipCameras.map((ip) => (
          <div key={ip} className="flex items-center mb-2">
            <span className="mr-2">{ip}</span>
            <Button onClick={() => handleAddCamera(ip)}>Add</Button>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {connectedCameras.map(({ ip, id, visible }) => (
          <Card key={ip} className="p-2" style={{ display: visible ? 'block' : 'none' }}>
            <h3 className="text-sm font-semibold mb-2">{ip}</h3>
            <img
              src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${ip}&id=${id}`}
              alt={`Camera at ${ip}`}
              className="w-full h-48 object-cover rounded"
            />
          </Card>
        ))}
      </div>
    </div>
  );
};

export default CameraList;

"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Card } from "@/components/ui/card";

const CameraList: React.FC = () => {
  const [ipCameras, setIpCameras] = useState<string[]>([]);
  const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [idCounter, setIdCounter] = useState(0);
  const [connectedCameras, setConnectedCameras] = useState<{ ip: string, id: number }[]>([]);

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

  const handleSelectChange = (value: string) => {
    const selected = value.split(",");
    setSelectedCameras(selected);
    const updated = selected.map((ip) => {
      const id = idCounter + Math.floor(Math.random() * 1000);
      return { ip, id };
    });
    setConnectedCameras(updated);
    setIdCounter((prev) => prev + updated.length);
  };

  if (loading) return <p>Scanning network...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Select IP Cameras to View</h2>

      <Select onValueChange={handleSelectChange}>
        <SelectTrigger className="w-full max-w-md">
          <SelectValue placeholder="Choose cameras" />
        </SelectTrigger>
        <SelectContent>
          {ipCameras.map((ip) => (
            <SelectItem key={ip} value={ip}>{ip}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {connectedCameras.map(({ ip, id }) => (
          <Card key={ip} className="p-2">
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
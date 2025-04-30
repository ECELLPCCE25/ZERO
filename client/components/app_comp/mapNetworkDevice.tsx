"use client";
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { createSupabaseClient } from "@/lib/supabase";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import clsx from "clsx";

const CameraList: React.FC = () => {
  const [ipCameras, setIpCameras] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [idCounter, setIdCounter] = useState(0);
  const [connectedCameras, setConnectedCameras] = useState<
    { ip: string; id: number; visible: boolean; pinned: boolean; name: string }[]
  >([]);
  const [newIp, setNewIp] = useState("");
  const [fullScreenCameraId, setFullScreenCameraId] = useState<number | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [currentPersonCount, setCurrentPersonCount] = useState<{ [key: string]: number }>({});
  const [personCountHistory, setPersonCountHistory] = useState<{ [key: string]: { timestamp: string; count: number }[] }>({});
  const imgRefs = useRef<{ [key: string]: HTMLImageElement | null }>({});

  useEffect(() => {
    const supabase = createSupabaseClient();
    const fetchUser = async () => {
      const { data, error } = await supabase.auth.getUser();
      if (!error) setUserId(data?.user?.id || null);
    };
    fetchUser();
  }, []);

  useEffect(() => {
    const sockets: { [key: number]: WebSocket } = {};
    connectedCameras.forEach(({ ip, id }) => {
      const ws = new WebSocket(`ws://localhost:8000/ws/${id}`);
      sockets[id] = ws;

      ws.onopen = () => ws.send("request_person_count");

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setCurrentPersonCount((prev) => ({ ...prev, [id]: data.current_person_count }));
        if (Array.isArray(data.person_count_data)) {
          setPersonCountHistory((prev) => ({
            ...prev,
            [id]: data.person_count_data.map(({ timestamp, count }) => ({ timestamp, count })),
          }));
        }
      };

      ws.onerror = (err) => console.error(`WebSocket error for camera ID: ${id}`, err);
    });

    return () => Object.values(sockets).forEach((ws) => ws.close());
  }, [connectedCameras]);

  const scanNetwork = () => {
    setLoading(true);
    setError(null);
    axios
      .get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/scan_network`)
      .then((res) => {
        setIpCameras(res.data.ip_cameras);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to fetch IP cameras");
        setLoading(false);
      });
  };

  const handleAddCamera = (ip: string) => {
    let name = prompt("Enter a name for the camera:", "Camera " + (idCounter + 1)) || "Camera " + (idCounter + 1);
    const existing = connectedCameras.find((c) => c.ip === ip);
    if (existing) {
      setConnectedCameras((prev) => prev.map((c) => (c.ip === ip ? { ...c, visible: true } : c)));
    } else {
      const id = idCounter + Math.floor(Math.random() * 1000);
      setConnectedCameras((prev) => [...prev, { ip, id, visible: true, pinned: false, name }]);
      setIdCounter((prev) => prev + 1);
    }
  };

  const handleDiscardCamera = (ip: string) => {
    axios
      .post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/stop_stream`, { ip })
      .finally(() =>
        setConnectedCameras((prev) => prev.filter((camera) => camera.ip !== ip))
      );
  };

  const getGridColumns = () => "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";

  return (
    <div className="relative p-6 min-h-screen bg-gradient-to-br from-slate-100 via-blue-100 to-purple-200 overflow-hidden">
      {loading && (
        <div className="fixed inset-0 flex flex-col items-center justify-center z-50 bg-white/30 backdrop-blur-sm">
          <DotLottieReact
            src="https://lottie.host/df70e3d3-df17-44e2-9207-5d60ffd56867/T1SfRUVnwV.lottie"
            loop
            autoplay
            style={{ width: 240, height: 240 }}
          />
        </div>
      )}

      <h2 className="text-3xl font-extrabold mb-8 text-gray-800 drop-shadow">
        🎥 Select IP Cameras to View
      </h2>

      <div className="mb-6">
        <Button
          onClick={scanNetwork}
          className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg hover:brightness-110"
        >
          🔍 Scan Network
        </Button>
      </div>

      {error && <p className="text-red-500 font-semibold mb-4">{error}</p>}

      <div className="space-y-3 mb-6">
        {ipCameras.map((ip) => (
          <div key={ip} className="bg-white/80 backdrop-blur-md p-3 rounded shadow flex justify-between">
            <span className="text-sm font-medium text-gray-700">{ip}</span>
            <Button onClick={() => handleAddCamera(ip)} className="bg-green-500 hover:bg-green-600 text-white">
              ➕ Add
            </Button>
          </div>
        ))}
      </div>

      <div className="flex mb-6">
        <input
          type="text"
          value={newIp}
          onChange={(e) => setNewIp(e.target.value)}
          placeholder="Enter new IP"
          className="flex-1 p-2 rounded-l border border-gray-300 shadow-inner"
        />
        <Button
          onClick={() => {
            if (newIp) handleAddCamera(newIp.trim());
            setNewIp("");
          }}
          className="rounded-l-none bg-blue-600 hover:bg-blue-700 text-white"
        >
          Submit
        </Button>
      </div>

      <div className={`grid gap-6 ${getGridColumns()}`}>
        {connectedCameras.map(({ ip, id, visible, pinned, name }) => (
          <Card
            key={ip}
            className={clsx(
              "relative p-4 bg-white/80 rounded-xl shadow-lg transition-all",
              { hidden: !visible && !pinned }
            )}
          >
            <input
              value={name}
              onChange={(e) =>
                setConnectedCameras((prev) =>
                  prev.map((c) => (c.ip === ip ? { ...c, name: e.target.value } : c))
                )
              }
              className="w-full text-sm p-2 mb-3 border border-gray-200 rounded"
            />
            <img
              ref={(el) => (imgRefs.current[id] = el)}
              src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${ip}&id=${id}&user_id=${userId}&name=${name}`}
              className="w-full h-48 object-cover rounded"
            />
            <p className="mt-2 text-gray-700 text-sm font-semibold">
              👥 Person Count: {currentPersonCount[id] || 0}
            </p>
            <div className="absolute top-2 right-2 flex gap-2">
              <Button size="sm" onClick={() => setFullScreenCameraId(id)} className="bg-yellow-500 text-white">
                📌 Pin
              </Button>
              <Button size="sm" onClick={() => handleDiscardCamera(ip)} className="bg-red-500 text-white">
                ❌ Discard
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {fullScreenCameraId !== null && (
        <div className="fixed inset-0 bg-black z-50 flex items-center justify-center">
          <Button
            onClick={() => setFullScreenCameraId(null)}
            className="absolute top-5 right-5 bg-white text-black"
          >
            🔙 Unpin
          </Button>
          <p className="absolute top-5 left-5 text-white text-xl">
            👥 Count: {currentPersonCount[fullScreenCameraId] || 0}
          </p>
        </div>
      )}
    </div>
  );
};

export default CameraList;

"use client";
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { createSupabaseClient } from "@/lib/supabase";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import clsx from "clsx";
import { toast } from "sonner";
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
    const supabase = createSupabaseClient();

    const channel = supabase.channel('camera_updates');

    channel.on(
        'broadcast',
        { event: 'UPDATE' },
        (payload) => {
            console.log('Broadcast message received:', payload);
            // Ensure the user ID matches, and the broadcast contains the necessary information
            if (payload.user_id === userId) {
                const { stream_id, person_count, message } = payload;

                // Handle the alert message and log it or show it as a toast
                console.log(`Alert received: ${message} (Person count: ${person_count})`);
                toast.success(`${message} (Person count: ${person_count})`);

                // Optionally, you can refetch the camera list or update specific state variables
                scanNetwork();
            }
        }
    ).subscribe();

    return () => {
        supabase.removeChannel(channel);
    };
}, [userId]);




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
          setPersonCountHistory((prev: any) => ({
            ...prev,
            [id]: data.person_count_data.map(({ timestamp, count }: { timestamp: any, count: any }) => ({
              timestamp,
              count,
            })),
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
    <div className="relative p-6 min-h-screen bg-gray-100 text-gray-800">
      {loading && (
        <div className="fixed inset-0 flex flex-col items-center justify-center z-50 bg-white/50 backdrop-blur-sm">
          <DotLottieReact
            src="https://lottie.host/df70e3d3-df17-44e2-9207-5d60ffd56867/T1SfRUVnwV.lottie"
            loop
            autoplay
            style={{ width: 200, height: 200 }}
          />
        </div>
      )}

      <h2 className="text-2xl font-semibold mb-6">📷 IP Camera Dashboard</h2>

      <div className="mb-4">
        <Button onClick={scanNetwork} className="bg-gray-800 hover:bg-gray-700 text-white">
          🔍 Scan Network
        </Button>
      </div>

      {error && <p className="text-red-500 font-medium mb-4">{error}</p>}

      <div className="space-y-2 mb-6">
        {ipCameras.map((ip) => (
          <div key={ip} className="bg-white border border-gray-200 p-3 rounded shadow-sm flex justify-between items-center">
            <span className="text-sm text-gray-700">{ip}</span>
            <Button onClick={() => handleAddCamera(ip)} className="bg-gray-600 hover:bg-gray-700 text-white">
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
          className="rounded-l-none bg-gray-800 hover:bg-gray-700 text-white"
        >
          Submit
        </Button>
      </div>

      <div className={`grid gap-6 ${getGridColumns()}`}>
        {connectedCameras.map(({ ip, id, visible, pinned, name }) => (
          <Card
            key={ip}
            className={clsx(
              "relative p-4 bg-white border border-gray-300 rounded-lg shadow-sm space-y-3",
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
              className="w-full text-sm p-2 border border-gray-200 rounded"
            />

            <img
              ref={(el) => (imgRefs.current[id] = el)}
              src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${ip}&id=${id}&user_id=${userId}&name=${name}`}
              className="w-full h-48 object-cover rounded border border-gray-200"
            />

            <div className="grid grid-cols-2 gap-2 mt-2">
              <Card className="border border-gray-200 p-2 text-center rounded">
                <p className="text-xs text-gray-500">Current</p>
                <p className="font-semibold text-sm">{currentPersonCount[id] || 0}</p>
              </Card>
              <Card className="border border-gray-200 p-2 text-center rounded">
                <p className="text-xs text-gray-500">Average</p>
                <p className="font-semibold text-sm">
                  {(() => {
                    const history = personCountHistory[id];
                    if (!history || history.length === 0) return "-";
                    const avg = (
                      history.reduce((acc, item) => acc + item.count, 0) / history.length
                    ).toFixed(1);
                    return avg;
                  })()}
                </p>
              </Card>
            </div>

            <div className="absolute top-2 right-2 flex gap-2">
              {/* <Button size="sm" onClick={() => setFullScreenCameraId(id)} className="bg-gray-500 text-white">
                📌 Pin
              </Button> */}
              <Button size="sm" onClick={() => handleDiscardCamera(ip)} className="bg-red-500 text-white">
                ❌ Discard
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {fullScreenCameraId !== null && (
        <div className="fixed inset-0 bg-white z-50 flex flex-col items-center justify-center p-6">
          <Button
            onClick={() => setFullScreenCameraId(null)}
            className="absolute top-5 right-5 bg-gray-800 text-white"
          >
            🔙 Unpin
          </Button>
          <p className="absolute top-5 left-5 text-gray-800 text-xl">
            👥 Count: {currentPersonCount[fullScreenCameraId] || 0}
          </p>
        </div>
      )}
    </div>
  );
};

export default CameraList;

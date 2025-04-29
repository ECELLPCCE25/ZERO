"use client";
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { createSupabaseClient } from "@/lib/supabase";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";

// Helper to generate random names
const generateRandomName = () => {
  const adjectives = ["Swift", "Silent", "Clever", "Sharp", "Lone", "Brave"];
  const animals = ["Falcon", "Tiger", "Wolf", "Panther", "Hawk", "Eagle"];
  const adjective = adjectives[Math.floor(Math.random() * adjectives.length)];
  const animal = animals[Math.floor(Math.random() * animals.length)];
  return `${adjective} ${animal}`;
};

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
      if (error) {
        console.error("Error fetching user:", error);
        return;
      }
      setUserId(data?.user?.id || null);
    };
    fetchUser();
  }, []);

  useEffect(() => {
    if (userId) {
      console.log("Authenticated User ID:", userId);
    }
  }, [userId]);

  useEffect(() => {
    const sockets: { [key: number]: WebSocket } = {};

    connectedCameras.forEach(({ ip, id }) => {
      const ws = new WebSocket(`ws://localhost:8000/ws/${id}`);
      sockets[id] = ws;

      ws.onopen = () => {
        console.log(`WebSocket connection opened for camera ID: ${id}`);
        ws.send("request_person_count");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const currentCount = data.current_person_count;
        const personCountArray = data.person_count_data;

        console.log(`Received data for camera ID: ${id}`, data);

        // Update current person count
        setCurrentPersonCount((prevCount) => ({
          ...prevCount,
          [id]: currentCount,
        }));

        // Update person count history
        if (Array.isArray(personCountArray)) {
          setPersonCountHistory((prevData) => ({
            ...prevData,
            [id]: personCountArray.map(({ timestamp, count }) => ({
              timestamp,
              count,
            })),
          }));
        }
      };

      ws.onclose = () => {
        console.log(`WebSocket connection closed for camera ID: ${id}`);
      };

      ws.onerror = (err) => {
        console.error(`WebSocket error for camera ID: ${id}`, err);
      };
    });

    return () => {
      Object.values(sockets).forEach((ws) => {
        ws.close();
      });
    };
  }, [connectedCameras]);

  const scanNetwork = () => {
    setLoading(true);
    setError(null);
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
  };

  const handleAddCamera = async (ip: string) => {
    let name = prompt("Enter a name for the camera:", generateRandomName());
    if (!name) {
      name = generateRandomName();
    }

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
        {
          ip,
          id,
          visible: true,
          pinned: false,
          name,
        },
      ]);
      setIdCounter((prev) => prev + 1);
    }
  };

  const handleSubmitNewIp = () => {
    if (newIp.trim()) {
      handleAddCamera(newIp.trim());
      setNewIp("");
    }
  };

  const handlePinCamera = (id: number) => {
    setFullScreenCameraId(id);
  };

  const handleUnpinCamera = () => {
    setFullScreenCameraId(null);
  };

  const handleDiscardCamera = (ip: string) => {
    axios
      .post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/stop_stream`, { ip })
      .then((response) => {
        if (response.status === 200) {
          setConnectedCameras((prevCameras) =>
            prevCameras.filter((camera) => camera.ip !== ip)
          );
        }
      })
      .catch((error) => {
        console.error("Failed to stop the stream:", error);
        setConnectedCameras((prevCameras) =>
          prevCameras.filter((camera) => camera.ip !== ip)
        );
      });
  };

  const getGridColumns = () => {
    const visibleCameras = connectedCameras.filter((camera) => camera.visible && !camera.pinned);
    return "grid-cols-3"; // Always use a 3x3 grid
  };

  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Select IP Cameras to View</h2>

      {!loading && (
        <div className="mb-4">
          <Button onClick={scanNetwork}>Scan Network</Button>
        </div>
      )}

      {loading && (
        <div className="fixed inset-0 flex items-center justify-center bg-white/80 z-50">
          <DotLottieReact
            src="https://lottie.host/1f517d86-fafe-4b55-ae27-536294b1472b/6iVFtyWYB4.lottie"
            loop
            autoplay
            style={{ width: 200, height: 200 }}
          />
        </div>
      )}

      {error && <p className="text-red-500 mb-4">{error}</p>}

      <div className="mb-4">
        {ipCameras.map((ip) => (
          <div key={ip} className="flex items-center mb-2">
            <span className="mr-2">{ip}</span>
            <Button onClick={() => handleAddCamera(ip)}>Add</Button>
          </div>
        ))}
      </div>

      <div className="mb-4">
        <input
          type="text"
          value={newIp}
          onChange={(e) => setNewIp(e.target.value)}
          placeholder="Enter new IP address"
          className="border p-2 mr-2"
        />
        <Button onClick={handleSubmitNewIp}>Submit</Button>
      </div>

      <div className={`grid ${getGridColumns()} gap-4`}>
        {connectedCameras.map(({ ip, id, visible, pinned, name }) => (
          <Card
            key={ip}
            className="p-2 relative"
            style={{ display: visible || pinned ? "block" : "none" }}
            onMouseEnter={(e) => {
              const card = e.currentTarget;
              const options = card.querySelector(".options") as HTMLElement;
              if (options) options.style.display = "flex";
            }}
            onMouseLeave={(e) => {
              const card = e.currentTarget;
              const options = card.querySelector(".options") as HTMLElement;
              if (options) options.style.display = "none";
            }}
          >
            <input
              type="text"
              value={name}
              onChange={(e) =>
                setConnectedCameras((prev) =>
                  prev.map((c) =>
                    c.ip === ip ? { ...c, name: e.target.value } : c
                  )
                )
              }
              className="text-sm font-semibold mb-2 w-full p-1 border rounded"
              placeholder="Camera name"
            />
            <img
              ref={(el) => (imgRefs.current[id] = el)}
              src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${ip}&id=${id}&user_id=${userId}&name=${name}`}
              alt={`Camera at ${ip}`}
              className={`w-full h-48 object-cover rounded ${fullScreenCameraId === id ? "fullscreen" : ""}`}
            />
            <p className="text-sm mt-2">
              Person Count: {currentPersonCount[id] || 0}
            </p>
            <div
              className="options absolute top-2 right-2 flex space-x-2"
              style={{ display: "none" }}
            >
              <Button onClick={() => handlePinCamera(id)}>Pin</Button>
              <Button onClick={() => handleDiscardCamera(ip)}>Discard</Button>
            </div>
          </Card>
        ))}
      </div>

      {fullScreenCameraId !== null && (
        <div className="fixed inset-0 bg-black flex items-center justify-center z-50">
          <Button
            onClick={handleUnpinCamera}
            className="absolute top-4 right-4 z-50"
          >
            Unpin
          </Button>
          <p className="absolute top-4 left-4 text-white z-50">
            Person Count: {currentPersonCount[fullScreenCameraId] || 0}
          </p>
        </div>
      )}
    </div>
  );
};

export default CameraList;
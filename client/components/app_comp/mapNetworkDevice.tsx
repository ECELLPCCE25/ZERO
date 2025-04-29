"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { createSupabaseClient } from "@/lib/supabase";
import { DotLottieReact } from '@lottiefiles/dotlottie-react';

const CameraList: React.FC = () => {
  const [ipCameras, setIpCameras] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [idCounter, setIdCounter] = useState(0);
  const [connectedCameras, setConnectedCameras] = useState<
    { ip: string; id: number; visible: boolean; pinned: boolean }[]
  >([]);
  const [newIp, setNewIp] = useState("");
  const [fullScreenCamera, setFullScreenCamera] = useState<{
    ip: string;
    id: number;
  } | null>(null);
  const [userId, setUserId] = useState<string | null>(null);

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
      // You can use this ID to fetch/store user-specific data.
    }
  }, [userId]);

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
        { ip, id, visible: true, pinned: false },
      ]);
      setIdCounter((prev) => prev + 1);
    }
  };

  const handleSubmitNewIp = () => {
    if (newIp.trim()) {
      handleAddCamera(newIp.trim());
      setNewIp(""); // Clear the input field after submission
    }
  };

  const handlePinCamera = (ip: string, id: number) => {
    setConnectedCameras((prevCameras) =>
      prevCameras.map((camera) =>
        camera.ip === ip ? { ...camera, pinned: true } : camera
      )
    );
    setFullScreenCamera({ ip, id });
  };

  const handleUnpinCamera = () => {
    setConnectedCameras((prevCameras) =>
      prevCameras.map((camera) =>
        camera.ip === fullScreenCamera?.ip
          ? { ...camera, pinned: false }
          : camera
      )
    );
    setFullScreenCamera(null);
  };

  const handleDiscardCamera = (ip: string) => {
    // Send a request to the backend to stop the stream
    axios
      .post(`${process.env.NEXT_PUBLIC_BACKEND_URL}/stop_stream`, { ip })
      .then(() => {
        setConnectedCameras((prevCameras) =>
          prevCameras.map((camera) =>
            camera.ip === ip ? { ...camera, visible: false } : camera
          )
        );
      })
      .catch((error) => {
        console.error("Failed to stop the stream:", error);
      });
  };

  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Select IP Cameras to View</h2>

      <div className="mb-4">
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

      </div>

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

      {fullScreenCamera ? (
        <div className="fixed inset-0 bg-black flex items-center justify-center z-50">
          <div className="relative w-full h-full">
            <img
              src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${fullScreenCamera.ip}&id=${fullScreenCamera.id}&user_id=${userId}`}
              alt={`Camera at ${fullScreenCamera.ip}`}
              className="w-full h-full object-contain"
            />
            <Button
              onClick={handleUnpinCamera}
              className="absolute top-4 right-4"
            >
              Unpin
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {connectedCameras.map(({ ip, id, visible, pinned }) => (
            <Card
              key={ip}
              className="p-2 relative"
              style={{ display: visible || pinned ? "block" : "none" }}
              onMouseEnter={(e) => {
                const card = e.currentTarget;
                const options = card.querySelector(".options") as HTMLElement;
                if (options) {
                  options.style.display = "flex";
                }
              }}
              onMouseLeave={(e) => {
                const card = e.currentTarget;
                const options = card.querySelector(".options") as HTMLElement;
                if (options) {
                  options.style.display = "none";
                }
              }}
            >
              <h3 className="text-sm font-semibold mb-2">{ip}</h3>
              <img
                src={`${process.env.NEXT_PUBLIC_BACKEND_URL}/ipcam?device=${ip}&id=${id}`}
                alt={`Camera at ${ip}`}
                className="w-full h-48 object-cover rounded"
              />
              <div
                className="options absolute top-2 right-2 flex space-x-2"
                style={{ display: "none" }}
              >
                <Button onClick={() => handlePinCamera(ip, id)}>Pin</Button>
                <Button onClick={() => handleDiscardCamera(ip)}>Discard</Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default CameraList;

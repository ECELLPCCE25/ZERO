const cameras = [
    { id: "NW-02" },
    { id: "NS-08" },
    { id: "NS-04" },
    { id: "WE-02", isActive: true },
  ];
  
  export default function CameraGrid() {
    return (
      <>
        {cameras.map((cam) => (
          <div
            key={cam.id}
            className={`relative rounded overflow-hidden ${
              cam.isActive ? "border-2 border-red-500" : ""
            }`}
          >
            <img
              src="/sample.jpg"
              alt={`Camera ${cam.id}`}
              className="w-full h-40 object-cover"
            />
            <div className="absolute top-1 left-1 text-white">📷</div>
            <div className="absolute bottom-2 left-2 right-2 flex justify-between bg-gray-400 bg-opacity-70 px-2 py-1 rounded">
              <button>📍</button>
              <button>❌</button>
            </div>
            <div className="absolute bottom-1 right-2 text-white text-sm">
              {cam.id}
            </div>
          </div>
        ))}
      </>
    );
  }
  
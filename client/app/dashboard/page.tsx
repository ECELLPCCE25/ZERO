import CameraGrid from "@/components/app_comp/CameraGrid";
import ChartSection from "@/components/app_comp/ChartSection";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-100 p-4 space-y-4">
      {/* Camera Header */}
      <div className="flex items-center gap-2">
        <select className="p-2 border rounded bg-white shadow">
          <option>Add New Camera</option>
        </select>
        <button className="p-2 bg-gray-300 rounded">+</button>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-3 gap-4">
        {/* Camera Feeds (2 cols) */}
        <div className="col-span-2 grid grid-cols-2 gap-4">
          <CameraGrid />
        </div>

        {/* Large right gray box */}
        <div className="bg-gray-300 h-[320px] rounded" />
      </div>

      {/* Chart section */}
      <ChartSection />
    </main>
  );
}

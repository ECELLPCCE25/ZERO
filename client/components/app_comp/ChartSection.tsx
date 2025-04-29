
export default function ChartSection() {
    return (
      <div className="bg-gray-300 h-[200px] rounded mt-4 overflow-hidden">
        {/* Placeholder chart line */}
        <svg viewBox="0 0 100 20" className="w-full h-full">
          <polyline
            fill="none"
            stroke="black"
            strokeWidth="1"
            points="0,10 10,5 20,15 30,10 40,5 50,7 60,3 70,15 80,18 90,10 100,10"
          />
        </svg>
      </div>
    );
  }
  
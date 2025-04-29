import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface DataPoint {
  timestamp: string;
  count: number;
}

interface GraphProps {
  data: DataPoint[];
}

const Graph: React.FC<GraphProps> = ({ data }) => {
  const PERSON_COUNT_THRESHOLD = parseInt(process.env.NEXT_PUBLIC_PERSON_COUNT_THRESHOLD || "5", 10);
  const ROLLING_WINDOW = 5;

  const calculateRollingAverage = (data: DataPoint[]): (DataPoint & { rollingAvg: number })[] => {
    return data.map((point, index) => {
      const start = Math.max(0, index - ROLLING_WINDOW + 1);
      const window = data.slice(start, index + 1);
      const avgCount = window.reduce((sum, p) => sum + p.count, 0) / window.length;
      return {
        ...point,
        rollingAvg: parseFloat(avgCount.toFixed(2)),
      };
    });
  };

  const formattedData = calculateRollingAverage(data).map((point) => ({
    ...point,
    time: new Date(point.timestamp).toLocaleTimeString(),
  }));

  const thresholdViolations = data
    .filter((point) => point.count > PERSON_COUNT_THRESHOLD)
    .map((point) => ({
      time: new Date(point.timestamp).toLocaleString(),
      count: point.count,
    }));

  return (
    <div className="w-full p-4 bg-gray-100 dark:bg-gray-900 min-h-screen">
      {/* Chart Card */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          Person Count Over Time
        </h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={formattedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
              <XAxis dataKey="time" tick={{ fill: "#666" }} />
              <YAxis allowDecimals={false} tick={{ fill: "#666" }} />
              <Tooltip contentStyle={{ backgroundColor: "#f9f9f9", borderColor: "#ccc" }} />
              <Legend />
              <Line
                type="monotone"
                dataKey="count"
                name="Person Count"
                stroke="#4b5563"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="rollingAvg"
                name={`Rolling Avg (${ROLLING_WINDOW} points)`}
                stroke="#9ca3af"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Threshold Violations Card */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
          Threshold Violations (Count &gt; {PERSON_COUNT_THRESHOLD})
        </h3>
        {thresholdViolations.length > 0 ? (
          <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700 dark:text-gray-300">
            {thresholdViolations.map((violation, index) => (
              <li key={index}>
                <span className="font-medium">{violation.time}</span>: {violation.count} people
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No threshold violations detected.
          </p>
        )}
      </div>
    </div>
  );
};

export default Graph;

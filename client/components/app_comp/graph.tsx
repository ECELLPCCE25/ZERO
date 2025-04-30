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
  const PERSON_COUNT_THRESHOLD = parseInt(
    process.env.NEXT_PUBLIC_PERSON_THRESHOLD ?? "10",
    10
  );
  const ROLLING_WINDOW = 5;

  if (!data || data.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500 dark:text-gray-400">
        No data available to display.
      </div>
    );
  }

  const calculateRollingAverage = (
    data: DataPoint[]
  ): (DataPoint & { rollingAvg: number })[] => {
    return data.map((point, index) => {
      const start = Math.max(0, index - ROLLING_WINDOW + 1);
      const window = data.slice(start, index + 1);
      const avgCount = window.reduce((sum, p) => sum + p.count, 0) / window.length;
      return {
        ...point,
        rollingAvg: Math.round(avgCount),
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

  const totalCount = data.reduce((sum, point) => sum + point.count, 0);
  const averageCount = Math.round(totalCount / data.length);

  const peak = data.reduce(
    (max, point) => (point.count > max.count ? point : max),
    data[0]
  );

  const totalViolations = thresholdViolations.length;
  const violationPercentage = Math.round((totalViolations / data.length) * 100);

  return (
    <div className="w-full p-4 bg-gray-100 dark:bg-gray-900 min-h-screen">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Chart Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6 col-span-1 md:col-span-2 xl:col-span-3">
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

        {/* Threshold Violations */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6 overflow-auto">
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

        {/* Individual Stat Cards */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Peak Count
          </h3>
          <p className="text-gray-700 dark:text-gray-300">
            {peak.count} people at {new Date(peak.timestamp).toLocaleString()}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Average Count
          </h3>
          <p className="text-gray-700 dark:text-gray-300">{averageCount} people</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Total Violations
          </h3>
          <p className="text-gray-700 dark:text-gray-300">{totalViolations}</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Time Above Threshold
          </h3>
          <p className="text-gray-700 dark:text-gray-300">{violationPercentage}%</p>
        </div>
      </div>
    </div>
  );
};

export default Graph;

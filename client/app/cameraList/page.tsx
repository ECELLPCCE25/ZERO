'use client'
import React, { useEffect, useState } from 'react'
import axios from 'axios'
import Graph from '@/components/app_comp/graph'
import CameraList from '@/components/app_comp/mapNetworkDevice'

interface DataPoint {
  timestamp: string
  count: number
}

function Page() {
  const [graphData, setGraphData] = useState<DataPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Replace this with your actual endpoint
        const response = await axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/get_person_count_by_stream_id?user_id=your_user_id&stream_id=248`)
        setGraphData(response.data)
      } catch (error) {
        console.error('Failed to fetch graph data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  return (
    <>
      <CameraList />
      {loading ? (
        <p className="p-4">Loading graph data...</p>
      ) : (
        <Graph data={graphData} />
      )}
    </>
  )
}

export default Page

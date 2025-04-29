'use client'
import React, { useEffect, useState } from 'react'
import axios from 'axios'
import Graph from '@/components/app_comp/graph'
import CameraList from '@/components/app_comp/mapNetworkDevice'
import { createSupabaseClient } from '@/lib/supabase'

interface DataPoint {
  timestamp: string
  count: number
}

function Page() {
  const [graphData, setGraphData] = useState<DataPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [user_id, setUser_id] = useState<string>("")

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Replace this with your actual endpoint
        const response = await axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/get_person_count_by_stream_id?user_id=${user_id}&stream_id=248`)
        setGraphData(response.data)
      } catch (error) {
        console.error('Failed to fetch graph data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  useEffect(() => {
    const supabase = createSupabaseClient();
    const fetchUser = async () => {
      const { data, error } = await supabase.auth.getUser();
      if (error) {
        console.error("Error fetching user:", error);
        return;
      }
      setUser_id(data?.user?.id);
    };
    fetchUser();
  }, []);
  useEffect(() => {
    if (user_id) {
      console.log("Authenticated User ID:", user_id);
      // You can use this ID to fetch/store user-specific data.
    }
  }, [user_id]);

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

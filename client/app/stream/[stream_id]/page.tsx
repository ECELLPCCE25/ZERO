'use client'
import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Graph from '@/components/app_comp/graph'
import { createSupabaseClient } from '@/lib/supabase'
import axios from 'axios'

interface DataPoint {
    timestamp: string
    count: number
  }

export default function page() {
    const {stream_id}:{stream_id:string} = useParams()

    const [graphData, setGraphData] = useState<DataPoint[]>([])
    const [loading, setLoading] = useState(true)

    const [user_id, setUser_id] = useState<string>("")

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
    const fetchData = async () => {
      if (!user_id) return; // Ensure user_id is set before making the API call
      try {
        const response = await axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/get_person_count_by_stream_id`, {
          params: { user_id, stream_id: stream_id }
        });
        setGraphData(response.data);
      } catch (error) {
        console.error('Failed to fetch graph data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [user_id]); // Add user_id as a dependency

  return (
    <>
    <p>{stream_id}</p>
    <Graph data={graphData}/>
    </>
  )
}

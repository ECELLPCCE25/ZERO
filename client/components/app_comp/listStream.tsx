'use client'

import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { createSupabaseClient } from '@/lib/supabase'
import Link from 'next/link'

interface Stream {
  stream_id: string
  cam_name:string
}

const StreamList: React.FC = () => {
  const [streams, setStreams] = useState<Stream[]>([])
  const [loading, setLoading] = useState(true)
  const [userId, setUserId] = useState<string | null>(null)

  useEffect(() => {
    const supabase = createSupabaseClient()
    const fetchUser = async () => {
      const { data, error } = await supabase.auth.getUser()
      if (error || !data?.user?.id) {
        console.error("Error fetching user:", error)
        setLoading(false)
        return
      }
      setUserId(data.user.id)
    }

    fetchUser()
  }, [])

  useEffect(() => {
    if (!userId) return

    const fetchStreams = async () => {
      try {
        const response = await axios.get<Stream[]>(
          'http://127.0.0.1:8000/get_streams_by_user_id',
          {
            params: { user_id: userId },
            headers: { accept: 'application/json' },
          }
        )
        setStreams(response.data)
      } catch (error) {
        console.error('Failed to fetch streams:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStreams()
  }, [userId]) // <== Reacts to userId being set

  if (loading) return <p className="p-4">Loading streams...</p>

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Stream List</h2>
      <ul className='flex flex-col gap-2'>
        {streams.map((stream, index) => (
          <Link className='border rounded-lg' href={`/stream/${stream.stream_id}`} key={index}>Stream ID: {stream.stream_id}
          <p>{stream.cam_name}</p>
          </Link>
        ))}
      </ul>
    </div>
  )
}

export default StreamList

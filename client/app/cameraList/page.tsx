'use client'
import React, { useEffect, useState } from 'react'
import CameraList from '@/components/app_comp/mapNetworkDevice'
import StreamList from '@/components/app_comp/listStream'


function Page() {

  return (
    <>
      <CameraList />
      <StreamList/>
    </>
  );
}

export default Page;

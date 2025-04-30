'use client'
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { toast } from "sonner";


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

    const router = useRouter();
    const [alert, setAlert] = useState<{
      message: string;
      progress: number;
      type: "REFRESH" | string;
    }>({
      message: "",
      progress: 0,
      type: "",
    });
  
    useEffect(() => {
      const eventSource = new EventSource(process.env.NEXT_PUBLIC_BACKEND_URL + "/events/");
      eventSource.onmessage = (event) => {
        const newAlert = JSON.parse(event.data);
        setAlert(newAlert);
toast(JSON.stringify(newAlert))        }
      
      return () => {
        eventSource.close();
      };
    }, []);
  return (
   <div>
        {children}
        </div>
  );
}


'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';


export default function Home() {
  
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background to-muted p-4">
      <h1 className="text-4xl font-bold mb-8 text-center">Welcome to Our App</h1>
      <div className="space-x-4">
      
            <Link href="/sign-in">
              <Button size="lg">Sign In</Button>
            </Link>
            <Link href="/sign-up">
              <Button size="lg" variant="outline">Sign Up</Button>
            </Link>
          
      </div>
    </div>
  );
}

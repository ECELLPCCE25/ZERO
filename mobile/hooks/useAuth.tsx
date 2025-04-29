import { useRouter, useSegments } from 'expo-router';
import { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from './useSupabase';
import React from 'react';
import { AuthError, AuthResponse, Session, User } from '@supabase/supabase-js';

// Define our auth context types
interface AuthContextProps {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  signIn: (email: string, password: string) => Promise<AuthResponse>;
  signUp: (email: string, password: string) => Promise<AuthResponse>;
  signOut: () => Promise<void>;
  error: AuthError | null;
}

// Create the context
const AuthContext = createContext<AuthContextProps>({
  user: null,
  session: null,
  isLoading: true,
  signIn: () => Promise.resolve({} as AuthResponse),
  signUp: () => Promise.resolve({} as AuthResponse),
  signOut: () => Promise.resolve(),
  error: null,
});

// Provider component to wrap our app with
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<AuthError | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const segments = useSegments();
  const router = useRouter();

  // Check auth state on mount
  useEffect(() => {
    // Get the current session
    const getSession = async () => {
      setIsLoading(true);
      const { data, error } = await supabase.auth.getSession();
      
      if (error) {
        setError(error);
        setIsLoading(false);
        return;
      }
      
      if (data?.session) {
        setSession(data.session);
        setUser(data.session.user);
      }
      
      setIsLoading(false);
    };

    getSession();

    // Subscribe to auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (session) {
        setSession(session);
        setUser(session.user);
      } else {
        setSession(null);
        setUser(null);
      }
      setIsLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Handle routing based on auth state
  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === '(auth)';

    if (!session && !inAuthGroup) {
      // If not signed in and not on an auth page, redirect to auth
      router.replace('/(auth)/login');
    } else if (session && inAuthGroup) {
      // If signed in and on an auth page, redirect to home
      router.replace('/(tabs)');
    }
  }, [session, segments, isLoading]);

  // Auth functions
  const signIn = async (email: string, password: string) => {
    setError(null);
    const response = await supabase.auth.signInWithPassword({ email, password });
    if (response.error) {
      setError(response.error);
    }
    return response;
  };

  const signUp = async (email: string, password: string) => {
    setError(null);
    const response = await supabase.auth.signUp({ email, password });
    if (response.error) {
      setError(response.error);
    }
    return response;
  };

  const signOut = async () => {
    setError(null);
    const { error } = await supabase.auth.signOut();
    if (error) {
      setError(error);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        session,
        isLoading,
        signIn,
        signUp,
        signOut,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use the auth context
export function useAuth() {
  return useContext(AuthContext);
}
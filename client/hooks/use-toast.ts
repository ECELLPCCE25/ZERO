'use client';

import { toast as sonnerToast } from 'sonner';

export function useToast() {
  return {
    toast: {
      success: (message: string) => sonnerToast.success(message),
      error: (message: string) => sonnerToast.error(message),
      info: (message: string) => sonnerToast.info(message),
      warning: (message: string) => sonnerToast.warning(message),
      default: (message: string) => sonnerToast(message),
    },
  };
}

export { useToast as toast };
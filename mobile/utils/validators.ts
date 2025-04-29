// Basic validation functions for form fields

// Email validation
export function validateEmail(email: string): string | null {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!email) return 'Email is required';
  if (!emailRegex.test(email)) return 'Please enter a valid email address';
  return null;
}

// Password validation
export function validatePassword(password: string): string | null {
  if (!password) return 'Password is required';
  if (password.length < 6) return 'Password must be at least 6 characters';
  return null;
}

// Confirm password validation
export function validateConfirmPassword(password: string, confirmPassword: string): string | null {
  if (!confirmPassword) return 'Please confirm your password';
  if (password !== confirmPassword) return 'Passwords do not match';
  return null;
}

// Name validation
export function validateName(name: string): string | null {
  if (!name) return 'Name is required';
  if (name.length < 2) return 'Name must be at least 2 characters';
  return null;
}

// Generic required field validation
export function validateRequired(value: string, fieldName: string): string | null {
  if (!value || value.trim() === '') return `${fieldName} is required`;
  return null;
}

// Form validation helper
export function validateForm(values: Record<string, any>, validations: Record<string, Function>): Record<string, string | null> {
  const errors: Record<string, string | null> = {};
  
  Object.keys(validations).forEach(key => {
    const error = validations[key](values[key]);
    if (error) {
      errors[key] = error;
    }
  });
  
  return errors;
}
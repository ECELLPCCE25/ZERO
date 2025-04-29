import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { Link, useRouter } from 'expo-router';
import Input from '@/components/Input';
import Button from '@/components/Button';
import AnimatedCard from '@/components/AnimatedCard';
import LogoAnimation from '@/components/LogoAnimation';
import { useAuth } from '@/hooks/useAuth';
import { validateEmail, validatePassword, validateForm } from '@/utils/validators';

export default function Login() {
  const { signIn, error: authError } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{ email: string | null; password: string | null }>({
    email: null,
    password: null,
  });

  const handleLogin = async () => {
    // Validate form fields
    const validationErrors = validateForm(
      { email, password },
      { email: validateEmail, password: validatePassword }
    );

    setErrors(validationErrors as { email: string | null; password: string | null });

    // Check if there are any validation errors
    if (Object.values(validationErrors).some(error => error !== null)) {
      return;
    }

    // Attempt login
    setIsLoading(true);
    try {
      const { error } = await signIn(email, password);
      if (error) throw error;
    } catch (e) {
      // Error is handled by auth provider
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.content}>
          <LogoAnimation />
          
          <AnimatedCard style={styles.card}>
            <Text style={styles.title}>Welcome Back</Text>
            <Text style={styles.subtitle}>Sign in to continue</Text>
            
            {authError && (
              <View style={styles.errorContainer}>
                <Text style={styles.errorText}>{authError.message}</Text>
              </View>
            )}
            
            <View style={styles.form}>
              <Input
                // label="Email"
                placeholder="Enter your email"
                value={email}
                onChangeText={setEmail}
                error={errors.email}
                keyboardType="email-address"
                autoCapitalize="none"
              />
              
              <Input
                // label="Password"
                placeholder="Enter your password"
                value={password}
                onChangeText={setPassword}
                error={errors.password}
                secureTextEntry
              />
              
              <Link href="/(auth)/forgot-password" asChild>
                <TouchableOpacity style={styles.forgotPasswordLink}>
                  <Text style={styles.forgotPasswordText}>Forgot password?</Text>
                </TouchableOpacity>
              </Link>
              
              <Button
                title="Sign In"
                onPress={handleLogin}
                isLoading={isLoading}
                style={styles.button}
              />
            </View>
          </AnimatedCard>
          
          <View style={styles.footer}>
            <Text style={styles.footerText}>Don't have an account?</Text>
            <Link href="/(auth)/signup" asChild>
              <TouchableOpacity>
                <Text style={styles.signupLink}>Sign Up</Text>
              </TouchableOpacity>
            </Link>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F1F6F9',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
  },
  content: {
    flex: 1,
    padding: 24,
    justifyContent: 'center',
  },
  card: {
    width: '100%',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    fontFamily: 'Inter-Bold',
    color: '#1E293B',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#64748B',
    marginBottom: 24,
    textAlign: 'center',
    fontFamily: 'Inter-Regular',
  },
  form: {
    marginTop: 8,
  },
  forgotPasswordLink: {
    alignSelf: 'flex-end',
    marginTop: 4,
    marginBottom: 24,
  },
  forgotPasswordText: {
    color: '#3F72AF',
    fontSize: 14,
    fontFamily: 'Inter-Medium',
  },
  button: {
    marginTop: 8,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 24,
  },
  footerText: {
    color: '#64748B',
    fontSize: 14,
    fontFamily: 'Inter-Regular',
  },
  signupLink: {
    color: '#3F72AF',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 4,
    fontFamily: 'Inter-SemiBold',
  },
  errorContainer: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 14,
    fontFamily: 'Inter-Regular',
  },
});
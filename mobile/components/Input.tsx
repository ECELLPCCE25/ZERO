import React, { useState } from 'react';
import {
  StyleSheet,
  TextInput,
  View,
  Text,
  StyleProp,
  ViewStyle,
  TextStyle,
  TouchableOpacity,
} from 'react-native';
import { Eye, EyeOff } from 'lucide-react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

interface InputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  error?: string;
  secureTextEntry?: boolean;
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  autoCorrect?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
  style?: StyleProp<ViewStyle>;
  inputStyle?: StyleProp<TextStyle>;
  disabled?: boolean;
}

const Input = ({
  label,
  placeholder,
  value,
  onChangeText,
  error,
  secureTextEntry = false,
  autoCapitalize = 'none',
  autoCorrect = false,
  keyboardType = 'default',
  style,
  inputStyle,
  disabled = false,
}: InputProps) => {
  const [isFocused, setIsFocused] = useState(false);
  const [showPassword, setShowPassword] = useState(!secureTextEntry);

  // Animation values
  const labelPosition = useSharedValue(value ? -25 : 0);
  const labelScale = useSharedValue(value ? 0.8 : 1);
  const borderColor = useSharedValue(isFocused ? '#3F72AF' : '#E2E8F0');

  // Handle focus/blur animations
  const handleFocus = () => {
    setIsFocused(true);
    labelPosition.value = withTiming(-25, { duration: 150 });
    labelScale.value = withTiming(0.8, { duration: 150 });
    borderColor.value = withTiming('#3F72AF', { duration: 200 });
  };

  const handleBlur = () => {
    setIsFocused(false);
    if (!value) {
      labelPosition.value = withTiming(0, { duration: 150 });
      labelScale.value = withTiming(1, { duration: 150 });
    }
    borderColor.value = withTiming('#E2E8F0', { duration: 200 });
  };

  // Animated styles
  const animatedLabelStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateY: labelPosition.value },
        { scale: labelScale.value },
      ],
      color: isFocused ? '#3F72AF' : '#94A3B8',
    };
  });

  const animatedContainerStyle = useAnimatedStyle(() => {
    return {
      borderColor: error ? '#F43F5E' : borderColor.value,
    };
  });

  return (
    <View style={[styles.container, style]}>
      {label && (
        <Animated.Text style={[styles.label, animatedLabelStyle]}>
          {label}
        </Animated.Text>
      )}
      <Animated.View
        style={[
          styles.inputContainer,
          disabled && styles.disabledContainer,
          animatedContainerStyle,
        ]}
      >
        <TextInput
          style={[styles.input, inputStyle]}
          placeholder={placeholder}
          value={value}
          onChangeText={onChangeText}
          onFocus={handleFocus}
          onBlur={handleBlur}
          secureTextEntry={secureTextEntry && !showPassword}
          autoCapitalize={autoCapitalize}
          autoCorrect={autoCorrect}
          keyboardType={keyboardType}
          editable={!disabled}
          selectionColor="#3F72AF"
          placeholderTextColor="#94A3B8"
        />
        {secureTextEntry && (
          <TouchableOpacity
            onPress={() => setShowPassword(!showPassword)}
            style={styles.toggleButton}
          >
            {showPassword ? (
              <EyeOff size={20} color="#94A3B8" />
            ) : (
              <Eye size={20} color="#94A3B8" />
            )}
          </TouchableOpacity>
        )}
      </Animated.View>
      {error && <Text style={styles.errorText}>{error}</Text>}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
    width: '100%',
  },
  label: {
    position: 'absolute',
    left: 16,
    top: 16,
    fontSize: 16,
    color: '#94A3B8',
    zIndex: 1,
    backgroundColor: 'transparent',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E2E8F0',
    borderRadius: 12,
    paddingHorizontal: 16,
    height: 56,
    backgroundColor: 'white',
  },
  disabledContainer: {
    backgroundColor: '#F8FAFC',
    opacity: 0.7,
  },
  input: {
    flex: 1,
    fontSize: 16,
    color: '#334155',
    height: '100%',
  },
  errorText: {
    color: '#F43F5E',
    fontSize: 12,
    marginTop: 4,
    marginLeft: 4,
  },
  toggleButton: {
    padding: 8,
  },
});

export default Input;
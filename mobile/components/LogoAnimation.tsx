import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withDelay,
  Easing,
  withSequence,
  withRepeat,
} from 'react-native-reanimated';
import { MessageSquare } from 'lucide-react-native';

const LogoAnimation = () => {
  const rotation = useSharedValue(0);
  const size = useSharedValue(0.5);
  const opacity = useSharedValue(0);
  const position = useSharedValue(10);

  useEffect(() => {
    // Sequence of animations
    position.value = withTiming(0, { duration: 500, easing: Easing.out(Easing.back()) });
    opacity.value = withTiming(1, { duration: 400 });
    size.value = withSequence(
      withTiming(1.1, { duration: 300, easing: Easing.inOut(Easing.quad) }),
      withTiming(1, { duration: 200 })
    );
    
    // Continuous gentle pulse effect
    rotation.value = withRepeat(
      withSequence(
        withTiming(0.05, { duration: 1500, easing: Easing.inOut(Easing.sin) }),
        withTiming(-0.05, { duration: 1500, easing: Easing.inOut(Easing.sin) })
      ),
      -1, // Repeat infinitely
      true // Reverse
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [
        { translateY: position.value },
        { scale: size.value },
        { rotate: `${rotation.value * 30}deg` }
      ],
    };
  });

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.logoContainer, animatedStyle]}>
        <MessageSquare size={46} color="#3F72AF" />
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 40,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 20,
    backgroundColor: '#F1F6F9',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
});

export default LogoAnimation;
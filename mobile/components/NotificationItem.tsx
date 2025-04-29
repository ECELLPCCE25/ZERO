import React, { useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Bell } from 'lucide-react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSequence,
  Easing,
} from 'react-native-reanimated';

interface NotificationItemProps {
  id: string;
  title: string;
  message: string;
  time: string;
  isNew?: boolean;
  onPress?: () => void;
}

const NotificationItem = ({
  title,
  message,
  time,
  isNew = false,
  onPress,
}: NotificationItemProps) => {
  const scale = useSharedValue(isNew ? 0.95 : 1);
  const opacity = useSharedValue(isNew ? 0 : 1);
  const translateX = useSharedValue(isNew ? 20 : 0);

  useEffect(() => {
    if (isNew) {
      opacity.value = withTiming(1, { duration: 300 });
      translateX.value = withTiming(0, { duration: 300, easing: Easing.out(Easing.back()) });
      scale.value = withSequence(
        withTiming(1.02, { duration: 150 }),
        withTiming(1, { duration: 150 })
      );
    }
  }, [isNew]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [
        { scale: scale.value },
        { translateX: translateX.value },
      ],
    };
  });

  return (
    <TouchableOpacity activeOpacity={0.7} onPress={onPress}>
      <Animated.View style={[styles.container, animatedStyle]}>
        <View style={[styles.iconContainer, isNew && styles.newIconContainer]}>
          <Bell size={20} color={isNew ? 'white' : '#94A3B8'} />
        </View>
        <View style={styles.content}>
          <View style={styles.headerRow}>
            <Text style={styles.title}>{title}</Text>
            <Text style={styles.time}>{time}</Text>
          </View>
          <Text style={styles.message} numberOfLines={2}>
            {message}
          </Text>
        </View>
        {isNew && <View style={styles.indicator} />}
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F1F6F9',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  newIconContainer: {
    backgroundColor: '#3F72AF',
  },
  content: {
    flex: 1,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  title: {
    fontWeight: '600',
    fontSize: 16,
    color: '#1E293B',
  },
  time: {
    fontSize: 12,
    color: '#94A3B8',
  },
  message: {
    fontSize: 14,
    color: '#64748B',
    lineHeight: 20,
  },
  indicator: {
    position: 'absolute',
    top: 16,
    right: 16,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#EF4444',
  },
});

export default NotificationItem;
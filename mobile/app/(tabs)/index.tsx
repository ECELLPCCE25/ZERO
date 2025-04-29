import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withSequence,
} from 'react-native-reanimated';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/hooks/useSupabase';
import { Bell, LogOut } from 'lucide-react-native';
import { formatDistanceToNow } from '@/utils/dateUtils';
import { router } from 'expo-router';

interface Notification {
  id: string;
  title: string;
  message: string;
  created_at: string;
}

export default function HomeScreen() {
  const { user, signOut } = useAuth();
  const [notifications, setNotifications] = React.useState<Notification[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [refreshing, setRefreshing] = React.useState(false);
  
  // Animation values
  const listOpacity = useSharedValue(0);
  const listTranslateY = useSharedValue(20);
  
  useEffect(() => {
    // Initial animations
    listOpacity.value = withTiming(1, { duration: 600 });
    listTranslateY.value = withSpring(0, { damping: 12 });
    
    // Fetch initial notifications
    fetchNotifications();
    
    // Set up real-time subscription
    const subscription = supabase
      .channel('notifications')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'notifications',
          filter: `user_id=eq.${user?.id}`,
        },
        (payload) => {
          if (payload.eventType === 'INSERT') {
            setNotifications((prev) => [payload.new as Notification, ...prev]);
            // Trigger animation for new notification
            listTranslateY.value = withSequence(
              withSpring(-10),
              withSpring(0)
            );
          }
        }
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const fetchNotifications = async () => {
    try {
      const { data, error } = await supabase
        .from('notifications')
        .select('*')
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setNotifications(data || []);
    } catch (error) {
      console.error('Error fetching notifications:', error);
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    fetchNotifications();
  }, []);

  // Animated styles
  const listAnimatedStyle = useAnimatedStyle(() => ({
    opacity: listOpacity.value,
    transform: [{ translateY: listTranslateY.value }],
  }));

  const renderNotification = (notification: Notification) => (
    <Animated.View key={notification.id} style={styles.notificationCard}>
      <View style={styles.notificationIcon}>
        <Bell size={20} color="#3F72AF" />
      </View>
      <View style={styles.notificationContent}>
        <Text style={styles.notificationTitle}>{notification.title}</Text>
        <Text style={styles.notificationMessage}>{notification.message}</Text>
        <Text style={styles.notificationTime}>
          {formatDistanceToNow(new Date(notification.created_at))}
        </Text>
      </View>
    </Animated.View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.headerTitle}>Home</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity onPress={() => router.push('/notifications')} style={styles.iconButton}>
            <Bell size={24} color="#3F72AF" />
          </TouchableOpacity>
          <TouchableOpacity onPress={signOut} style={styles.iconButton}>
            <LogOut size={24} color="#EF4444" />
          </TouchableOpacity>
        </View>
      </View>
      
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {isLoading ? (
          <View style={styles.centerContent}>
            <Text>Loading notifications...</Text>
          </View>
        ) : notifications.length === 0 ? (
          <View style={styles.centerContent}>
            <Bell size={48} color="#94A3B8" />
            <Text style={styles.emptyText}>No notifications yet</Text>
          </View>
        ) : (
          <Animated.View style={listAnimatedStyle}>
            {notifications.map(renderNotification)}
          </Animated.View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F1F6F9',
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E2E8F0',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerLeft: {
    flex: 1,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1E293B',
    fontFamily: 'Inter-Bold',
  },
  iconButton: {
    padding: 8,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 300,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
    color: '#64748B',
    fontFamily: 'Inter-Medium',
  },
  notificationCard: {
    flexDirection: 'row',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  notificationIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F1F6F9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  notificationContent: {
    flex: 1,
  },
  notificationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1E293B',
    marginBottom: 4,
    fontFamily: 'Inter-SemiBold',
  },
  notificationMessage: {
    fontSize: 14,
    color: '#64748B',
    marginBottom: 8,
    fontFamily: 'Inter-Regular',
  },
  notificationTime: {
    fontSize: 12,
    color: '#94A3B8',
    fontFamily: 'Inter-Regular',
  },
});

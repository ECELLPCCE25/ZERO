import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNotifications } from '@/hooks/useNotifications';
import NotificationItem from '@/components/NotificationItem';
import { formatDistanceToNow } from '@/utils/dateUtils';
import Button from '@/components/Button';
import { Bell, BellOff, Check } from 'lucide-react-native';
import Animated, { 
  FadeIn, 
  FadeOut,
  SlideInRight,
  Layout 
} from 'react-native-reanimated';

export default function NotificationsScreen() {
  const { 
    notifications, 
    isLoading, 
    error, 
    markAsRead, 
    markAllAsRead,
    unreadCount 
  } = useNotifications();
  const [selectedNotification, setSelectedNotification] = useState<string | null>(null);

  // Handle notification press
  const handleNotificationPress = (id: string) => {
    setSelectedNotification(id);
    markAsRead(id);
  };

  // Handle clear all notifications
  const handleClearAll = () => {
    markAllAsRead();
  };

  // Empty state
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <BellOff size={64} color="#94A3B8" />
      <Text style={styles.emptyTitle}>No Notifications</Text>
      <Text style={styles.emptyText}>
        You don't have any notifications yet. We'll notify you when something important happens.
      </Text>
    </View>
  );

  // Error state
  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Notifications</Text>
        </View>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            An error occurred while loading notifications.
          </Text>
          <Button 
            title="Try Again" 
            onPress={() => window.location.reload()} 
            style={styles.errorButton}
          />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Notifications</Text>
        {notifications.length > 0 && (
          <TouchableOpacity
            style={styles.clearAllButton}
            onPress={handleClearAll}
            disabled={unreadCount === 0}
          >
            <Text 
              style={[
                styles.clearAllText, 
                unreadCount === 0 && styles.disabledText
              ]}
            >
              Mark all as read
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3F72AF" />
        </View>
      ) : (
        <>
          {unreadCount > 0 && (
            <Animated.View 
              style={styles.unreadBanner}
              entering={FadeIn.duration(400)}
              exiting={FadeOut.duration(300)}
            >
              <Bell size={16} color="#FFFFFF" />
              <Text style={styles.unreadText}>
                {unreadCount} unread {unreadCount === 1 ? 'notification' : 'notifications'}
              </Text>
            </Animated.View>
          )}

          <FlatList
            data={notifications}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <Animated.View
                layout={Layout.springify()}
                entering={SlideInRight.duration(400).delay(item.id.charCodeAt(0) % 4 * 100)}
              >
                <NotificationItem
                  id={item.id}
                  title={item.title}
                  message={item.message}
                  time={formatDistanceToNow(new Date(item.created_at))}
                  isNew={!item.read}
                  onPress={() => handleNotificationPress(item.id)}
                />
              </Animated.View>
            )}
            contentContainerStyle={styles.listContent}
            ListEmptyComponent={renderEmptyState}
            showsVerticalScrollIndicator={false}
          />
        </>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F1F6F9',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E2E8F0',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1E293B',
    fontFamily: 'Inter-SemiBold',
  },
  clearAllButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  clearAllText: {
    fontSize: 14,
    color: '#3F72AF',
    fontFamily: 'Inter-Medium',
  },
  disabledText: {
    color: '#94A3B8',
  },
  unreadBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#3F72AF',
    paddingVertical: 8,
    paddingHorizontal: 16,
    gap: 8,
  },
  unreadText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontFamily: 'Inter-Medium',
  },
  listContent: {
    padding: 16,
    paddingBottom: 32,
    flexGrow: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    minHeight: 400,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1E293B',
    marginTop: 16,
    marginBottom: 8,
    fontFamily: 'Inter-SemiBold',
  },
  emptyText: {
    fontSize: 14,
    color: '#64748B',
    textAlign: 'center',
    maxWidth: 280,
    fontFamily: 'Inter-Regular',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  errorText: {
    fontSize: 16,
    color: '#EF4444',
    marginBottom: 16,
    textAlign: 'center',
    fontFamily: 'Inter-Medium',
  },
  errorButton: {
    marginTop: 12,
  },
});
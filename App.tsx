import 'react-native-url-polyfill/auto';
import 'text-encoding';

import 'react-native-gesture-handler';
import React, { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import HomeScreen from './src/screens/HomeScreen';
import { checkAndDownloadModels } from './src/ai/modelManager';
import { initDatabase } from './src/db/database';

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState('Checking models...');

  useEffect(() => {
    async function setupApp() {
      try {
        // 1. Initialize OP-SQLite and sqlite-vec
        await initDatabase();
        
        // 2. Check and Download AI Models
        await checkAndDownloadModels((status) => {
            setDownloadProgress(status);
        });

        setIsReady(true);
      } catch (error) {
        console.error("Setup failed:", error);
        setDownloadProgress("Error initializing app.");
      }
    }
    
    setupApp();
  }, []);

  if (!isReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.loadingText}>{downloadProgress}</Text>
      </View>
    );
  }

  return <HomeScreen />;
}

const styles = StyleSheet.create({
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { marginTop: 20, fontSize: 16 }
});
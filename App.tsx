import 'react-native-url-polyfill/auto';
import 'text-encoding';

import 'react-native-gesture-handler';
import React, { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator, StyleSheet, TouchableOpacity } from 'react-native';
import HomeScreen from './src/screens/HomeScreen';
import { checkAndDownloadModels } from './src/ai/modelManager';
import { initDatabase } from './src/db/database';

/**
 * Get memory information if available
 */
function getMemoryInfo() {
  if (typeof performance !== 'undefined' && (performance as any).memory) {
    return (performance as any).memory;
  }
  return null;
}

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState('Checking models...');
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [memoryWarning, setMemoryWarning] = useState('');

  useEffect(() => {
    async function setupApp() {
      try {
        setHasError(false);
        setErrorMessage('');
        setMemoryWarning('');

        // Check device memory before starting
        const memoryInfo = getMemoryInfo();
        if (memoryInfo && memoryInfo.jsHeapSizeLimit < 300 * 1024 * 1024) { // Less than 300MB
          setMemoryWarning('⚠️ Low memory device detected. App may be slow or unstable.');
        }

        // 1. Initialize OP-SQLite and sqlite-vec
        await initDatabase();
        
        // 2. Check and Download AI Models
        await checkAndDownloadModels((status) => {
            setDownloadProgress(status);
        });

        setIsReady(true);
      } catch (error) {
        console.error("Setup failed:", error);
        setHasError(true);
        setErrorMessage(error instanceof Error ? error.message : 'Unknown error occurred');
        setDownloadProgress("Error initializing app.");
      }
    }
    
    setupApp();
  }, []);

  const retrySetup = () => {
    setIsReady(false);
    setHasError(false);
    setErrorMessage('');
    setDownloadProgress('Retrying setup...');
    // Trigger useEffect again by forcing a re-render
    setTimeout(() => {
      const setupApp = async () => {
        try {
          setHasError(false);
          setErrorMessage('');
          await initDatabase();
          await checkAndDownloadModels((status) => {
              setDownloadProgress(status);
          });
          setIsReady(true);
        } catch (error) {
          console.error("Setup failed:", error);
          setHasError(true);
          setErrorMessage(error instanceof Error ? error.message : 'Unknown error occurred');
          setDownloadProgress("Error initializing app.");
        }
      };
      setupApp();
    }, 100);
  };

  if (hasError) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorTitle}>Setup Failed</Text>
        <Text style={styles.errorText}>{errorMessage}</Text>
        <Text style={styles.errorText}>{downloadProgress}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={retrySetup}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!isReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.loadingText}>{downloadProgress}</Text>
        {memoryWarning ? (
          <Text style={styles.memoryWarning}>{memoryWarning}</Text>
        ) : null}
      </View>
    );
  }

  return <HomeScreen />;
}

const styles = StyleSheet.create({
  loadingContainer: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center',
    backgroundColor: '#fff'
  },
  loadingText: { 
    marginTop: 20, 
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 20
  },
  memoryWarning: {
    marginTop: 20,
    fontSize: 14,
    color: '#ff6b35',
    textAlign: 'center',
    fontWeight: '600',
    paddingHorizontal: 20
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 20
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ff4444',
    marginBottom: 10
  },
  errorText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 20,
    lineHeight: 22
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 8
  },
  retryText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600'
  }
});
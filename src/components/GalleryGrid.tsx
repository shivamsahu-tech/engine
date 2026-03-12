import React from 'react';
import { FlatList, StyleSheet, View, Text, ActivityIndicator } from 'react-native';
import { GestureHandlerRootView, PinchGestureHandler, State } from 'react-native-gesture-handler';
import ImageItem from './ImageItem';
import { GalleryImage } from '../types/gallery';

interface GalleryGridProps {
    images: GalleryImage[];
    isLoading: boolean;
    numColumns: number;
    onColumnsChange: (columns: number) => void;
    onImagePress: (index: number) => void;
}

const GalleryGrid: React.FC<GalleryGridProps> = ({ images, isLoading, numColumns, onColumnsChange, onImagePress }) => {

    const onPinchGestureEvent = (event: any) => {
        // We can use scale to determine if we should zoom in or out
        // Zoom in (scale > 1) -> fewer columns (bigger images)
        // Zoom out (scale < 1) -> more columns (smaller images)
    };

    const onPinchHandlerStateChange = (event: any) => {
        if (event.nativeEvent.state === State.END) {
            const scale = event.nativeEvent.scale;
            if (scale > 1.2 && numColumns > 2) {
                onColumnsChange(numColumns - 1);
            } else if (scale < 0.8 && numColumns < 5) {
                onColumnsChange(numColumns + 1);
            }
        }
    };

    if (isLoading && images.length === 0) {
        return (
            <View style={styles.center}>
                <ActivityIndicator size="large" color="#007AFF" />
                <Text style={styles.loadingText}>Searching images...</Text>
            </View>
        );
    }

    return (
        <GestureHandlerRootView style={styles.container}>
            <PinchGestureHandler
                onGestureEvent={onPinchGestureEvent}
                onHandlerStateChange={onPinchHandlerStateChange}
            >
                <View style={styles.container}>
                    <FlatList
                        key={numColumns} // Force re-render when numColumns changes
                        data={images}
                        keyExtractor={(item) => item.id}
                        numColumns={numColumns}
                        renderItem={({ item, index }) => (
                            <ImageItem
                                image={item}
                                numColumns={numColumns}
                                onPress={() => onImagePress(index)}
                            />
                        )}
                        contentContainerStyle={styles.listContent}
                        ListEmptyComponent={
                            !isLoading ? (
                                <View style={styles.center}>
                                    <Text style={styles.emptyText}>No images found</Text>
                                </View>
                            ) : null
                        }
                    />
                </View>
            </PinchGestureHandler>
        </GestureHandlerRootView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    listContent: {
        paddingBottom: 80, // Space for FAB
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40,
    },
    loadingText: {
        marginTop: 12,
        color: '#666',
        fontSize: 16,
    },
    emptyText: {
        color: '#999',
        fontSize: 16,
        textAlign: 'center',
    },
});

export default GalleryGrid;

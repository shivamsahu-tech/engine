import React, { useState } from 'react';
import { StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import SearchBar from '../components/SearchBar';
import SortOptions from '../components/SortOptions';
import GalleryGrid from '../components/GalleryGrid';
import CameraModal from '../components/CameraModal';
import { useGallery } from '../hooks/useGallery';
import { StatusBar } from 'expo-status-bar';
import ImageView from 'react-native-image-viewing';

const HomeScreen: React.FC = () => {
    const {
        displayImages,
        searchQuery,
        isSearching,
        isIngesting,
        sortType,
        handleSearch,
        handleIngest,
        handleSort,
        clearSearch
    } = useGallery();

    const [isCameraVisible, setIsCameraVisible] = useState(false);
    const [isViewerVisible, setIsViewerVisible] = useState(false);
    const [viewerIndex, setViewerIndex] = useState(0);
    const [numColumns, setNumColumns] = useState(3);

    const onCapture = (uri: string) => {
        setIsCameraVisible(false);
        handleIngest(uri);
    };

    const pickImage = async () => {
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['images'],
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled) {
            handleIngest(result.assets[0].uri);
        }
    };

    const openViewer = (index: number) => {
        setViewerIndex(index);
        setIsViewerVisible(true);
    };

    const viewerImages = displayImages.map(img => ({ uri: img.uri }));

    return (
        <SafeAreaProvider>
            <SafeAreaView style={styles.container}>
                <StatusBar style="dark" />

                <SearchBar
                    value={searchQuery}
                    onChangeText={handleSearch}
                    onSubmit={() => handleSearch(searchQuery)}
                    onClear={clearSearch}
                    isLoading={isSearching}
                />

                <SortOptions
                    currentSort={sortType}
                    onSortChange={handleSort}
                />

                <View style={styles.content}>
                    <GalleryGrid
                        images={displayImages}
                        isLoading={isSearching}
                        numColumns={numColumns}
                        onColumnsChange={setNumColumns}
                        onImagePress={openViewer}
                    />

                    {isIngesting && (
                        <View style={styles.toast}>
                            <Text style={styles.toastText}>Processing image...</Text>
                        </View>
                    )}
                </View>

                <View style={styles.fabContainer}>
                    <TouchableOpacity
                        style={styles.galleryFab}
                        onPress={pickImage}
                    >
                        <Ionicons name="images" size={24} color="#007AFF" />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.fabMain}
                        onPress={() => setIsCameraVisible(true)}
                    >
                        <Ionicons name="camera" size={30} color="#fff" />
                    </TouchableOpacity>
                </View>

                <CameraModal
                    isVisible={isCameraVisible}
                    onClose={() => setIsCameraVisible(false)}
                    onCapture={onCapture}
                />

                <ImageView
                    images={viewerImages}
                    imageIndex={viewerIndex}
                    visible={isViewerVisible}
                    onRequestClose={() => setIsViewerVisible(false)}
                />
            </SafeAreaView>
        </SafeAreaProvider>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    content: {
        flex: 1,
        position: 'relative',
    },
    fabContainer: {
        position: 'absolute',
        bottom: 30,
        flexDirection: 'row',
        alignSelf: 'center',
        alignItems: 'center',
        gap: 16,
    },
    fabMain: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: '#007AFF',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 8,
    },
    galleryFab: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: '#fff',
        borderWidth: 1,
        borderColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 4,
    },
    toast: {
        position: 'absolute',
        top: 20,
        alignSelf: 'center',
        backgroundColor: 'rgba(0,0,0,0.8)',
        paddingHorizontal: 20,
        paddingVertical: 10,
        borderRadius: 20,
        zIndex: 100,
    },
    toastText: {
        color: '#fff',
        fontWeight: '600',
    },
});

export default HomeScreen;

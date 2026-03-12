import React from 'react';
import { StyleSheet, TouchableOpacity, Dimensions } from 'react-native';
import { Image } from 'expo-image';
import { GalleryImage } from '../types/gallery';

interface ImageItemProps {
    image: GalleryImage;
    numColumns: number;
    onPress: () => void;
}

const ImageItem: React.FC<ImageItemProps> = ({ image, numColumns, onPress }) => {
    const windowWidth = Dimensions.get('window').width;
    const size = windowWidth / numColumns;

    return (
        <TouchableOpacity
            activeOpacity={0.8}
            style={[styles.container, { width: size, height: size }]}
            onPress={onPress}
        >
            <Image
                source={{ uri: image.uri }}
                style={styles.image}
                contentFit="cover"
                transition={200}
            />
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    container: {
        padding: 1,
    },
    image: {
        flex: 1,
        backgroundColor: '#eee',
    },
});

export default ImageItem;

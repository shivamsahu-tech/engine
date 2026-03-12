import React, { useState, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Modal } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';

interface CameraModalProps {
    isVisible: boolean;
    onClose: () => void;
    onCapture: (uri: string) => void;
}

const CameraModal: React.FC<CameraModalProps> = ({ isVisible, onClose, onCapture }) => {
    const [facing, setFacing] = useState<CameraType>('back');
    const [permission, requestPermission] = useCameraPermissions();
    const cameraRef = useRef<any>(null);

    if (!permission) {
        // Camera permissions are still loading.
        return <View />;
    }

    if (!permission.granted) {
        // Camera permissions are not granted yet.
        return (
            <Modal visible={isVisible} animationType="slide">
                <View style={styles.permissionContainer}>
                    <Text style={styles.message}>We need your permission to show the camera</Text>
                    <TouchableOpacity onPress={requestPermission} style={styles.button}>
                        <Text style={styles.buttonText}>Grant Permission</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                        <Ionicons name="close" size={30} color="#000" />
                    </TouchableOpacity>
                </View>
            </Modal>
        );
    }

    const toggleCameraFacing = () => {
        setFacing(current => (current === 'back' ? 'front' : 'back'));
    };

    const takePicture = async () => {
        if (cameraRef.current) {
            try {
                const photo = await cameraRef.current.takePictureAsync();
                onCapture(photo.uri);
            } catch (error) {
                console.error("Failed to take picture:", error);
            }
        }
    };

    return (
        <Modal visible={isVisible} animationType="slide" transparent={false}>
            <View style={styles.container}>
                <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
                    <SafeAreaView style={styles.overlay}>
                        <View style={styles.header}>
                            <TouchableOpacity onPress={onClose} style={styles.iconButton}>
                                <Ionicons name="close" size={28} color="#fff" />
                            </TouchableOpacity>
                        </View>

                        <View style={styles.footer}>
                            <TouchableOpacity onPress={toggleCameraFacing} style={styles.iconButton}>
                                <Ionicons name="camera-reverse" size={28} color="#fff" />
                            </TouchableOpacity>

                            <TouchableOpacity onPress={takePicture} style={styles.captureButton}>
                                <View style={styles.captureButtonInner} />
                            </TouchableOpacity>

                            <View style={{ width: 44 }} />
                        </View>
                    </SafeAreaView>
                </CameraView>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#000',
    },
    camera: {
        flex: 1,
    },
    overlay: {
        flex: 1,
        justifyContent: 'space-between',
    },
    header: {
        padding: 16,
        flexDirection: 'row',
        justifyContent: 'flex-start',
    },
    footer: {
        paddingBottom: 40,
        paddingHorizontal: 30,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    iconButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(0,0,0,0.3)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    captureButton: {
        width: 80,
        height: 80,
        borderRadius: 40,
        borderWidth: 4,
        borderColor: '#fff',
        justifyContent: 'center',
        alignItems: 'center',
    },
    captureButtonInner: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: '#fff',
    },
    permissionContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    message: {
        textAlign: 'center',
        marginBottom: 20,
        fontSize: 16,
    },
    button: {
        backgroundColor: '#007AFF',
        paddingHorizontal: 20,
        paddingVertical: 10,
        borderRadius: 8,
    },
    buttonText: {
        color: '#fff',
        fontWeight: '600',
    },
    closeButton: {
        position: 'absolute',
        top: 50,
        right: 20,
    }
});

export default CameraModal;

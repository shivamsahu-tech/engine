import React from 'react';
import { View, TextInput, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface SearchBarProps {
    value: string;
    onChangeText: (text: string) => void;
    onSubmit: () => void;
    onClear: () => void;
    isLoading: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({ value, onChangeText, onSubmit, onClear, isLoading }) => {
    return (
        <View style={styles.container}>
            <View style={styles.searchSection}>
                <Ionicons name="search" size={20} color="#666" style={styles.searchIcon} />
                <TextInput
                    style={styles.input}
                    placeholder="Search your images..."
                    placeholderTextColor="#999"
                    value={value}
                    onChangeText={onChangeText}
                    onSubmitEditing={onSubmit}
                    returnKeyType="search"
                    editable={!isLoading}
                />
                {value.length > 0 && (
                    <TouchableOpacity onPress={onClear} style={styles.clearButton}>
                        <Ionicons name="close-circle" size={20} color="#666" />
                    </TouchableOpacity>
                )}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        paddingHorizontal: 16,
        paddingVertical: 12,
        backgroundColor: '#fff',
    },
    searchSection: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#f0f0f0',
        borderRadius: 12,
        paddingHorizontal: 12,
        height: 48,
    },
    searchIcon: {
        marginRight: 8,
    },
    input: {
        flex: 1,
        height: '100%',
        color: '#000',
        fontSize: 16,
    },
    clearButton: {
        padding: 4,
    },
});

export default SearchBar;

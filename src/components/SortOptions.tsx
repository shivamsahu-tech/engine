import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';

export type SortType = 'Newest' | 'Oldest' | 'A-Z';

interface SortOptionsProps {
    currentSort: SortType;
    onSortChange: (sort: SortType) => void;
}

const SortOptions: React.FC<SortOptionsProps> = ({ currentSort, onSortChange }) => {
    const options: SortType[] = ['Newest', 'Oldest', 'A-Z'];

    return (
        <View style={styles.container}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                {options.map((option) => (
                    <TouchableOpacity
                        key={option}
                        style={[
                            styles.option,
                            currentSort === option && styles.activeOption
                        ]}
                        onPress={() => onSortChange(option)}
                    >
                        <Text style={[
                            styles.optionText,
                            currentSort === option && styles.activeOptionText
                        ]}>
                            {option}
                        </Text>
                    </TouchableOpacity>
                ))}
            </ScrollView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        paddingVertical: 8,
        backgroundColor: '#fff',
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    scrollContent: {
        paddingHorizontal: 16,
    },
    option: {
        paddingHorizontal: 16,
        paddingVertical: 6,
        borderRadius: 20,
        backgroundColor: '#f0f0f0',
        marginRight: 8,
    },
    activeOption: {
        backgroundColor: '#007AFF',
    },
    optionText: {
        fontSize: 14,
        color: '#666',
        fontWeight: '500',
    },
    activeOptionText: {
        color: '#fff',
    },
});

export default SortOptions;

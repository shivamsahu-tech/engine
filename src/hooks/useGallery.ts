import { useState, useEffect, useCallback } from 'react';
import { GalleryImage } from '../types/gallery';
import { SortType } from '../components/SortOptions';
import { RetrievalFunction } from '../pipeline/retrieval';
import { IngestionFunction } from '../pipeline/ingestion';


export const useGallery = () => {
    const [allImages, setAllImages] = useState<GalleryImage[]>([]);
    const [displayImages, setDisplayImages] = useState<GalleryImage[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [isIngesting, setIsIngesting] = useState(false);
    const [sortType, setSortType] = useState<SortType>('Newest');

    // Initial mock data
    useEffect(() => {
        const mockImages: GalleryImage[] = [
            { id: '1', uri: 'https://picsum.photos/id/10/400/400', createdAt: Date.now() - 1000000 },
            { id: '2', uri: 'https://picsum.photos/id/20/400/400', createdAt: Date.now() - 500000 },
            { id: '3', uri: 'https://picsum.photos/id/30/400/400', createdAt: Date.now() - 2000000 },
            { id: '4', uri: 'https://picsum.photos/id/40/400/400', createdAt: Date.now() - 100000 },
            { id: '5', uri: 'https://picsum.photos/id/50/400/400', createdAt: Date.now() - 3000000 },
            { id: '6', uri: 'https://picsum.photos/id/60/400/400', createdAt: Date.now() - 4000000 },
        ];
        setAllImages(mockImages);
        setDisplayImages(mockImages);
    }, []);

    const handleSearch = useCallback(async (query: string) => {
        setSearchQuery(query);
        setIsSearching(true);
        try {
            const results = await RetrievalFunction(query, allImages);
            setDisplayImages(results);
        } catch (error) {
            console.error("Search failed:", error);
            // Fallback to showing all images if search fails
            setDisplayImages(allImages);
        } finally {
            setIsSearching(false);
        }
    }, [allImages]);

    const handleIngest = useCallback(async (uri: string) => {
        setIsIngesting(true);
        try {
            const newImage = await IngestionFunction(uri);
            setAllImages(prev => [newImage, ...prev]);
            // If we are currently searching, we might want to refresh search or just show all
            if (!searchQuery) {
                setDisplayImages(prev => [newImage, ...prev]);
            }
        } catch (error) {
            console.error("Ingestion failed:", error);
            // Don't crash the app - just log the error
            // Optionally show a toast or alert to user
        } finally {
            setIsIngesting(false);
        }
    }, [searchQuery]);

    const handleSort = useCallback((type: SortType) => {
        setSortType(type);
        setDisplayImages(prev => {
            const sorted = [...prev];
            if (type === 'Newest') {
                return sorted.sort((a, b) => b.createdAt - a.createdAt);
            } else if (type === 'Oldest') {
                return sorted.sort((a, b) => a.createdAt - b.createdAt);
            } else if (type === 'A-Z') {
                // Just mock sorting for UI feedback
                return sorted.sort((a, b) => a.id.localeCompare(b.id));
            }
            return sorted;
        });
    }, []);

    const clearSearch = useCallback(() => {
        setSearchQuery('');
        setDisplayImages(allImages);
    }, [allImages]);

    return {
        displayImages,
        searchQuery,
        isSearching,
        isIngesting,
        sortType,
        handleSearch,
        handleIngest,
        handleSort,
        clearSearch
    };
};

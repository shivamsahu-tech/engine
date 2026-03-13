import {
    env,
    AutoProcessor,
    SiglipVisionModel,
    AutoTokenizer,
    SiglipTextModel
} from '@xenova/transformers';

import { documentDirectory, makeDirectoryAsync, getInfoAsync, downloadAsync } from 'expo-file-system/legacy';



// Mobile Configuration:
// Configure Transformers to strictly use local files only
env.allowRemoteModels = false; // CRITICAL: Prevent fetch() downloads through JS bridge
env.allowLocalModels = true; // Allow loading from disk
env.localModelPath = documentDirectory + 'models/';
env.useBrowserCache = false;

// Aggressive mobile memory configuration
env.backends.onnx.wasm.numThreads = 1; // Single thread to reduce memory
env.backends.onnx.wasm.wasmPaths = {}; // Let it auto-detect
env.backends.onnx.wasm.initTimeout = 30000; // 30 second timeout

// Disable progress callbacks that can send binary data chunks to JS
env.backends.onnx.logLevel = 'error'; // Only log errors, not progress

// Model configuration for CLIP patch16 (smaller model)
const MODEL_NAME = 'Xenova/clip-vit-base-patch16';
const MODEL_BASE_URL = 'https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main';

// Files needed for the CLIP model (pre-download these to avoid JS bridge)
const MODEL_FILES = [
    'onnx/model_quantized.onnx',
    'tokenizer.json',
    'preprocessor_config.json',
    'config.json',
    'special_tokens_map.json',
    'tokenizer_config.json'
];

const MODEL_DIR = documentDirectory + 'models/' + MODEL_NAME + '/';

// Singleton variables to keep models in memory after loading
let processor: any = null;
let visionModel: any = null;
let tokenizer: any = null;
let textModel: any = null;
let isInitialized = false;
let memoryWarningShown = false;

/**
 * The Critical Math Fix: L2 Normalization
 * Forces the vector into a standard coordinate space so distances compare correctly.
 */
export function normalizeVector(array: Float32Array): Float32Array {
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
        sum += array[i] * array[i];
    }
    const magnitude = Math.sqrt(sum);
    
    // Prevent division by zero
    if (magnitude === 0) return array;
    
    for (let i = 0; i < array.length; i++) {
        array[i] /= magnitude;
    }
    return array;
}

/**
 * Pre-computes the Dot Product (Cosine Similarity) of two already L2-normalized vectors in pure JS 
 */
export function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
    let dotProduct = 0;
    const len = Math.min(vecA.length, vecB.length);
    for (let i = 0; i < len; i++) {
        dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
}

/**
 * Downloads and caches the AI models on the device.
 * Takes a callback to update the loading screen UI.
 * 
 * IMPORTANT: Make sure to disable network inspection in React Native Dev Menu
 * to prevent Base64 encoding of large downloads that causes OOM crashes.
 */
export async function checkAndDownloadModels(onProgress: (status: string) => void): Promise<void> {
    try {
        if (isInitialized) {
            onProgress('Models already loaded!');
            return;
        }

        onProgress('Checking model files...');

        // Ensure model directory exists
        await makeDirectoryAsync(MODEL_DIR, { intermediates: true });

        // Check which files need to be downloaded
        const filesToDownload: string[] = [];
        for (const file of MODEL_FILES) {
            const localPath = MODEL_DIR + file;
            const fileInfo = await getInfoAsync(localPath);
            if (!fileInfo.exists) {
                filesToDownload.push(file);
            }
        }

        if (filesToDownload.length > 0) {
            onProgress(`Downloading ${filesToDownload.length} model files natively...`);

            // Download files one by one using native FileSystem (bypasses JS bridge)
            for (let i = 0; i < filesToDownload.length; i++) {
                const file = filesToDownload[i];
                const remoteUrl = MODEL_BASE_URL + '/' + file;
                const localPath = MODEL_DIR + file;

                onProgress(`Downloading ${file} (${i + 1}/${filesToDownload.length})...`);
                console.log(`Downloading ${file} from ${remoteUrl} to ${localPath}`);

                const downloadResult = await downloadAsync(remoteUrl, localPath);
                console.log(`✅ Downloaded ${file} (${downloadResult.status})`);
            }

            onProgress('All model files downloaded successfully!');
        } else {
            onProgress('All model files already exist locally.');
        }

        // Now load the models from local files (no network calls)
        onProgress('Initializing AI models from local files...');

        // Load models one by one with memory cleanup between loads
        onProgress('Loading Vision Processor...');
        processor = await loadModelWithRetry(
            () => AutoProcessor.from_pretrained(MODEL_NAME),
            'Vision Processor'
        );

        await cleanupMemory();

        onProgress('Loading Vision Model...');
        visionModel = await loadModelWithRetry(
            () => SiglipVisionModel.from_pretrained(MODEL_NAME, {
                quantized: true,
            }),
            'Vision Model'
        );

        await cleanupMemory();

        onProgress('Loading Text Tokenizer...');
        tokenizer = await loadModelWithRetry(
            () => AutoTokenizer.from_pretrained(MODEL_NAME),
            'Text Tokenizer'
        );

        await cleanupMemory();

        onProgress('Loading Text Model...');
        textModel = await loadModelWithRetry(
            () => SiglipTextModel.from_pretrained(MODEL_NAME, {
                quantized: true,
            }),
            'Text Model'
        );

        await cleanupMemory();
        isInitialized = true;
        onProgress('All models ready!');
        console.log("✅ Models loaded into memory from local files.");
    } catch (error) {
        console.error("❌ Model loading failed:", error);
        onProgress('Error loading AI models.');
        // Clean up any partially loaded models
        cleanupModels();
        throw error;
    }
}

/**
 * Returns the cached model instances for the ingestion and retrieval pipelines.
 */
export async function getModels() {
    if (!isInitialized || !processor || !visionModel || !tokenizer || !textModel) {
        throw new Error("Models are not loaded yet. Call checkAndDownloadModels first.");
    }
    return { processor, visionModel, tokenizer, textModel };
}

/**
 * Force cleanup of models to free memory (useful for low-memory situations)
 */
export function cleanupModels() {
    processor = null;
    visionModel = null;
    tokenizer = null;
    textModel = null;
    isInitialized = false;

    // Force garbage collection if available
    if (global.gc) {
        global.gc();
    }

    console.log("🧹 Models cleaned up from memory");
}

/**
 * Load a model with retry logic and memory management
 */
async function loadModelWithRetry(loader: () => Promise<any>, modelName: string, maxRetries = 2): Promise<any> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`Loading ${modelName} (attempt ${attempt})`);
            const model = await Promise.race([
                loader(),
                new Promise((_, reject) =>
                    setTimeout(() => reject(new Error(`${modelName} loading timeout`)), 60000)
                )
            ]);
            return model;
        } catch (error) {
            console.warn(`${modelName} loading failed (attempt ${attempt}):`, error);
            if (attempt < maxRetries) {
                await cleanupMemory();
                await new Promise(resolve => setTimeout(resolve, 2000)); // Wait before retry
            } else {
                throw error;
            }
        }
    }
}

/**
 * Get memory information if available
 */
function getMemoryInfo() {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
        return (performance as any).memory;
    }
    return null;
}

/**
 * Aggressive memory cleanup
 */
async function cleanupMemory() {
    // Force garbage collection if available
    if (global.gc) {
        global.gc();
    }

    // Small delay to allow cleanup
    await new Promise(resolve => setTimeout(resolve, 100));
}
import { 
    env, 
    AutoProcessor, 
    SiglipVisionModel, 
    AutoTokenizer, 
    SiglipTextModel 
} from '@xenova/transformers';



// Mobile Configuration:
// Tell transformers.js not to look for local desktop files, but to download
// and cache the models using the React Native file system.
env.allowLocalModels = false; 
env.useBrowserCache = false; 

// Singleton variables to keep models in memory after loading
let processor: any = null;
let visionModel: any = null;
let tokenizer: any = null;
let textModel: any = null;

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
 */
export async function checkAndDownloadModels(onProgress: (status: string) => void): Promise<void> {
    try {
        onProgress('Downloading Vision Processor...');
        processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
        
        onProgress('Downloading Vision Model (This may take a minute)...');
        visionModel = await SiglipVisionModel.from_pretrained('Xenova/clip-vit-base-patch32', { quantized: true });
        
        onProgress('Downloading Text Tokenizer...');
        tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
        
        onProgress('Downloading Text Model...');
        textModel = await SiglipTextModel.from_pretrained('Xenova/clip-vit-base-patch32', { quantized: true });
        
        onProgress('All models ready!');
        console.log("✅ Models loaded into memory.");
    } catch (error) {
        console.error("❌ Model loading failed:", error);
        onProgress('Error loading AI models.');
        throw error;
    }
}

/**
 * Returns the cached model instances for the ingestion and retrieval pipelines.
 */
export async function getModels() {
    if (!processor || !visionModel || !tokenizer || !textModel) {
        throw new Error("Models are not loaded yet. Call checkAndDownloadModels first.");
    }
    return { processor, visionModel, tokenizer, textModel };
}
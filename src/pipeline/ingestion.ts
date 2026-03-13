import TextRecognition from '@react-native-ml-kit/text-recognition';
import { getModels, normalizeVector } from '../ai/modelManager';
import { db } from '../db/database';

export interface GalleryImage {
  id: string;
  uri: string;
  createdAt: number;
}

export const IngestionFunction = async (imageUri: string): Promise<GalleryImage> => {
    console.log("Starting real ingestion for:", imageUri);
    const createdAt = Date.now();

    try {
        // 1. Run Mobile OCR Extraction (Google ML Kit)
        const ocrResult = await TextRecognition.recognize(imageUri);
        const cleanText = ocrResult.text.replace(/\n/g, ' ').trim().toLowerCase();
        console.log(`OCR Found: ${cleanText.length > 0 ? 'Yes' : 'No'}`);

        // 2. Run Visual Embedding
        const { processor, visionModel } = await getModels();
        
        // Add timeout and error handling for model processing
        const imageInputs = await Promise.race([
            processor(imageUri),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Image processing timeout')), 30000)
            )
        ]);
        
        const { pooler_output } = await visionModel(imageInputs);

        // 3. Normalize the Vector
        const imageVector = new Float32Array(pooler_output.data);
        normalizeVector(imageVector);
        
        // Clean up model outputs to free memory
        pooler_output.data = null;
        
        // 4. Store as BLOB in SQLite (no native extensions needed)
        const vectorBlob = new Uint8Array(imageVector.buffer).buffer; 

        let rowId = 0;

        await db.transaction(async (tx) => {
            const docResult = tx.execute(
                'INSERT INTO documents (path, ocr_text, embedding, created_at) VALUES (?, ?, ?, ?)', 
                [imageUri, cleanText, vectorBlob, createdAt]
            );
            rowId = (await docResult).insertId!;
        });

        console.log(`✅ Successfully ingested and stored in DB!`);

        return {
            id: rowId.toString(),
            uri: imageUri,
            createdAt: createdAt,
        };

    } catch (error) {
        console.error(`❌ Failed to ingest ${imageUri}:`, error);
        
        // If model processing fails, still store the image with OCR text but no embedding
        try {
            const ocrResult = await TextRecognition.recognize(imageUri);
            const cleanText = ocrResult.text.replace(/\n/g, ' ').trim().toLowerCase();
            
            let rowId = 0;
            await db.transaction(async (tx) => {
                const docResult = tx.execute(
                    'INSERT INTO documents (path, ocr_text, embedding, created_at) VALUES (?, ?, ?, ?)', 
                    [imageUri, cleanText, null, createdAt] // null embedding
                );
                rowId = (await docResult).insertId!;
            });
            
            console.log(`⚠️ Stored image with OCR only (no embedding due to model error)`);
            return {
                id: rowId.toString(),
                uri: imageUri,
                createdAt: createdAt,
            };
        } catch (fallbackError) {
            console.error(`❌ Fallback storage also failed:`, fallbackError);
            throw error; // Throw original error
        }
    }
};
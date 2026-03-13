import { db } from '../db/database';
import { getModels, normalizeVector, cosineSimilarity } from '../ai/modelManager';

export interface GalleryImage {
  id: string;
  uri: string;
  createdAt: number;
}

export const RetrievalFunction = async (query: string, allImages: GalleryImage[] = []): Promise<GalleryImage[]> => {
    console.log("Starting Hybrid search for:", query);

    const cleanQuery = query.trim().toLowerCase();
    const finalResults: GalleryImage[] = [];
    const seenIds = new Set<string>();

    try {
        // 1. Handle Empty Queries — return all recent images
        if (!cleanQuery) {
            await db.transaction(async (tx) => {
                const result = await tx.execute(
                    'SELECT id, path as uri, created_at as createdAt FROM documents ORDER BY created_at DESC LIMIT 100'
                );
                const rows = result.rows || [];
                rows.forEach((row: any) => {
                    finalResults.push({ 
                        id: row.id.toString(), 
                        uri: row.uri, 
                        createdAt: row.createdAt 
                    });
                });
            });
            return finalResults;
        }

        // 2. Run AI text embedding OUTSIDE the database transaction
        const { tokenizer, textModel } = await getModels();
        
        const textInputs = await Promise.race([
            tokenizer([cleanQuery], { padding: 'max_length', truncation: true }),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Text tokenization timeout')), 10000)
            )
        ]);
        
        const { pooler_output } = await textModel(textInputs);
        
        const textVector = new Float32Array(pooler_output.data);
        normalizeVector(textVector); 
        
        // Clean up model outputs to free memory
        pooler_output.data = null; 

        // 3. Run Search INSIDE the transaction wrapper
        await db.transaction(async (tx) => {
            
            // A. Lexical Search: plain LIKE match on OCR text (no FTS5 needed)
            const textRes = await tx.execute(
                'SELECT id, path as uri, created_at as createdAt FROM documents WHERE ocr_text LIKE ?',
                [`%${cleanQuery}%`]
            );
            const textMatches = textRes.rows || [];

            for (const match of textMatches) {
                const idStr = match.id?.toString();
                if (idStr && !seenIds.has(idStr)) {
                    seenIds.add(idStr);
                    finalResults.push({ 
                        id: idStr, 
                        uri: match.uri as string, 
                        createdAt: match.createdAt as number 
                    });
                }
            }
            
            // B. Semantic JS Vector Search: load all embeddings, compute cosine similarity in JS
            const vectorSqlRes = await tx.execute(
                'SELECT id, path as uri, created_at as createdAt, embedding FROM documents'
            );
            const allDocs = vectorSqlRes.rows || [];
            
            const vectorMatches: { id: string; uri: string; createdAt: number; score: number }[] = [];
            
            for (const doc of allDocs) {
                const idStr = doc.id?.toString();
                // Skip docs already added via text match
                if (idStr && seenIds.has(idStr)) continue;

                if (doc.embedding) {
                   const docVec = new Float32Array(doc.embedding as ArrayBuffer);
                   const similarity = cosineSimilarity(textVector, docVec);
                   vectorMatches.push({
                        id: idStr!,
                        uri: doc.uri as string,
                        createdAt: doc.createdAt as number,
                        score: similarity
                   });
                }
            }
            
            // Sort by score descending, take top 50
            vectorMatches.sort((a, b) => b.score - a.score);
            const TOP_K = 50;
            for (let i = 0; i < Math.min(TOP_K, vectorMatches.length); i++) {
                const match = vectorMatches[i];
                if (!seenIds.has(match.id)) {
                    seenIds.add(match.id);
                    finalResults.push({
                        id: match.id,
                        uri: match.uri,
                        createdAt: match.createdAt
                    });
                }
            }
        });

        console.log(`✅ Search complete! Returning ${finalResults.length} results.`);
        return finalResults;

    } catch (error) {
        console.error("❌ Search failed:", error);
        
        // Fallback: return lexical search only if semantic search fails
        try {
            const fallbackResults: GalleryImage[] = [];
            await db.transaction(async (tx) => {
                const textRes = await tx.execute(
                    'SELECT id, path as uri, created_at as createdAt FROM documents WHERE ocr_text LIKE ? ORDER BY created_at DESC LIMIT 50',
                    [`%${cleanQuery}%`]
                );
                const textMatches = textRes.rows || [];
                textMatches.forEach((match: any) => {
                    fallbackResults.push({ 
                        id: match.id?.toString() || '', 
                        uri: match.uri as string, 
                        createdAt: match.createdAt as number 
                    });
                });
            });
            console.log(`⚠️ Using fallback lexical search. Found ${fallbackResults.length} results.`);
            return fallbackResults;
        } catch (fallbackError) {
            console.error("❌ Fallback search also failed:", fallbackError);
            return allImages; // Return original images as last resort
        }
    }
};
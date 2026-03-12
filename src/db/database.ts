import { open } from "@op-engineering/op-sqlite";

// Initialize the database connection
export const db = open({
  name: 'visionvault.sqlite',
});

export async function initDatabase(): Promise<void> {
  try {
    // Create the documents table (plain SQL - no FTS5 extension needed)
    await db.transaction(async (tx) => {
      tx.execute(`
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            path TEXT,
            ocr_text TEXT,
            embedding BLOB,
            created_at INTEGER
        );
      `);
    });
    
    console.log("✅ Database initialized successfully.");
  } catch (error) {
    console.error("❌ Database initialization failed:", error);
    throw error;
  }
}
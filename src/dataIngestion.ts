import { ElasticVectorSearch } from "@langchain/community/vectorstores/elasticsearch";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Client } from "@elastic/elasticsearch";
import { readFile } from "node:fs/promises";
import dotenv from "dotenv";

dotenv.config();

const VECTOR_INDEX = "test-index";

// Types
export interface DocumentMetadata {
  title?: string;
  airline?: string;
  from_city?: string;
  to_city?: string;
  airport_code?: string;
  airport_name?: string;
  price?: number;
  time_approx?: string;
  date?: string;
  [key: string]: any;
}

export interface Document {
  pageContent: string;
  metadata: DocumentMetadata;
}

interface RawDocument {
  pageContent?: string;
  text?: string;
  metadata?: DocumentMetadata;
}

// Initialize Elasticsearch client
export const esClient = new Client({
  node: process.env.ELASTICSEARCH_ENDPOINT!,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY!,
  },
});

// Initialize embeddings
export const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

// Initialize vector store
export const vectorStore = new ElasticVectorSearch(embeddings, {
  client: esClient,
  indexName: VECTOR_INDEX,
});

/**
 * Load dataset from a JSON file
 * @param path - Path to the JSON file
 * @returns Array of documents with pageContent and metadata
 */
export async function loadDataset(path: string): Promise<Document[]> {
  const raw = await readFile(path, "utf-8");
  const data: RawDocument[] = JSON.parse(raw);

  return data.map((d) => ({
    pageContent: String(d.pageContent ?? d.text ?? ""),
    metadata: d.metadata ?? {},
  }));
}

/**
 * Ingest data into Elasticsearch vector store
 * Creates the index if it doesn't exist and loads initial dataset
 */
export async function ingestData(): Promise<void> {
  const vectorExists = await esClient.indices.exists({ index: VECTOR_INDEX });

  if (!vectorExists) {
    console.log("CREATING VECTOR INDEX...");

    // Vector index mapping
    await esClient.indices.create({
      index: VECTOR_INDEX,
      mappings: {
        properties: {
          text: { type: "text" },
          embedding: {
            type: "dense_vector",
            dims: 1536,
            index: true,
            similarity: "cosine",
          },
          metadata: { type: "object", enabled: true },
        },
      },
    });
  }

  const indexExists = await esClient.indices.exists({ index: VECTOR_INDEX });

  if (indexExists) {
    const indexCount = await esClient.count({ index: VECTOR_INDEX });
    const documentCount = indexCount.count;

    // Only ingest if index is empty
    if (documentCount > 0) {
      console.log(
        `Index already contains ${documentCount} documents. Skipping ingestion.`
      );
      return;
    }

    console.log("INGESTING DATASET...");
    const datasetPath = "./dataset.json";
    const initialDocs = await loadDataset(datasetPath).catch(() => []);

    await vectorStore.addDocuments(initialDocs);
    console.log(`✅ Successfully ingested ${initialDocs.length} documents`);
  }
}

export { VECTOR_INDEX };

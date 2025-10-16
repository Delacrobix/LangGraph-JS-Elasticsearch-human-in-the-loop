import {
  StateGraph,
  Annotation,
  interrupt,
  Command,
  MemorySaver,
} from "@langchain/langgraph";
import { ElasticVectorSearch } from "@langchain/community/vectorstores/elasticsearch";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { Client } from "@elastic/elasticsearch";
import dotenv from "dotenv";
import fs from "node:fs/promises";
import { writeFileSync } from "node:fs";
import readline from "readline";

dotenv.config();

const VECTOR_INDEX = "test-index";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const esClient = new Client({
  node: process.env.ELASTICSEARCH_ENDPOINT,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY,
  },
});

const vectorStore = new ElasticVectorSearch(embeddings, {
  client: esClient,
  indexName: VECTOR_INDEX,
});

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

async function loadDataset(path) {
  const raw = await fs.readFile(path, "utf-8");
  const data = JSON.parse(raw);

  return data.map((d) => ({
    pageContent: String(d.pageContent ?? d.text ?? ""),
    metadata: d.metadata ?? {},
  }));
}

async function ingestData() {
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
    if (documentCount > 0) return;

    console.log("INGESTING DATASET...");
    const datasetPath = "./dataset.json";
    const initialDocs = await loadDataset(datasetPath).catch(() => []);

    await vectorStore.addDocuments(initialDocs);
  }
}

// Define the state schema for application workflow
const SupportState = Annotation.Root({
  input: Annotation(), // original user query
  candidates: Annotation(), // retrieved flight candidates
  userChoice: Annotation(), // user's selection
  selected: Annotation(), // chosen candidate doc
  final: Annotation(), // final answer
});

// Node 1: Retrieve data from Elasticsearch
async function retrieveFlights(state) {
  const results = await vectorStore.similaritySearch(state.input, 5);
  const candidates = [];

  for (const d of results) {
    candidates.push(d);
    if (candidates.length >= 2) break;
  }

  console.log(`📋 Found ${candidates.length} different flights`);
  return { candidates };
}

// Node 2: Evaluate if there are 1 or multiple responses
function evaluateResults(state) {
  const candidates = state.candidates || [];

  // If there 1 result, auto-select it
  if (candidates.length === 1) {
    const metadata = candidates[0].metadata || {};

    return {
      selected: candidates[0],
      final: `Selected flight: ${metadata.title} - ${metadata.airline}
      From: ${metadata.from_city} (${
        metadata.from_city?.slice(0, 3).toUpperCase() || "N/A"
      })
      To: ${metadata.to_city} (${metadata.airport_code})
      Airport: ${metadata.airport_name}
      Price: $${metadata.price}
      Duration: ${metadata.time_approx}
      Date: ${metadata.date}`,
    };
  }

  return { candidates };
}

// Node 3: Show results only
function showResults(state) {
  const candidates = state.candidates || [];
  const formattedOptions = candidates
    .map((d, index) => {
      const m = d.metadata || {};

      return `${index + 1}. ${m.title} - ${m.to_city} - ${m.airport_name}(${
        m.airport_code
      }) airport - ${m.airline} - $${m.price} - ${m.time_approx}`;
    })
    .join("\n");

  console.log(`\n📋 Flights found:\n${formattedOptions}\n`);

  return state;
}

// Node 4: Request user choice (separate from showing)
function requestUserChoice() {
  const userChoice = interrupt({
    question: `Which flight do you prefer?:`,
  });

  return { userChoice };
}

// Node 5: Disambiguate user choice and provide final answer
async function disambiguateAndAnswer(state) {
  const candidates = state.candidates || [];
  const userInput = state.userChoice || "";

  const prompt = `
    Based on the user's response: "${userInput}"

    These are the available flights:
    ${candidates
      .map(
        (d, i) =>
          `${i + 1}. ${d.metadata?.title} - ${d.metadata?.to_city} (${
            d.metadata?.airport_code
          }) - ${d.metadata?.airline} - $${d.metadata?.price} - ${
            d.metadata?.time_approx
          }`
      )
      .join("\n")}

      Which flight is the user selecting? Respond ONLY with the flight number (1, 2, or 3).
  `;

  const llmResponse = await llm.invoke([
    {
      role: "system",
      content:
        "You are an assistant that interprets user selection. Respond ONLY with the selected flight number.",
    },
    { role: "user", content: prompt },
  ]);

  const selectedNumber = parseInt(llmResponse.content.trim()) - 1;
  const selectedFlight =
    selectedNumber >= 0 && selectedNumber < candidates.length
      ? candidates[selectedNumber]
      : candidates[0];
  const metadata = selectedFlight.metadata || {};

  return {
    selected: selectedFlight,
    final: `Selected flight: ${metadata.title} - ${metadata.airline}
    From: ${metadata.from_city} (${
      metadata.from_city?.slice(0, 3).toUpperCase() || "N/A"
    })
    To: ${metadata.to_city} (${metadata.airport_code})
    Airport: ${metadata.airport_name}
    Price: $${metadata.price}
    Duration: ${metadata.time_approx}
    Date: ${metadata.date}`,
  };
}

// Build the graph
const workflow = new StateGraph(SupportState)
  .addNode("retrieveFlights", retrieveFlights)
  .addNode("evaluateResults", evaluateResults)
  .addNode("showResults", showResults)
  .addNode("requestUserChoice", requestUserChoice)
  .addNode("disambiguateAndAnswer", disambiguateAndAnswer)
  .addEdge("__start__", "retrieveFlights")
  .addEdge("retrieveFlights", "evaluateResults")
  .addConditionalEdges(
    "evaluateResults",
    (state) => {
      if (state.final) return "complete"; // 0 or 1 result
      return "multiple"; // multiple results
    },
    {
      complete: "__end__",
      multiple: "showResults",
    }
  )
  .addEdge("showResults", "requestUserChoice")
  .addEdge("requestUserChoice", "disambiguateAndAnswer")
  .addEdge("disambiguateAndAnswer", "__end__");

// Function to get user input from the command line
function getUserInput(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

async function saveGraphImage(app) {
  try {
    const graph = app.getGraph();
    const graphImage = await graph.drawMermaidPng();
    const graphArrayBuffer = await graphImage.arrayBuffer();

    const filePath = "./workflow_graph.png";
    writeFileSync(filePath, new Uint8Array(graphArrayBuffer));
    console.log(`📊 Workflow graph saved as: ${filePath}`);
  } catch (error) {
    console.log("⚠️  Could not save graph image:", error.message);
  }
}

await ingestData();
const app = workflow.compile({ checkpointer: new MemorySaver() });
const config = { configurable: { thread_id: "hitl-thread" } };

await saveGraphImage(app);

const question = "Flights to Tokyo"; // User query // TODO: make dynamic
console.log(`🔍 SEARCHING USER QUESTION: "${question}"\n`);

// Execute workflow
let currentState = await app.invoke({ input: question }, config);

// Handle interruption
if (currentState.__interrupt__ && currentState.__interrupt__.length > 0) {
  console.log("\n💭 APPLICATION PAUSED WAITING FOR USER INPUT...");
  const userChoice = await getUserInput("👤 CHOICE ONE OPTION: ");

  currentState = await app.invoke(new Command({ resume: userChoice }), config);
}

console.log("\n✅ Final result: \n", currentState.final);

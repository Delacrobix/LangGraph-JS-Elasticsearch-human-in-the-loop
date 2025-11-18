import {
  StateGraph,
  Annotation,
  interrupt,
  Command,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ElasticVectorSearch } from "@langchain/community/vectorstores/elasticsearch";
import { Client } from "@elastic/elasticsearch";
import { writeFileSync } from "node:fs";
import readline from "node:readline";
import { ingestData, Document } from "./dataIngestion.ts";

const VECTOR_INDEX = "flights-offerings";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});

const esClient = new Client({
  node: process.env.ELASTICSEARCH_ENDPOINT,
  auth: {
    apiKey: process.env.ELASTICSEARCH_API_KEY ?? "",
  },
});

const vectorStore = new ElasticVectorSearch(embeddings, {
  client: esClient,
  indexName: VECTOR_INDEX,
});

// Define the state schema for application workflow
const SupportState = Annotation.Root({
  input: Annotation<string>(),
  filteredFlights: Annotation<Document[]>(),
  options: Annotation<string[]>(),
  selectedCountry: Annotation<string>(),
  selectedCity: Annotation<string>(),
  selectedAirport: Annotation<string>(),
  userChoice: Annotation<string>(),
});

// Node 1: Initialize - Retrieve ALL flights from Elasticsearch
async function initialize(state: typeof SupportState.State) {
  const results = await vectorStore.similaritySearch(state.input, 10);
  const filteredFlights = results.map((d) => d as Document);

  console.log(`📋 Retrieved ${filteredFlights.length} flights from database\n`);
  return { filteredFlights };
}

// Node 2: Show available flights
function showFlights(state: typeof SupportState.State) {
  const flights = state.filteredFlights || [];

  // Check if we've narrowed down to exactly one flight
  if (flights.length === 1) {
    console.log("\n✅ Found unique flight!");
    return { filteredFlights: flights };
  }

  console.log("Available flights:\n");
  for (let i = 0; i < flights.length; i++) {
    const flight = flights[i];
    const m = flight.metadata;
    console.log(`${i + 1}. ${m.title} - ${m.airline}`);
    console.log(
      `   From: ${m.from_city} → To: ${m.to_city}, ${m.country} (${m.airport_name})`
    );
    console.log(
      `   Price: $${m.price} | Duration: ${m.time_approx} | Date: ${m.date}\n`
    );
  }

  return { filteredFlights: flights };
}

// Node 3: Request user refinement
function requestChoice(state: typeof SupportState.State) {
  const question =
    "Refine your search (e.g., 'I want flights to Japan', 'Show me the cheapest', or select by number):";
  const userChoice = interrupt({ question });
  return { userChoice };
}

// Node 4: Disambiguate and filter based on user input
async function disambiguateSelection(state: typeof SupportState.State) {
  const flights = state.filteredFlights || [];
  const userInput = state.userChoice || "";

  // Check if user selected by number
  const numberRegex = /^\d+$/;
  const numberMatch = numberRegex.exec(userInput);
  if (numberMatch) {
    const selectedIndex = Number.parseInt(userInput, 10) - 1;
    if (selectedIndex >= 0 && selectedIndex < flights.length) {
      console.log(`✅ Selected flight #${userInput}`);
      return { filteredFlights: [flights[selectedIndex]] };
    }
  }

  // Use LLM to interpret natural language refinement
  const flightsList = flights
    .map((f, i) => {
      const m = f.metadata;
      return `${i + 1}. ${m.title} - ${m.airline} | ${m.from_city} → ${
        m.to_city
      } (${m.country}) | ${m.airport_name} (${m.airport_code}) | $${m.price}`;
    })
    .join("\n");

  const prompt = `
    The user said: "${userInput}"

    These are the available flights:
    ${flightsList}

    Based on the user's request, which flight(s) match their criteria? 
    Respond with ONLY the flight numbers (e.g., "1" or "1,3,5" for multiple matches).
    If the user is asking to filter by criteria (country, city, price, etc.), return ALL matching flight numbers.
  `;

  const llmResponse = await llm.invoke([
    {
      role: "system",
      content:
        "You are an assistant that filters flights. Respond ONLY with flight numbers separated by commas.",
    },
    { role: "user", content: prompt },
  ]);

  const selectedNumbers = (llmResponse.content as string)
    .trim()
    .split(",")
    .map((n) => Number.parseInt(n.trim(), 10) - 1)
    .filter((i) => i >= 0 && i < flights.length);

  const filteredFlights = selectedNumbers.map((i) => flights[i]);

  console.log(`✅ Filtered to ${filteredFlights.length} flight(s)`);
  return { filteredFlights };
}

// Node 5: Show final flight details
function showFinalFlight(state: typeof SupportState.State) {
  const flights = state.filteredFlights || [];
  const flight = flights[0];

  if (!flight) {
    console.log("\n❌ No flight found matching your selection.");
    return {};
  }

  const m = flight.metadata;

  console.log(`
    ✅ Final result:
    Selected flight: ${m.title} - ${m.airline}
    From: ${m.from_city} (${m.from_city?.slice(0, 3).toUpperCase() || "N/A"})
    To: ${m.to_city} (${m.airport_code})
    Airport: ${m.airport_name}
    Price: $${m.price}
    Duration: ${m.time_approx}
    Date: ${m.date}
  `);

  return {};
}

// Build the circular workflow graph
const workflow = new StateGraph(SupportState)
  .addNode("initialize", initialize)
  .addNode("showFlights", showFlights)
  .addNode("requestChoice", requestChoice)
  .addNode("disambiguateSelection", disambiguateSelection)
  .addNode("showFinalFlight", showFinalFlight)
  .addEdge("__start__", "initialize")
  .addEdge("initialize", "showFlights")
  .addConditionalEdges(
    "showFlights",
    (state: typeof SupportState.State) => {
      // If we've narrowed down to one flight, show it
      const flights = state.filteredFlights || [];
      if (flights.length === 1) return "final";
      // Otherwise, continue the selection loop
      return "continue";
    },
    {
      final: "showFinalFlight",
      continue: "requestChoice",
    }
  )
  .addEdge("requestChoice", "disambiguateSelection")
  .addEdge("disambiguateSelection", "showFlights") // Loop back!
  .addEdge("showFinalFlight", "__end__"); // End after showing flight

/**
 * Get user input from the command line
 * @param question - Question to ask the user
 * @returns User's answer
 */
function getUserInput(question: string): Promise<string> {
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

/**
 * Save workflow graph as PNG image
 * @param app - Compiled workflow application
 */
async function saveGraphImage(app: any): Promise<void> {
  try {
    const graph = app.getGraph();
    const graphImage = await graph.drawMermaidPng();
    const graphArrayBuffer = await graphImage.arrayBuffer();

    const filePath = "./workflow_graph.png";
    writeFileSync(filePath, new Uint8Array(graphArrayBuffer));
    console.log(`📊 Workflow graph saved as: ${filePath}`);
  } catch (error) {
    console.log("⚠️  Could not save graph image:", (error as Error).message);
  }
}

/**
 * Main execution function
 */
async function main() {
  // Ingest data
  await ingestData();

  // Compile workflow
  const app = workflow.compile({ checkpointer: new MemorySaver() });
  const config = { configurable: { thread_id: "hitl-circular-thread" } };

  // Save graph image
  await saveGraphImage(app);

  // Execute workflow
  const question = "Flights to Asia"; // User query
  console.log(`🔍 SEARCHING USER QUESTION: "${question}"\n`);

  let currentState = await app.invoke({ input: question }, config);

  // Handle all interruptions in a loop
  while ((currentState as any).__interrupt__?.length > 0) {
    console.log("\n💭 APPLICATION PAUSED WAITING FOR USER INPUT...");
    const userChoice = await getUserInput("👤 CHOICE ONE OPTION: ");

    currentState = await app.invoke(
      new Command({ resume: userChoice }),
      config
    );
  }
}

// Run main function
await main();

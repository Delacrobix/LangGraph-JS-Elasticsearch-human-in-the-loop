import {
  StateGraph,
  Annotation,
  interrupt,
  Command,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { writeFileSync } from "node:fs";
import readline from "node:readline";
import {
  ingestData,
  vectorStore,
  Document,
  DocumentMetadata,
} from "./dataIngestion.js";

// Initialize LLM
const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

// Define the state schema for application workflow
interface SupportStateType {
  input: string;
  candidates?: Document[];
  userChoice?: string;
  selected?: Document;
  final?: string;
  __interrupt__?: any[];
}

const SupportState = Annotation.Root({
  input: Annotation<string>(),
  candidates: Annotation<Document[]>(),
  userChoice: Annotation<string>(),
  selected: Annotation<Document>(),
  final: Annotation<string>(),
});

// Node 1: Retrieve data from Elasticsearch
async function retrieveFlights(
  state: SupportStateType
): Promise<Partial<SupportStateType>> {
  const results = await vectorStore.similaritySearch(state.input, 5);
  const candidates: Document[] = [];

  for (const d of results) {
    candidates.push(d as Document);
    if (candidates.length >= 2) break;
  }

  console.log(`📋 Found ${candidates.length} different flights`);
  return { candidates };
}

// Node 2: Evaluate if there are 1 or multiple responses
function evaluateResults(state: SupportStateType): Partial<SupportStateType> {
  const candidates = state.candidates || [];

  // If there is 1 result, auto-select it
  if (candidates.length === 1) {
    const metadata = candidates[0].metadata || {};

    return {
      selected: candidates[0],
      final: formatFlightDetails(metadata),
    };
  }

  return { candidates };
}

// Node 3: Show results only
function showResults(state: SupportStateType): SupportStateType {
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
function requestUserChoice(): Partial<SupportStateType> {
  const userChoice = interrupt({
    question: `Which flight do you prefer?:`,
  });

  return { userChoice };
}

// Node 5: Disambiguate user choice and provide final answer
async function disambiguateAndAnswer(
  state: SupportStateType
): Promise<Partial<SupportStateType>> {
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

  const selectedNumber = Number.parseInt(llmResponse.content as string, 10) - 1;
  const selectedFlight =
    selectedNumber >= 0 && selectedNumber < candidates.length
      ? candidates[selectedNumber]
      : candidates[0];

  const metadata = selectedFlight.metadata || {};

  return {
    selected: selectedFlight,
    final: formatFlightDetails(metadata),
  };
}

// Helper function to format flight details
function formatFlightDetails(metadata: DocumentMetadata): string {
  return `Selected flight: ${metadata.title} - ${metadata.airline}
    From: ${metadata.from_city} (${
    metadata.from_city?.slice(0, 3).toUpperCase() || "N/A"
  })
    To: ${metadata.to_city} (${metadata.airport_code})
    Airport: ${metadata.airport_name}
    Price: $${metadata.price}
    Duration: ${metadata.time_approx}
    Date: ${metadata.date}`;
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
    (state: SupportStateType) => {
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
async function main(): Promise<void> {
  // Ingest data
  await ingestData();

  // Compile workflow
  const app = workflow.compile({ checkpointer: new MemorySaver() });
  const config = { configurable: { thread_id: "hitl-thread" } };

  // Save graph image
  await saveGraphImage(app);

  // Execute workflow
  const question = "Flights to Tokyo"; // User query
  console.log(`🔍 SEARCHING USER QUESTION: "${question}"\n`);

  let currentState = await app.invoke({ input: question }, config as any);

  // Handle interruption
  if (currentState.__interrupt__ && currentState.__interrupt__.length > 0) {
    console.log("\n💭 APPLICATION PAUSED WAITING FOR USER INPUT...");
    const userChoice = await getUserInput("👤 CHOICE ONE OPTION: ");

    currentState = await app.invoke(
      new Command({ resume: userChoice }),
      config as any
    );
  }

  console.log("\n✅ Final result: \n", currentState.final);
}

// Run main function
await main();

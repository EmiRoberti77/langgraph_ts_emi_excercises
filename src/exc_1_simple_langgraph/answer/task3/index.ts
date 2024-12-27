//imports for the chain
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import {
  START,
  END,
  StateGraph,
  MemorySaver,
  Annotation,
} from '@langchain/langgraph';
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import dotenv from 'dotenv';
dotenv.config();

const tavilyToolSchema = {
  // define schema
  name: 'Tavily tool',
  description: 'this tool is look up the latest news on the internet',
  schema: z.object({
    prompt: z
      .string()
      .describe(
        'this is the input prompt from the user of what they are looking for'
      ),
    context: z
      .string()
      .describe('this is the context to help answer the question'),
  }),
};

const refineToolSchema = {
  // define schema
  name: 'refine response',
  description: 'this tool is to refine the response from the chain of tools',
  schema: z.object({
    prompt: z
      .string()
      .describe(
        'this is the prompt to define what weather information we are looking for'
      ),
    context: z
      .string()
      .describe('this is the context to help answer the question'),
    messages: z
      .array(
        z.object({
          content: z.string().describe('The content of the AI message'),
          additional_kwargs: z.record(z.any()).optional(), // Metadata, optional
          response_metadata: z.record(z.any()).optional(), // Metadata, optional
          tool_calls: z.array(z.any()).optional(), // Tool calls, optional
          invalid_tool_calls: z.array(z.any()).optional(), // Invalid tool calls, optional
        })
      )
      .describe('State messages from previous tools or interactions'),
  }),
};

//create Annotation state
// * include messages
// * prompt
// * context
const GraphState = Annotation.Root({
  // the messages are an array of string that are collected
  // during the workflow of the langGraph
  messages: Annotation<AIMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  // this is the prompt that is passed
  // via the state and the tools
  // to find the best answer
  prompt: Annotation<string>(),
  // this is a extra context attribute to
  // to help refine the responses
  context: Annotation<string>(),
});

// adding a live search tool to extract information from news on the internet
// that the LLM are not trained on.  this is to get more relevant content within a chain
const tavilyTool = tool(async ({ prompt, context }) => {
  console.log('tavily tool');
  //make sure .env file has the correct api key to use the tool
  const tavilySearch = new TavilySearchResults({
    maxResults: 3,
  });
  const results = await tavilySearch.invoke(prompt);
  console.log(results);
  return {
    messages: [new AIMessage(JSON.stringify(results))],
  };
}, tavilyToolSchema);

// this tool is called at the end of the workflow to refine the answer
// the responses from the state messages are passed into a separate LLM
// to provide a more refined answer
const refineResponseTool = tool(async ({ messages, prompt, context }) => {
  console.log('refine tool ');
  const toolMessages = messages;
  const lastToolMessage = toolMessages[toolMessages.length - 1].content;
  //create a prompt template that has more content for the new LLM to respond with
  const refinedMessage = `context:${context}-prompt:${prompt}-lastMessage:${lastToolMessage}`;
  //console.log(refinedMessage);
  const llm = new ChatAnthropic({
    model: 'claude-3-5-sonnet-20240620',
    temperature: 0.7,
  });

  const response = await llm.invoke(refinedMessage);

  return {
    messages: [new AIMessage(response)],
  };
}, refineToolSchema);

function shouldContinue(state: typeof GraphState.State) {
  console.log('should continue');
  const messages = state.messages;
  console.log(messages);
  const lastMessage = messages[messages.length - 1] as AIMessage;
  if (lastMessage.tool_calls?.length) {
    console.log('tool call detected __news__');
    return '__news__';
  }
  console.log('no tool call detected');
  return END;
}

//create tool list
const tools = [tavilyTool, refineResponseTool];
const toolNode = new ToolNode<typeof GraphState.State>(tools);

//bind tool a llm
const model = new ChatOpenAI({
  temperature: 0.7,
  model: 'gpt-4o',
}).bindTools(tools);

//create langGraph workflow
const workflow = new StateGraph(GraphState)
  .addNode('__news__', tavilyTool)
  .addNode('refined', refineResponseTool)
  .addEdge(START, '__news__')
  .addConditionalEdges('__news__', shouldContinue)
  .addEdge('__news__', 'refined')
  .addEdge('refined', END);

//create memory check point and compile
const checkpointer = new MemorySaver();
const app = workflow.compile({ checkpointer });

//invoke and create prompt
async function runGraph() {
  const payload = {
    prompt: 'who won the 2024 Formula 1 championship',
    context: 'I am F1 fan and like to know technical details',
    messages: [new AIMessage('start the workflow')],
  };

  const finalState = await app.invoke(payload, {
    configurable: {
      thread_id: 'emi1',
    },
  });

  const response = finalState.messages[finalState.messages.length - 1].content;
  console.log(response);
}

runGraph();

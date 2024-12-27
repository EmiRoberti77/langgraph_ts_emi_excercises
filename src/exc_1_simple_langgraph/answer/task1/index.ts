//imports for the chain
import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import {
  START,
  END,
  StateGraph,
  MemorySaver,
  Annotation,
  messagesStateReducer,
} from '@langchain/langgraph';
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import dotenv from 'dotenv';
dotenv.config();

const weatherToolSchema = {
  // define schema
  name: 'weather lookup tool',
  description: 'this tool is to get up to date information about the weather',
  schema: z.object({
    prompt: z
      .string()
      .describe(
        'this is the prompt to define what weather information we are looking for'
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
      .array(z.string())
      .describe('State messages from previous tools or interactions'),
  }),
};

//create Annotation state
// - include messages
// - prompt
// - context
const GraphState = Annotation.Root({
  messages: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  prompt: Annotation<string>(),
  context: Annotation<string>(),
});

//create tool
const weatherTool = tool(async ({ prompt }) => {
  console.log('weather tool is called', prompt);
  let aiResponse;
  if (prompt.includes('Rome')) {
    aiResponse = JSON.stringify({
      weather: 'Rome',
      temperature: 32,
      date: '27-dec-2024',
      windSpeed: 4,
      condition: 'sunny',
    });
  } else {
    aiResponse = JSON.stringify({
      weather: 'Milan',
      temperature: 23,
      date: '27-dec-2024',
      windSpeed: 10,
      condition: 'cloudy',
    });
  }

  console.log('weather tool message:', [aiResponse]);
  return {
    messages: [aiResponse],
  };
}, weatherToolSchema);

const refineResponseTool = tool(async ({ messages, prompt, context }) => {
  console.log('refine tool ');
  const toolMessages = messages;
  const lastToolMessage = toolMessages[toolMessages.length - 1];
  //console.log('last tool massage', lastToolMessage);
  const refinedMessage = `context:${context}-prompt:${prompt}-lastMessage:${lastToolMessage}`;
  console.log(refinedMessage);
  const llm = new ChatAnthropic({
    model: 'claude-3-5-sonnet-20240620',
    temperature: 0.7,
  });
  const response = await llm.invoke(refinedMessage);
  console.log('refine tool message:', [response.content]);
  return {
    messages: [response.content],
  };
}, refineToolSchema);

//create tool list
const tools = [weatherTool, refineResponseTool];
const toolNode = new ToolNode<typeof GraphState.State>(tools);

//bind tool a llm
const model = new ChatOpenAI({
  temperature: 0.7,
  model: 'gpt-4o',
}).bindTools(tools);

//create langGraph workflow
const workflow = new StateGraph(GraphState)
  .addNode('weather', weatherTool)
  .addNode('refined', refineResponseTool)
  .addEdge(START, 'weather')
  .addEdge('weather', 'refined')
  .addEdge('refined', END);
//.addEdge('weather', END);

//create memory check point and compile
const checkpointer = new MemorySaver();
const app = workflow.compile({ checkpointer });

//invoke and create prompt
async function runGraph() {
  const payload = {
    prompt: 'what is the weather like in Rome today',
    context: 'today is the 27th Dec 2024',
    messages: [],
  };

  const finalState = await app.invoke(payload, {
    configurable: {
      thread_id: 'emi1',
    },
  });
  const response = finalState.messages[finalState.messages.length - 1];
  console.log(response);
}

runGraph();

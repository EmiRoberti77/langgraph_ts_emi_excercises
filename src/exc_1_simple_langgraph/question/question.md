## Exercises on using LangGraph

### task 1 - simple agent

Create a simple LangGraph workflow that is able
to access a simple tool, append data to State Annotation
and format response

### sample code to answer the task can be found here

https://github.com/EmiRoberti77/langgraph_ts_emi_excercises/blob/main/src/exc_1_simple_langgraph/answer/task1/index.ts

### task 2 - live search agent

Using the code base from task 1, lets remove the weather tool and replace it with a tool that can go online
and get up to date information that a LLM is not trained on yet. i.e the latest news in sports
I am using Tavily for this agent tool

### sample code to asnwer the task can be found here

https://github.com/EmiRoberti77/langgraph_ts_emi_excercises/blob/main/src/exc_1_simple_langgraph/answer/task2/index.ts

### task 3 - conditional workflows

Using the code base from task 2, lets build a workflow that decides the langGraph needs to call
and agent tool again or complete the flow. create a conditional node that is able to determine how to proceed

### sample code to asnwer the task can be found here

https://github.com/EmiRoberti77/langgraph_ts_emi_excercises/blob/main/src/exc_1_simple_langgraph/answer/task3/index.ts

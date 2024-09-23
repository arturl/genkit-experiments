package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	// Import the Genkit core libraries.
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	// Import the Google AI plugin.
	"github.com/firebase/genkit/go/plugins/googleai"

	"bufio"
	"encoding/json"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// Define tools/functions
func getWeather(location string) string {
	return fmt.Sprintf("The weather in %s is sunny.", location)
}

func getJoke(weather string) string {
	return fmt.Sprintf("Why don't %s clouds ever break up? Because they always stick together!", weather)
}

func runChat() {
	// Initialize OpenAI client
	// Read OpenAI key from environment:
	openaiKey := os.Getenv("OPENAI_API_KEY")
	client := openai.NewClient(openaiKey)
	ctx := context.Background()

	// Initialize scanner for user input
	scanner := bufio.NewScanner(os.Stdin)
	_ = scanner

	// Define initial system message
	messages := []openai.ChatCompletionMessage{
		{Role: "system", Content: "You are a helpful assistant. You can tell jokes about the weather."},
	}

	for {
		// Ask for user input
		fmt.Print("You: ")
		scanner.Scan()
		userInput := scanner.Text()

		//userInput := "Tell me a joke about the weather."

		// Break the loop if the user types "end"
		if strings.ToLower(userInput) == "end" {
			fmt.Println("Ending the chat. Goodbye!")
			break
		}

		// Add user input to the conversation
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    "user",
			Content: userInput,
		})

		for {
			// Call OpenAI Chat Completion with function calling
			resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
				Model:    "gpt-4-0613",
				Messages: messages,
				Functions: []openai.FunctionDefinition{
					{
						Name:        "get_weather",
						Description: "Get the current weather for a location.",
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"location": map[string]interface{}{
									"type":        "string",
									"description": "The location to get the weather for",
								},
							},
							"required": []string{"location"},
						},
					},
					{
						Name:        "get_joke",
						Description: "Tell a joke based on the weather.",
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"weather": map[string]interface{}{
									"type":        "string",
									"description": "Weather description",
								},
							},
							"required": []string{"weather"},
						},
					},
				},
				FunctionCall: "auto",
			})

			if err != nil {
				log.Fatalf("Error calling OpenAI API: %v", err)
			}

			message := resp.Choices[0].Message

			// If the model returns a function call, process it
			if message.FunctionCall != nil {
				functionName := message.FunctionCall.Name
				var args map[string]string
				err := json.Unmarshal([]byte(message.FunctionCall.Arguments), &args)
				if err != nil {
					log.Fatalf("Error parsing function arguments: %v", err)
				}

				// Handle different tool functions
				switch functionName {
				case "get_weather":
					location := args["location"]
					weather := getWeather(location)
					messages = append(messages, openai.ChatCompletionMessage{
						Role:    "function",
						Name:    "get_weather",
						Content: weather,
					})
				case "get_joke":
					weather := args["weather"]
					joke := getJoke(weather)
					messages = append(messages, openai.ChatCompletionMessage{
						Role:    "function",
						Name:    "get_joke",
						Content: joke,
					})
				}
			} else {
				// No more function calls, print the final response
				messages = append(messages, message)
				fmt.Printf("Assistant: %s\n", message.Content)
				break
			}
		}
	}
}

func getJoke2(ctx context.Context) string {
	model := googleai.Model("gemini-1.5-flash")

	myJokeTool := ai.DefineTool(
		"myJoke",
		"useful when you need a joke to tell",
		func(ctx context.Context, input *any) (string, error) {
			responseText, err := ai.GenerateText(ctx, model, ai.WithTextPrompt("Tell me a joke about ice cream."))
			if err != nil {
				return "error", err
			}
			return responseText, nil
		},
	)

	response, _ := ai.Generate(ctx, model,
		ai.WithTextPrompt("Tell me a joke."),
		ai.WithTools(myJokeTool))

	text := response.Text()
	return text
}

func toolLoop(ctx context.Context) string {
	model := googleai.Model("gemini-1.5-flash")

	jokeTool := ai.DefineTool(
		"jokeTool",
		"call this to tell a joke about weather. Input describes the weather",
		func(ctx context.Context, input string) (string, error) {
			return fmt.Sprintf("Ha-ha, this is a joke about %s", input), nil
		},
	)
	_ = jokeTool

	weatherTool := ai.DefineTool(
		"weatherTool",
		"call this to get the current weather",
		func(ctx context.Context, input *any) (string, error) {
			return "It snows", nil
		},
	)
	_ = weatherTool

	msg1 := ai.Message{
		Content: []*ai.Part{ai.NewTextPart("Tell me a joke about the current weather")},
		Role:    ai.RoleUser,
	}
	msg2 := ai.Message{
		Content: []*ai.Part{ai.NewTextPart("Use weatherTool to get current weather")},
		Role:    ai.RoleSystem,
	}

	response, err := ai.Generate(ctx, model,
		ai.WithMessages(&msg1, &msg2),
		ai.WithTools(jokeTool, weatherTool),
	)

	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	text := response.Text()
	fmt.Printf("Result:\n%s", text)
	return text
}

func test01() {
	model := googleai.Model("gemini-1.5-flash")
	prompt := "What is a lemon"
	history := []*ai.Message{{
		Content: []*ai.Part{ai.NewTextPart(prompt)},
		Role:    ai.RoleUser,
	}}

	h1 := history[0]
	fmt.Println(h1.Content)

	response, _ := ai.Generate(context.Background(), model, ai.WithMessages(history...))
	text2 := response.Text()
	fmt.Println(text2)
}

func testFlows(ctx context.Context) {
	// Define a simple flow that prompts an LLM to generate menu suggestions.
	flow1 := genkit.DefineFlow("Tell joke", func(ctx context.Context, input string) (string, error) {
		text := getJoke2(ctx)
		return text, nil
	})

	// Define a simple flow that prompts an LLM to generate menu suggestions.
	flow2 := genkit.DefineFlow("People Encyclopedia", func(ctx context.Context, input string) (string, error) {
		// The Google AI API provides access to several generative models. Here,
		// we specify gemini-1.5-flash.
		m := googleai.Model("gemini-1.5-flash")
		if m == nil {
			return "", errors.New("People Encyclopedia: failed to find model")
		}

		// Construct a request and send it to the model API.
		resp, err := m.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage(fmt.Sprintf(`Who is %s`, input))),
			nil)
		if err != nil {
			return "", err
		}

		// Handle the response from the model API. In this sample, we just
		// convert it to a string, but more complicated flows might coerce the
		// response into structured output or chain the response into another
		// LLM call, etc.
		text := resp.Text()
		return text, nil
	})

	suggestion, _ := flow1.Run(context.Background(), "")
	fmt.Println(suggestion)

	_ = flow1
	_ = flow2

	// Initialize Genkit and start a flow server. This call must come last,
	// after all of your plug-in configuration and flow definitions. When you
	// pass a nil configuration to Init, Genkit starts a local flow server,
	// which you can interact with using the developer UI.
	if err := genkit.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}
}

func main() {
	//ctx := context.Background()

	runChat()

	// if err := googleai.Init(ctx, nil); err != nil {
	// 	log.Fatal(err)
	// }

	//text := getJoke(ctx)
	//fmt.Println(text)

	//test01()

	//toolLoop(ctx)

	fmt.Println("Done!")

}

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "@/components/chat-message";
import { ChatOption } from "@/components/chat-option";
import { TypingIndicator } from "@/components/typing-indicator";
import { SendIcon } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/lib/api";

type Message = {
  text: string;
  isUser: boolean;
};

const INITIAL_MESSAGES: Message[] = [
  {
    text: "Hello! I'm Agro Insight AI, your intelligent farming assistant. How can I help you today? Feel free to ask me about crop recommendations, yield predictions, fertilizers, plant diseases, or any general farming questions.",
    isUser: false,
  },
];

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const chatOptions = [
    { label: "Crop Recommendation", prompt: "What crops should I plant in my farm given the soil type and climate?" },
    { label: "Yield Prediction", prompt: "Can you predict the yield for my maize crop this season?" },
    { label: "Fertilizer Recommendation", prompt: "What fertilizers should I use for my tomato plants?" },
    { label: "Disease Detection", prompt: "How can I identify and treat common diseases in rice crops?" },
    { label: "General Questions", prompt: "What are sustainable farming practices I can implement?" },
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (messageText: string) => {
    console.log("handleSendMessage called with:", messageText);
    
    if (!messageText || typeof messageText !== 'string') {
      console.error("Invalid message format:", messageText);
      return;
    }

    const trimmedMessage = messageText.trim();
    if (!trimmedMessage) {
      console.error("Empty message after trimming");
      return;
    }

    // Add user message
    const userMessage: Message = { text: trimmedMessage, isUser: true };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    try {
      console.log("Preparing to send message to API:", trimmedMessage);
      const response = await api.chat(trimmedMessage);
      console.log("API Response:", response);

      if (response.error) {
        throw new Error(response.error);
      }

      // Check if the response is asking for parameters
      if (response.message.includes("To recommend the best crop")) {
        const botMessage: Message = {
          text: response.message,
          isUser: false,
        };
        setMessages((prev) => [...prev, botMessage]);
        return;
      }

      // If we have parameters in the message, try to extract them
      const params = extractCropParameters(trimmedMessage);
      if (params) {
        console.log("Extracted crop parameters:", params);
        const cropResponse = await api.cropRecommendation(params);
        console.log("Crop recommendation response:", cropResponse);
        
        if (cropResponse.error) {
          throw new Error(cropResponse.error);
        }
        
        const botMessage: Message = {
          text: cropResponse.message,
          isUser: false,
        };
        setMessages((prev) => [...prev, botMessage]);
        return;
      }

      // If no parameters were found, just show the chat response
      const botMessage: Message = {
        text: response.message,
        isUser: false,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error in handleSendMessage:", error);
      let errorMessage = "An error occurred while processing your request.";
      
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      }
      
      console.error("Error message to display:", errorMessage);
      
      const botMessage: Message = {
        text: `I'm sorry, there was an error: ${errorMessage}. Please try again.`,
        isUser: false,
      };
      setMessages((prev) => [...prev, botMessage]);
      
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsTyping(false);
    }
  };

  // Helper function to extract crop parameters from user input
  const extractCropParameters = (message: string) => {
    const params: {
      N: number;
      P: number;
      K: number;
      temperature: number;
      humidity: number;
      ph: number;
      rainfall: number;
    } = {
      N: 0,
      P: 0,
      K: 0,
      temperature: 0,
      humidity: 0,
      ph: 0,
      rainfall: 0
    };
    
    const paramRegex = /(N|P|K|temp|humidity|ph|rainfall)\s*=\s*(\d+(?:\.\d+)?)/gi;
    let match;
    
    while ((match = paramRegex.exec(message)) !== null) {
      const key = match[1].toLowerCase();
      const value = parseFloat(match[2]);
      
      switch (key) {
        case 'n':
          params.N = value;
          break;
        case 'p':
          params.P = value;
          break;
        case 'k':
          params.K = value;
          break;
        case 'temp':
          params.temperature = value;
          break;
        case 'humidity':
          params.humidity = value;
          break;
        case 'ph':
          params.ph = value;
          break;
        case 'rainfall':
          params.rainfall = value;
          break;
      }
    }

    // Check if all required parameters are present
    const requiredParams = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
    const missingParams = requiredParams.filter(param => !params[param as keyof typeof params]);
    
    if (missingParams.length > 0) {
      return null;
    }

    return params;
  };

  const handleOptionClick = (prompt: string) => {
    handleSendMessage(prompt);
    toast({
      title: "Option selected",
      description: `You selected: ${prompt}`,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(inputValue);
  };

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 px-4 py-2">
        <div className="max-w-3xl mx-auto">
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message.text}
              isUser={message.isUser}
            />
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div className="border-t bg-background p-3">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-2 mb-3">
            <Input
              placeholder="Type your message here..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" className="bg-primary hover:bg-primary/90">
              <SendIcon className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </form>

          <div className="flex flex-wrap gap-2 justify-center">
            {chatOptions.map((option, index) => (
              <ChatOption
                key={index}
                label={option.label}
                onClick={() => handleOptionClick(option.prompt)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

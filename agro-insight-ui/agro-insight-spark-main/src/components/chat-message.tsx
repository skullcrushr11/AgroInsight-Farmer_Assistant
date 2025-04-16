
import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";

type ChatMessageProps = {
  message: string;
  isUser: boolean;
};

export function ChatMessage({ message, isUser }: ChatMessageProps) {
  return (
    <div className={cn("flex items-start gap-2 py-2", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-primary text-primary-foreground">
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div className={cn(
        "chat-bubble",
        isUser ? "user-message" : "bot-message"
      )}>
        {message}
      </div>
      {isUser && (
        <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-background text-foreground">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}


export function TypingIndicator() {
  return (
    <div className="flex items-start gap-2 py-1">
      <div className="typing-indicator ml-10 chat-bubble bot-message">
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
      </div>
    </div>
  );
}

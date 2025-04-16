
import { Button } from "@/components/ui/button";

type ChatOptionProps = {
  label: string;
  onClick: () => void;
};

export function ChatOption({ label, onClick }: ChatOptionProps) {
  return (
    <Button 
      variant="outline" 
      size="sm"
      className="border-primary/20 hover:bg-primary/10 hover:text-primary-foreground/90 transition-colors"
      onClick={onClick}
    >
      {label}
    </Button>
  );
}

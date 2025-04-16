import React from 'react';
import { ChatInterface } from "@/components/chat-interface";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={
              <ThemeProvider defaultTheme="light">
                <div className="h-screen flex flex-col overflow-hidden">
                  <header className="h-16 flex items-center justify-between px-4 bg-background/80 backdrop-blur-sm border-b">
                    <div className="flex items-center">
                      <div className="flex items-center gap-2">
                        <div className="h-8 w-8 rounded-full bg-primary"></div>
                        <h1 className="text-xl font-bold text-primary">Agro Insight</h1>
                      </div>
                    </div>
                    <ThemeToggle />
                  </header>
                  <main className="flex-1 overflow-hidden">
                    <ChatInterface />
                  </main>
                </div>
              </ThemeProvider>
            } />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;

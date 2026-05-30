"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";

type Theme = "dark" | "light";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | null>(null);

export function useTheme(): ThemeContextType {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  // On mount: read saved theme from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("datapilot-theme");
      if (saved === "light") {
        setTheme("light");
        document.documentElement.setAttribute("data-theme", "light");
      } else {
        setTheme("dark");
        document.documentElement.removeAttribute("data-theme");
      }
    } catch {}
    setMounted(true);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      try {
        if (next === "light") {
          document.documentElement.setAttribute("data-theme", "light");
          localStorage.setItem("datapilot-theme", "light");
        } else {
          document.documentElement.removeAttribute("data-theme");
          localStorage.setItem("datapilot-theme", "dark");
        }
      } catch {}
      return next;
    });
  }, []);

  // Prevent flash: don't render until mounted
  if (!mounted) return null;

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

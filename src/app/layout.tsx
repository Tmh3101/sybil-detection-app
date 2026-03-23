import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { TopHeader } from "@/components/layout/top-header";

import { ThemeProvider } from "@/store/theme-store";
import QueryProvider from "@/providers/query-provider";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "SYBIL ENGINE | INDUSTRIAL DASHBOARD",
  description: "Autonomous Sybil Detection & Discovery Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${jetbrainsMono.variable} antialiased font-mono bg-background text-foreground overflow-hidden h-screen w-screen selection:bg-accent-cyan selection:text-black transition-colors duration-300`}
      >
        <QueryProvider>
          <ThemeProvider>
            <div className="flex h-full w-full relative">
              {/* Global Background Scanline/Grid Effect */}
              <div className="absolute inset-0 pointer-events-none opacity-[0.03] z-[9999] bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%]" />
              
              <Sidebar />
              
              <div className="flex flex-col flex-1 overflow-hidden">
                <TopHeader />
                <main className="flex-1 overflow-y-auto p-10 relative scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent">
                  {/* Subtle radial gradient to focus content */}
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_var(--background)_100%)] pointer-events-none opacity-40 transition-colors duration-300" />
                  
                  <div className="relative z-10 h-full max-w-[1600px] mx-auto">
                    {children}
                  </div>
                </main>
              </div>
            </div>
          </ThemeProvider>
        </QueryProvider>
      </body>
    </html>
  );
}

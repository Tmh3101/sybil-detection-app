import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "sonner";
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
        className={`${jetbrainsMono.variable} bg-background text-foreground selection:bg-accent-cyan h-screen w-screen overflow-hidden font-mono antialiased transition-colors duration-300 selection:text-black`}
      >
        <QueryProvider>
          <ThemeProvider>
            <div className="relative flex h-full w-full">
              {/* Global Background Scanline/Grid Effect */}
              <div className="pointer-events-none absolute inset-0 z-[9999] bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] opacity-[0.03]" />

              <Sidebar />

              <div className="flex flex-1 flex-col overflow-hidden">
                <TopHeader />
                <main className="scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent relative flex-1 overflow-y-auto p-10">
                  {/* Subtle radial gradient to focus content */}
                  <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_var(--background)_100%)] opacity-40 transition-colors duration-300" />

                  <div className="relative z-10 mx-auto h-full max-w-[1600px]">
                    {children}
                  </div>
                </main>
              </div>
            </div>
            <Toaster
              theme="dark"
              toastOptions={{
                className:
                  "bg-slate-950 border border-slate-800 text-accent-cyan font-mono rounded-none shadow-[4px_4px_0px_0px_rgba(0,0,0,0.3)]",
              }}
            />
          </ThemeProvider>
        </QueryProvider>
      </body>
    </html>
  );
}

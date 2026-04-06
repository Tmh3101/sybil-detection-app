import "../globals.css";
import { Toaster } from "sonner";
import { Sidebar } from "@/components/layout/sidebar";

import { ThemeProvider } from "@/store/theme-store";
import QueryProvider from "@/providers/query-provider";
import { NextIntlClientProvider } from "next-intl";
import { getMessages, getTranslations } from "next-intl/server";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "Metadata" });

  return {
    title: t("title"),
    description: t("description"),
  };
}

export default async function RootLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}>) {
  const { locale } = await params;
  const messages = await getMessages();

  return (
    <html lang={locale} suppressHydrationWarning>
      <body
        className="bg-background text-foreground selection:bg-accent-cyan h-screen w-screen overflow-hidden font-mono antialiased transition-colors duration-300 selection:text-black"
        suppressHydrationWarning
      >
        <NextIntlClientProvider messages={messages}>
          <QueryProvider>
            <ThemeProvider>
              <div className="relative flex h-full w-full">
                {/* Global Background Scanline/Grid Effect */}
                <div className="pointer-events-none absolute inset-0 z-[9999] bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] opacity-[0.03]" />

                <Sidebar />

                <main className="scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent relative flex-1 overflow-y-auto p-10">
                  {/* Subtle radial gradient to focus content */}
                  <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_var(--background)_100%)] opacity-40 transition-colors duration-300" />

                  <div className="relative z-10 mx-auto h-full max-w-[1600px]">
                    {children}
                  </div>
                </main>
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
        </NextIntlClientProvider>
      </body>
    </html>
  );
}

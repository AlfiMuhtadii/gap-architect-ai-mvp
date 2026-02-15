import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "gap-architect",
  description: "gap-architect"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="border-b px-6 py-4">
            <div className="text-lg font-semibold">gap-architect</div>
          </header>
          <main className="px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}

export default function RootLayout({
    children,
  }: Readonly<{
    children: React.ReactNode;
  }>) {
    return (
      <>
        <div className="flex items-center justify-center h-screen bg-neutral-950">
          {children}
        </div>
      </>
    );
  }